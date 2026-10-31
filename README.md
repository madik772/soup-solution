# Soup Kernel Challenge — a packed, recomputing QLoRA step with our own Triton kernels

`soup train` (soup-cli 0.77.0) fine-tunes Qwen2.5-1.5B-Instruct (NF4 4-bit with double quant,
LoRA r=16 / alpha=32 / dropout 0.05 on `q_proj` and `v_proj`) for 64 steps on a Kaggle T4.
`solution.apply()` replaces the training step with one that does the same math on the real
tokens only, keeps almost nothing for the backward pass, and never builds the
`[batch, seq, 151936]` logits. Most of the work that is not a cuBLAS GEMM or SDPA runs in our
own `@triton.jit` kernels in `solution/`.

**Public Kaggle notebook:** TODO — https://www.kaggle.com/code/madiyarzhumazhan/notebook0b34895d24
(make it public, set `COMMIT`, Save & Run All, paste the version link here and in `SUBMISSION.md`).

## Results (Kaggle T4, `harness/run_task.py`, one base and one solution run per session)

| | Base Soup | Ours | |
|---|---|---|---|
| Whole `soup train` command | 90.8 s / 78.2 s | 61.7 s / 57.8 s | **1.47x / 1.35x faster** |
| of which the 64 training steps | 56.4 s / 55.9 s | 32.2 s / 32.8 s | 1.75x / 1.70x |
| Median step | 0.893 s / 0.884 s | 0.489 s / 0.495 s | 1.8x |
| Peak GPU memory (nvidia-smi) | 14,345 MiB | 1,779 MiB | **8.06x less** |
| Held-out eval loss (public rows) | 2.04970 / 2.04966 | 2.04955 / 2.04953 | −0.007 % / −0.006 % |
| Loss curve vs base | | first step 0.00 %, mean 0.05 % | |
| Our kernels' share of training time (check 4) | | 16.3 % / 15.9 % | |

Two Kaggle sessions (commit 62c1000 / commit 6974411, the same step). All 10 checks of
`harness/check_rules.py` pass in both, including the 79 Soup regression tests with the patches
applied. The base command's time varies a lot between sessions (78–91 s for us, mostly outside
the training steps); the ratio is taken within one session.

Where a step's GPU time goes now (torch.profiler, ~426 ms per step): cuBLAS GEMMs 68 % (about
22 TFLOPS, close to what a T4 sustains), SDPA attention 12 %, NF4 dequantization 7 %, the other
kernels of ours ~10 %. About 23 s of the command is Soup's own startup, model loading and
saving, the same as for the base run.

## What we did

1. **Packed tokens end to end** (`solution/fast_qwen2.py`). TRL pads each batch on the right.
   Right after the embedding the real tokens (attention_mask = 1) are gathered into one
   `[N, 1536]` matrix and stay packed through all 28 layers and the loss. Every projection, norm
   and activation runs on N rows instead of B·S (a step has ~800 real tokens). Only SDPA sees
   padded q/k/v; with right padding the causal mask is exact for every real token.
2. **One autograd Function per decoder layer, with recomputation.** The forward keeps only the
   layer's fp32 input (the residual stream: 6 KB per token and layer). The backward recomputes
   the layer, but not `down_proj`, whose input gradient needs only the weight, and it reuses the
   weights it dequantizes for the recomputation in its own GEMMs: each frozen weight is
   dequantized twice per step (forward, backward), not three times. This is what takes the peak
   from 14 GB to 1.8 GB.
3. **Fused projections.** q|k|v and gate|up are one GEMM each over a jointly dequantized weight;
   the LoRA update is added in the GEMM epilogue (`addmm_(…, alpha=scaling)`); one GEMM takes
   each fused gradient back to the layer input.
4. **Own NF4 dequantization** (`solution/nf4.py`). bitsandbytes' layout (two 4-bit codes per
   byte, absmax per 64 values, absmax itself 8-bit over a 256-entry map with one fp32 scale per
   256 blocks plus an offset), with the same fp32 arithmetic as bitsandbytes, so the GEMMs see
   bit-identical weights (`solution/test_fast_path.py` checks it against
   `bitsandbytes.functional.dequantize_4bit`).
5. **LoRA dropout without stored masks.** peft applies dropout to the input norm's output before
   `lora_A`. The input-norm kernel writes the normalized activations and the two dropped copies
   (q and v adapters) in one pass; the masks come from a Philox counter (seed, element index), so
   the recomputing backward and the norm's backward kernel regenerate them.
6. **RoPE + GQA + padding in one kernel** (`solution/rope.py`): rotates q and k in fp32 (as
   transformers does), repeats the two KV heads for the twelve query heads, writes the padded
   layout SDPA reads; its backward undoes all of it (inverse rotation, sum over the repeated
   heads, back to packed) in one pass per token.
7. **Head + loss + metrics in one pass** (`solution/cross_entropy.py`). Packed tokens, labelled
   ones first, in chunks of 512: one GEMM gives fp16 logits; `_custom_lmhead_ce_kernel` reads each
   row once (online max, sum-exp and sum(exp·x), argmax) and writes log-sum-exp, the token loss,
   TRL's per-token entropy and the argmax hit; for labelled rows it then overwrites the logits
   with `(softmax − onehot) · 1024`, the gradient, and a second GEMM over the labelled rows only
   takes it back to the hidden states — in the forward pass. The backward only rescales by the
   incoming loss scale. Prompt tokens never enter a gradient GEMM, and the head is never
   recomputed.
8. **Soup's outputs from the fused pass.** TRL's `entropy`, `num_tokens` and
   `mean_token_accuracy` are logged by TRL's formulas; Soup's rewind recorder gets the per-token
   losses through the interface Soup itself defines for steps without logits
   (`trainer._soup_chunked_step.token_loss`, see `soup_cli/utils/stream_loss_chunk.py`).
9. **The frozen tied embedding stays fp16.** `prepare_model_for_kbit_training` would replace it
   by a 933 MB fp32 copy. The model is loaded in fp16 and the head multiplies in fp16 anyway, so
   we keep the fp16 tensor — the very values the fp32 copy would hold — and the lookup writes
   fp32 rows into the residual stream as before.
10. **Compilation off the critical path** (`solution/warmup.py`). Compile time counts, and every
    run starts with empty caches: three threads compile the step's kernel variants (and build
    their launchers) while Soup loads the data and the model. The first step waits 0 s.

Anything outside what this path covers (another architecture, LoRA on o/MLP projections, DoRA,
several adapters, left-padded or packed batches, a batch without labelled tokens) is detected
before the step and goes through transformers' own code, which our older kernels (RMSNorm,
SwiGLU, RoPE, cross-entropy, unpadded MLP) still speed up.

### Our kernels

| Kernel | File | What it does |
|---|---|---|
| `_custom_nf4_dequant_kernel` | `nf4.py` | NF4 + double-quant dequantization into fused weight groups |
| `_custom_rmsnorm_drop_fwd_kernel` / `_bwd_kernel` | `rmsnorm.py` | input RMSNorm with the LoRA dropout masks (Philox) fused in |
| `_custom_fused_add_rmsnorm_fwd_kernel` / `_bwd_kernel` | `rmsnorm.py` | residual add + post-attention RMSNorm |
| `_custom_rmsnorm_fwd_kernel` / `_bwd_kernel` | `rmsnorm.py` | final norm |
| `_custom_qkv_rope_pad_fwd_kernel` / `_bwd_kernel` | `rope.py` | RoPE + KV head repeat + padding for SDPA, and back |
| `_custom_fused_swiglu_fwd_kernel` / `_bwd_kernel` | `swiglu.py` | SiLU(gate)·up on the fused gate|up output; backward in place |
| `_custom_row_gather_kernel` | `unpad.py` | unpad, pad (one pass, zero fill), embedding lookup, scaled gradient scatter |
| `_custom_lmhead_ce_kernel` | `cross_entropy.py` | log-sum-exp, loss, entropy, argmax and in-place gradient per row |

The remaining kernels in `solution/` (`_custom_ce_*`, `_custom_fused_rope_*`) serve the fallback
path only.

### Numerics

The same dequantized weights as bitsandbytes; the residual stream in fp32; RMSNorm, RoPE and
softmax statistics in fp32; fp16 GEMMs with fp32 accumulation — the precisions of the base step.
The loss gradient is formed in fp32 and rounded to fp16 once, scaled by 1024 so that small
probabilities survive (the role GradScaler's scale plays in the base step). LoRA dropout draws
from its own Philox stream (seeded from the run's seed), so its masks differ from PyTorch's the
way any other seed's would. The packed step matches transformers + peft + bitsandbytes on a small
random Qwen2 (`test_packed_step_matches_transformers`: loss, LoRA gradients, entropy, rewind
losses).

Ideas we built on, with our own code: chunked / fused linear cross-entropy as in Liger-Kernel and
Apple's Cut Cross-Entropy; NF4 / double quant as in QLoRA (bitsandbytes); padding-free training
and layer recomputation (gradient checkpointing) as in Unsloth.

## How to run

Kaggle: **Create → New Notebook → File → Import Notebook**, `templates/kaggle_notebook.ipynb`,
Accelerator GPU T4 x2, Internet on, put this repository in the first cell, Run All (~12 min).

Locally on Linux with a CUDA GPU:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q            # kernels and the packed step vs transformers + peft + bnb
python harness/run_task.py              # base, then solution -> runs/result.json
python harness/check_rules.py           # the 10 checks -> runs/checks.json
python harness/score.py runs/result.json
```

`python run_remote.py` (our helper) pushes the repository and runs the notebook on Kaggle
through its API; it needs your own Kaggle credentials in `~/.kaggle/kaggle.json`.

## Limits and next steps

- **Speed vs memory.** Recomputation costs ~25 % more GEMM work in the backward pass (~85 ms
  of a 0.5 s step). Keeping each layer's gate|up output and residual sum instead would buy most
  of that back for ~60 MB per layer at the longest batch; we chose the memory result.
- **Fixed cost.** 25–30 s of our ~60 s are Soup's imports, model loading and saving, untouched
  (the base command spends the same there).
- **Attention** is PyTorch's memory-efficient SDPA on padded q/k/v. A varlen kernel would skip
  the padding, but on a T4 per-sequence calls leave most SMs idle; a Triton flash kernel was not
  faster for us earlier.
- **Scope.** The packed path is for Qwen2-style decoders with bitsandbytes NF4 weights and LoRA on
  q/k/v, the shape of this task; it has been measured on the T4 only (sm_75: fp16 tensor cores,
  `num_stages=2`). Everything else falls back to the standard path.

## Rules of the challenge

The original task description, rules and checks are in the starter kit:
https://github.com/MakazhanAlpamys/soup-kernel-challenge (`task/`, `harness/` and `tests/` here
are unchanged copies). License: Apache-2.0, see `LICENSE` and `NOTICE`.

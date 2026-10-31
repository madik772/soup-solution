# Soup Kernel Challenge — a packed, recomputing QLoRA step with our own Triton kernels

`soup train` (soup-cli 0.77.0) fine-tunes Qwen2.5-1.5B-Instruct (NF4 4-bit with double quant,
LoRA r=16 / alpha=32 / dropout 0.05 on `q_proj` and `v_proj`) for 64 steps on a Kaggle T4.
`solution.apply()` replaces the training step with one that does the same math on the real
tokens only, keeps little for the backward pass, and never builds the `[batch, seq, 151936]`
logits. Most of the work that is not a cuBLAS GEMM or attention runs in our own `@triton.jit`
kernels in `solution/`.

**Public Kaggle notebook:** https://www.kaggle.com/code/madiyarzhumazhan/notebook0b34895d24
(the saved version with `COMMIT` pinned is linked in `SUBMISSION.md`).

## Results (Kaggle T4, `harness/run_task.py`, base and solution in the same session)

| | Base Soup | Ours | |
|---|---|---|---|
| Whole `soup train` command | 87.4 s | 50.8 s | **1.72x faster** |
| of which the 64 training steps | 55.4 s | 27.5 s | 2.0x |
| Median step | 0.880 s | 0.407 s | 2.2x |
| Peak GPU memory (nvidia-smi) | 14,345 MiB | 2,099 MiB | **6.83x less** |
| Held-out eval loss (public rows) | 2.04967 | 2.04958 | −0.004 % |
| Loss curve vs base | | first step 0.00 %, mean 0.05 % | |
| Our kernels' share of training time (check 4) | | 17.0 % | |

Commit c64181f. All 10 checks of `harness/check_rules.py` pass, including the 79 Soup regression
tests with the patches applied, and the 34 tests of `solution/`. The same code in four other
sessions: 1.66x, 1.59x, 1.58x, 1.50x — a T4's clock and the machine's disk and CPU vary between
Kaggle sessions (base 82–90 s, our training loop 26.8–31.6 s), so the ratio moves by ±0.1.

Where a step's GPU time goes (torch.profiler, ~360 ms per step): cuBLAS GEMMs ~73 % (about 22
TFLOPS, close to what a T4 sustains), NF4 dequantization ~8 %, attention ~7 %, the rest of our
kernels ~10 %. About 23 s of the command is Soup's startup, model loading and saving.

## What we did

1. **Packed tokens end to end** (`solution/fast_qwen2.py`). TRL pads each batch on the right.
   Right after the embedding the real tokens (attention_mask = 1) are gathered into one
   `[N, 1536]` matrix and stay packed through all 28 layers and the loss (~800 real tokens per
   step instead of B·S padded positions).
2. **Varlen causal attention.** Attention runs PyTorch's memory-efficient kernel in its
   `cu_seqlens` form on the packed tokens: one launch for all sequences, no padded q/k/v, no work
   on padding (half the attention time of padded SDPA). A one-time self-test against SDPA guards
   it; padded SDPA is the fallback.
3. **One autograd Function per decoder layer, with recomputation.** The forward keeps the layer's
   fp32 input (the residual stream, 6 KB per token and layer). The backward recomputes what it
   needs, but not `down_proj` (its input gradient needs only the weight), and reuses the weights
   it dequantizes for the recomputation in its own GEMMs. Within one fixed 512 MiB budget for
   everything the forward leaves behind, the last layers of a step also keep their gate|up output
   and residual sum and skip that recomputation — more layers for short batches, none once the
   inputs alone fill the budget, so the peak does not grow with the batch length.
4. **Fused projections.** q|k|v and gate|up are one GEMM each over a jointly dequantized weight;
   the LoRA update is added in the GEMM epilogue (`addmm_(…, alpha=scaling)`); one GEMM takes
   each fused gradient back to the layer input.
5. **Own NF4 dequantization** (`solution/nf4.py`): bitsandbytes' layout (two 4-bit codes per
   byte, absmax per 64 values, absmax itself 8-bit over a 256-entry map with one fp32 scale per
   256 blocks plus an offset) with the same fp32 arithmetic, so the GEMMs see bit-identical
   weights (tested against `bitsandbytes.functional.dequantize_4bit`). One tile = 32 blocks, the
   scale read once per block; `tools/nf4_bench.py` compares tilings (~210 GB/s on a T4).
6. **LoRA dropout without stored masks.** peft applies dropout to the input norm's output before
   `lora_A`. The input-norm kernel writes the normalized activations and the two dropped copies
   (q and v adapters) in one pass; the masks are a counter hash of (seed, element index) — the
   murmur3 finalizer — so the recomputing backward and the norm's backward kernel regenerate
   them exactly (`tools/mask_probe.py` and the tests check that forward, backward and a torch
   model of the hash agree on every element).
7. **RoPE + GQA in one kernel** (`solution/rope.py`): rotates q and k in fp32 (as transformers
   does), repeats the two KV heads for the twelve query heads, writes the packed layout attention
   reads; its backward undoes all of it in one pass per token.
8. **Head + loss + metrics in one pass** (`solution/cross_entropy.py`). Packed tokens, labelled
   ones first, in chunks of 512: one GEMM gives fp16 logits; `_custom_lmhead_ce_kernel` reads each
   row once (online max, sum-exp and sum(exp·x), argmax) and writes log-sum-exp, the token loss,
   TRL's per-token entropy and the argmax hit; for labelled rows it then overwrites the logits
   with `(softmax − onehot) · 1024`, the gradient, and a second GEMM over the labelled rows only
   takes it back to the hidden states — in the forward pass. The backward only rescales. Prompt
   tokens need the forward GEMM (TRL's entropy covers every token) but never a gradient GEMM.
9. **Soup's outputs from the fused pass.** TRL's `entropy`, `num_tokens` and
   `mean_token_accuracy` are logged by TRL's formulas; Soup's rewind recorder gets the per-token
   losses through the interface Soup itself defines for steps without logits
   (`trainer._soup_chunked_step.token_loss`, see `soup_cli/utils/stream_loss_chunk.py`).
10. **The frozen tied embedding stays fp16.** `prepare_model_for_kbit_training` would replace it
    by a 933 MB fp32 copy; the model is loaded in fp16 and the head multiplies in fp16 anyway, so
    we keep the fp16 tensor — the very values the fp32 copy would hold.
11. **Startup.** Compile time counts and every run starts with empty caches: three threads
    compile the step's kernel variants (and build their launchers) while Soup loads the data and
    the model; the first step waits 0 s. And TRL rebuilds `importlib.metadata.packages_distributions()`
    at import — stat-ing every file of every installed package, ~4.7 s on Kaggle's image — moments
    after transformers built the same mapping; while TRL imports, it is handed transformers' map
    (nothing is installed in between; the stdlib function is restored right after).

Anything outside what this path covers (another architecture, LoRA on o/MLP projections, DoRA,
several adapters, left-padded or packed batches, a batch without labelled tokens) is detected
before the step and goes through transformers' own code, which our older kernels (RMSNorm,
SwiGLU, RoPE, cross-entropy, unpadded MLP) still speed up.

### Our kernels

| Kernel | File | What it does |
|---|---|---|
| `_custom_nf4_dequant_kernel` | `nf4.py` | NF4 + double-quant dequantization into fused weight groups |
| `_custom_rmsnorm_drop_fwd_kernel` / `_bwd_kernel` | `rmsnorm.py` | input RMSNorm with the LoRA dropout masks fused in (`_custom_dropout_keep`) |
| `_custom_fused_add_rmsnorm_fwd_kernel` / `_bwd_kernel` | `rmsnorm.py` | residual add + post-attention RMSNorm |
| `_custom_rmsnorm_fwd_kernel` / `_bwd_kernel` | `rmsnorm.py` | final norm |
| `_custom_qkv_rope_pad_fwd_kernel` / `_bwd_kernel` | `rope.py` | RoPE + KV head repeat, packed (or padded) for attention, and back |
| `_custom_fused_swiglu_fwd_kernel` / `_bwd_kernel` | `swiglu.py` | SiLU(gate)·up on the fused gate\|up output; backward in place |
| `_custom_row_gather_kernel` | `unpad.py` | unpad / pad, embedding lookup, scaled gradient scatter |
| `_custom_lmhead_ce_kernel` | `cross_entropy.py` | log-sum-exp, loss, entropy, argmax and in-place gradient per row |

The remaining kernels in `solution/` (`_custom_ce_*`, `_custom_fused_rope_*`) serve the fallback
path only.

### Numerics

The same dequantized weights as bitsandbytes; the residual stream in fp32; RMSNorm, RoPE and
softmax statistics in fp32; fp16 GEMMs with fp32 accumulation — the precisions of the base step.
The loss gradient is formed in fp32 and rounded to fp16 once, scaled by 1024 so that small
probabilities survive (the role GradScaler's scale plays in the base step). LoRA dropout draws
its masks from its own hash stream (seeded from the run's seed), so they differ from PyTorch's
the way another seed's would. The packed step matches transformers + peft + bitsandbytes on a
small random Qwen2 (`test_packed_step_matches_transformers`, both attention paths, with and
without recomputation: loss, LoRA gradients, entropy, rewind losses).

Ideas we built on, with our own code: chunked / fused linear cross-entropy as in Liger-Kernel and
Apple's Cut Cross-Entropy; NF4 / double quant as in QLoRA (bitsandbytes); padding-free training
and layer recomputation (gradient checkpointing) as in Unsloth; varlen attention as in
FlashAttention's `cu_seqlens` interface (here PyTorch's own memory-efficient kernel).

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

`python run_remote.py` (our helper) pushes the repository and runs the notebook on Kaggle through
its API, adding a cell that prints the logs, a profiled run and the `tools/` diagnostics; it needs
your own Kaggle credentials in `~/.kaggle/kaggle.json`.

## Limits and next steps

- **The GEMM floor.** Three quarters of a step is cuBLAS at ~22 TFLOPS; the forward and the
  input-gradient GEMMs that cannot be avoided are ~4.8 TFLOP per step, ~14 s of the training at
  best. With ~23 s of fixed startup and saving, ~40 s is the floor for this command on a T4.
- **Speed vs memory.** Recomputation is the price of the memory result; the activation budget
  (512 MiB) sets the trade. Without kept layers the peak is 1,779 MiB (8.06x) and the step ~10 %
  slower; keeping every layer would cost ~2.5 GB more.
- **Attention** is PyTorch's memory-efficient kernel (varlen). A Triton flash kernel of ours was
  not faster on the T4 earlier.
- **Scope.** The packed path is for Qwen2-style decoders with bitsandbytes NF4 weights and LoRA on
  q/k/v, the shape of this task; it has been measured on the T4 only (sm_75: fp16 tensor cores,
  `num_stages=2`). Everything else falls back to the standard path.

## Rules of the challenge

The original task description, rules and checks are in the starter kit:
https://github.com/MakazhanAlpamys/soup-kernel-challenge (`task/`, `harness/` and `tests/` here
are unchanged copies). License: Apache-2.0, see `LICENSE` and `NOTICE`.

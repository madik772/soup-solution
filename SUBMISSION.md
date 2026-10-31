# Submission: TODO team name

## Links
- Repository, final commit: https://github.com/madik772/soup-solution/commit/TODO-final-sha
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README): https://www.kaggle.com/code/madiyarzhumazhan/notebook0b34895d24?scriptVersionId=TODO
- Demo video (up to 3 minutes): TODO
- Presentation: TODO

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| TODO | | madik772 | madiyarzhumazhan | |

## Our kernels
Shares from `own_kernel_ms` / `timed_loop_s` of the solution run (steps 3+, Kaggle T4, commit
62c1000).

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `_custom_nf4_dequant_kernel` | `solution/nf4.py` | bitsandbytes NF4 dequantization (into fused q\|k\|v and gate\|up weights) | 6.75 % |
| `_custom_fused_swiglu_bwd_kernel` | `solution/swiglu.py` | SiLU·up backward (in place on the fused gate\|up buffer) | 1.80 % |
| `_custom_rmsnorm_drop_fwd_kernel` | `solution/rmsnorm.py` | input RMSNorm + peft's LoRA dropout (two masks, Philox) | 1.32 % |
| `_custom_qkv_rope_pad_bwd_kernel` | `solution/rope.py` | RoPE / repeat_kv / pad backward | 1.16 % |
| `_custom_fused_swiglu_fwd_kernel` | `solution/swiglu.py` | SiLU(gate)·up | 1.06 % |
| `_custom_qkv_rope_pad_fwd_kernel` | `solution/rope.py` | RoPE + repeat_kv + padding for SDPA | 1.02 % |
| `_custom_fused_add_rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | residual add + post-attention RMSNorm | 0.78 % |
| `_custom_lmhead_ce_kernel` | `solution/cross_entropy.py` | cross-entropy, its backward, TRL's entropy and argmax accuracy over the full logits | 0.76 % |
| `_custom_rmsnorm_drop_bwd_kernel` | `solution/rmsnorm.py` | RMSNorm backward + LoRA dropout backward + residual gradient | 0.65 % |
| `_custom_row_gather_kernel` | `solution/unpad.py` | unpad / pad / embedding lookup / gradient scatter | 0.53 % |
| `_custom_fused_add_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | its backward | 0.42 % |
| `_custom_rmsnorm_fwd_kernel`, `_custom_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | final norm | 0.04 % |
| **Total** | | | **16.3 %** |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Fused linear cross-entropy, chunked so the logits never exist whole: Liger-Kernel
  (https://github.com/linkedin/Liger-Kernel), Cut Cross-Entropy (https://github.com/apple/ml-cross-entropy).
- NF4 with double quantization: QLoRA / bitsandbytes (https://github.com/bitsandbytes-foundation/bitsandbytes).
- Padding-free training and layer recomputation: Unsloth (https://github.com/unslothai/unsloth).
- The per-token no-logits interface for Soup's rewind log: Soup's own `stream_loss_chunk`.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 90.76 | 61.70 | speedup 1.471x |
| Peak GPU memory, MiB | 14345 | 1779 | base/ours 8.063x |
| Eval loss (public rows) | 2.04970 | 2.04955 | −0.007 % vs base |

(The same code in another session: 78.21 s vs 57.76 s, 1.354x; base time varies by session.)

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps — see README.md.

## Limitations
See README.md, "Limits and next steps": recomputation costs ~25 % more GEMM work in exchange for
the 8x memory result; ~23 s of the command is Soup's own startup and saving; attention is
PyTorch's SDPA on padded q/k/v; the packed path covers Qwen2 + NF4 + LoRA on q/k/v and falls
back to the standard path otherwise.

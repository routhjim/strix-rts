# Measurement log

Conditions unless stated: Ryzen AI Max+ 395, 128 GB LPDDR5X, Fedora 44,
llama.cpp Vulkan (b10433 / PR-27342 build), warm prefill cache, 400-token
generations, coherence-gated (unique-word ratio ≥ 0.25).

## Speculation stack, Qwen3.8-27B Q4_K_S

| config | freeform | code | long 8.8k |
|---|---|---|---|
| autoregressive | 13.4 | — | 12.9 |
| MTP drafter n4 | 26.5 | 43.6 | 31.1 |
| MTP n4 + ngram-mod n-min 24 | 28.8 | 66.5 | 31.5 |
| DFlash2 alone | 26.6 | — | 29.9 |
| **DFlash2 + ngram** | 28.7 | 66.2 | **38.6** |
| MTP n5 / n8 (under ngram) | 25.5 / 14.7 | 65.8 / 48.3 | — |

Draft acceptance: DFlash2 0.745, MTP 0.685 on prose; 0.61 on reasoning traces
(reasoning is the hardest content to draft — each token is less predictable).

## Backend

| workload | Vulkan | ROCm/HIP |
|---|---|---|
| freeform | 29.1 | 29.5 |
| code | 67.5 | **90.2** |
| long-doc 8.8k (salted) | **30.7** | 15.6 |
| prefill 8.8k | 253 | 261 |

ROCm wins where speculation accepts *wide* (batched verify = GEMM, rocBLAS's
strength) and loses badly at long context. Unexplained; the crater is the open
question if anyone wants to dig.

## Batching (`-np`), DFlash2+ngram vs autoregressive

| clients | with speculation | AR only |
|---|---|---|
| 1 | 28.6 | 13.4 |
| 2 | 34.6 | 22.6 |
| 4 | 33.1 | **39.5** |

Speculation and batching compete for the same amortisation. Use speculation to
~3 concurrent; beyond that turn it off and batch.

## Reasoning effort (previously-wrong MMLU, n=92)

| effort | recovery | tokens/answer | wall (117 answers) |
|---|---|---|---|
| medium | 46.7 % | 424 | 27 min |
| xhigh | 53.3 % | 2 516 | 155 min |

xhigh's marginal wins were long-grind *verbal* reasoning (law, ethics, econ);
medium already recovers 87 % of the quantitative misses. On terminal-agent
tasks xhigh went 0-for-8 vs medium.

## Thinking under retrieval noise

| longreason | direct | rts |
|---|---|---|
| thinking off, 182 tok / 8.2k / 16k | 19 / 13 / 13 (of 24) | 21 / 20 / 20 |
| **thinking medium, all lengths** | **24 / 24 / 24** | **24 / 24 / 24** |

Thinking fully removes length degradation for +18 % wall clock. A distractor
QA suite was 180/180 with thinking on. We did not reproduce inverse scaling.

## Sparse MoE comparison (DeepSeek-V4-Flash 284B/13B-active, 3-bit pruned)

Raw AR decode 19.3 t/s at 5.84 GB/token — *faster* than the dense 27B's 13.4.
But every amortisation lever failed: DSpark −16 %, ngram −61 %, verify-width
flat, decode sparsity −18 %. Best speculative config still −12 % vs its own AR.
On a 16-task terminal-agent set: dense 27B 4/11, MoE-35B 2/11, V4 1/11 — all
V4 failures were timeouts at ~half the token rate.

Conclusion: sparsity spends its bandwidth saving *per token* and makes tokens
non-fungible; dense weights are one shared read, so bulk tricks multiply.

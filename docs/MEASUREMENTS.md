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

## End-to-end stack validation (120 TriviaQA questions, public ground truth)

Same 120 questions, four configurations, matched `MAX_DOC_CHARS=2400`, run
**sequentially** — C and D share the answerer endpoint, so running them
concurrently would have measured GPU contention instead of architecture.

Graded bidirectionally against the dataset's own alias lists (see
[the grader note](#grading)); strict-substring scores in parentheses.

| arm | accuracy | median | p95 | total |
|---|---|---|---|---|
| A closed-book (no search, no thinking) | 88/120 = **73.3 %** (70.8) | 0.6 s | 0.9 s | 1.4 min |
| B closed-book + thinking medium | 90/120 = **75.0 %** (72.5) | 2.3 s | 21.1 s | 12.4 min |
| D single-model (q38 extracts for itself) | 104/120 = **86.7 %** (85.0) | 27.3 s | 42.2 s | 56.5 min |
| **C cross-model** (A3B extracts → q38 answers) | **110/120 = 91.7 %** (88.3) | **11.8 s** | 21.8 s | **24.9 min** |

### Why the cross-model split is the whole design

**C beats D on both axes at once: 2.3× faster *and* +5.0 points more accurate.**
That is the result worth reproducing, because "add a second model" normally buys
speed at the cost of fidelity, and here it buys both.

Speed is the expected half — a 3 B-active MoE reads far less memory per token
than a 27 B dense model, and extraction is the token-heavy call (it ingests all
six documents; the answer call sees only the extracted spans).

Accuracy is the surprise, and it has a specific mechanism. Paired over the 120:

| | count |
|---|---|
| both correct | 101 |
| **C only** | **9** |
| D only | 3 |
| both wrong | 7 |

| | C cross-model | D single-model |
|---|---|---|
| sufficiency refusals ("couldn't find a reliable source") | **11/120** | **26/120** |
| answers citing a retrieved source | **101/120** | 84/120 |

q38 self-extracting **discards evidence it was actually handed** more than twice
as often, then falls back to parametric memory and misses. Every one of the nine
C-only wins is this pattern — the fact was verbatim in the retrieved documents:

> *"Which northern English beer was originally launched by Col. James Porter in 1927?"*
> C: `Newcastle Brown Ale [1]` · D: *"I couldn't find a reliable source confirming
> this specific detail. From my own knowledge…"*

A big instruction-tuned reasoner reads a raw six-document search dump as a
question about *trustworthiness*; a small extractor whose only job is span
extraction just returns the span. Specialising the extractor is not merely a
cheaper way to do the same work — **it is a more faithful way**, because the
answerer never sees the noise that triggers its own caution.

### Search vs thinking

**+16.7 points (fair: +18.4) from retrieval.** Arm B is the control that makes
this interpretable: **thinking alone bought +1.7 points for 4× the latency**,
because factual recall is a *knowledge* gap and deliberation cannot invent facts.
The mirror result is in the multi-hop suite above, where thinking took 45/72 →
72/72 and retrieval was irrelevant.

> **Search fixes knowledge gaps. Thinking fixes reasoning gaps. They are not
> substitutes, and a router should spend each only where it pays** — which is why
> the proxy decides them independently rather than tying thinking to search.

Paired against closed-book, search corrected **17** questions the model missed and
**broke 3** — retrieved snippets displacing correct parametric knowledge. That is
the documented distractor effect in practice, and the reason the sufficiency gate
exists (and the reason it must not be *too* eager — see D above).

Cost note: 11.8 s median is ~20× a closed-book turn. Auto-routing matters precisely
because it spends that only on turns that need it; a social turn still returns in ~1.3 s.

### Context budget per search result

Measured on the 11 questions the stack actually **failed** — an earlier
single-question test wrongly suggested 1200 was optimal, because on that question
every cap worked:

| `MAX_DOC_CHARS` | gold fact reached the extractor | search+extract |
|---|---|---|
| 1200 | 4/11 | 3.7 s |
| 6000 | 6/11 | 10.5 s |
| uncapped | 6/11 | 14.1 s |

Above ~6000 buys nothing, and the remaining 5 are **search-coverage** failures no
context size can fix. Default is **2400** as the compromise; raise it to 6000 to
buy the last ~1.7 points for ~7 s more per search.

### Grading

Scores are reported with a **bidirectional** alias matcher
([`bench/fairgrade.py`](../bench/fairgrade.py)). The obvious strict test — "is a
gold alias a substring of the reply?" — rejects correct answers that are *shorter*
or reordered than the alias: `Eva Cassidy` vs alias `Eva Marie Cassidy`,
`violin` vs `The Violin`, `The molecular structure of DNA` vs `DNA structure`.
It also scores the reply's **last** line, since models list candidates before
committing. This moves every arm up 2–3 points and does not change any ranking;
strict numbers are kept in parentheses above so both are reproducible.

## Sparse MoE comparison (DeepSeek-V4-Flash 284B/13B-active, 3-bit pruned)

Raw AR decode 19.3 t/s at 5.84 GB/token — *faster* than the dense 27B's 13.4.
But every amortisation lever failed: DSpark −16 %, ngram −61 %, verify-width
flat, decode sparsity −18 %. Best speculative config still −12 % vs its own AR.
On a 16-task terminal-agent set: dense 27B 4/11, MoE-35B 2/11, V4 1/11 — all
V4 failures were timeouts at ~half the token rate.

Conclusion: sparsity spends its bandwidth saving *per token* and makes tokens
non-fungible; dense weights are one shared read, so bulk tricks multiply.

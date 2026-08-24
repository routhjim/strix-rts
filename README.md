# strix-rts — a measured serving stack for AMD Strix Halo

A **retrieve-then-solve proxy** and the tuning findings behind it, for running local
LLMs well on a Ryzen AI Max+ 395 (Strix Halo, 128 GB unified LPDDR5X, ~256 GB/s).

Everything here is measured on one box over a week of controlled A/Bs, including
the things that **didn't** work. Numbers are reproducible with the harness in `bench/`.

> **On confidence:** this is **n=1 hardware, and most cells are single runs.** The
> directions are consistent and each has a mechanism behind it, but these are not
> multi-run means with error bars — treat any individual figure as ±10 % and trust
> the *ordering* more than the value. Repeat-run spread on identical configs was
> ~5 %, and speculative decoding shows genuine bimodal draft-alignment basins, so
> long-context throughput is a distribution, not a point.

---

## 1. What the proxy does

`proxy/rts_proxy.py` is a single OpenAI-compatible endpoint that transparently runs
a two-model retrieval pipeline. Any client that speaks OpenAI — Open WebUI, OpenClaw,
curl — points at it and inherits the behaviour with no plugin.

```
client ──/v1/chat/completions──▶  RTS proxy (:8090)
    1. decide: does this turn need a search?      (auto | always | off)
    2. web_search                                  (Ollama cloud API)
    3. EXTRACT   ── cheap high-prefill model copies the relevant lines verbatim
    4. SUFFICIENCY ── YES/NO gate: no evidence ⇒ say so, don't invent a citation
    5. ANSWER    ── main model reasons over the ~300-char extract, with [n] cites
```

**Why two models.** On this hardware a sparse MoE (Qwen3.6-35B-A3B) prefills at
**938 t/s** while a dense 27B does **253 t/s**. Retrieval is a *reading* job, answering
is a *reasoning* job — so the MoE reads and the dense model thinks.

Measured on a multi-hop benchmark with same-shape distractors:

| context | single-model RTS | **cross-model RTS** | speedup |
|---|---|---|---|
| 182 tok | 5.6 s | 3.0 s | 1.9× |
| 8.2 k | 35.4 s | 11.2 s | **3.2×** |
| 16 k | 69.1 s | 20.3 s | **3.4×** |

Accuracy did not drop — it rose slightly (61/72 → 66/72), because the MoE is the
better *copier*. The extract is ~0.3 % of the input, so the answering call is nearly free.

**Why extraction has no escape hatch.** Letting the extractor reply `NONE` is
model-dependent and dangerous: one model took that option on **73 %** of inputs where
the answer was verbatim in doc 1 (others: 2.2 %, 0 %). The failure is silent — you still
get fluent answers, sourced from the model's memory instead of your search results.
Extraction is now unconditional and sufficiency is a **separate** call.

---

## 2. Best-found configuration (Strix Halo, llama.cpp)

**Answerer — Qwen3.8-27B, Q4_K_S, Vulkan:**
```
-m Qwen3.8-27B-Q4_K_S.gguf \
-md Qwen3.8-27B-DFlash2-Q4_K_M.gguf -ngld 99 \
--spec-type draft-dflash,ngram-mod --spec-ngram-mod-n-min 24 \
-ngl 99 -fa on --jinja --chat-template-file <fixed-template>.jinja \
-c 65536 -b 4096 -ub 2048 -np 1
```

| workload | t/s |
|---|---|
| autoregressive (no speculation) | 13.4 |
| freeform prose | **29** |
| code / structured | **66** (90 on ROCm) |
| long-doc 8.8 k | **31** (38 warm) |
| prefill | 253 |
| reasoning traces (thinking on) | 27 |

**Extractor — Qwen3.6-35B-A3B Q8_0:** 938 t/s prefill, 61 t/s decode, `-ub 2048`.

**Switchable modes:** ROCm for code-heavy (+34 %), `-np 4` + speculation off for 4+
concurrent users (~40 t/s aggregate), A3B alone for prefill-dominated bulk work.

### The speculation stack is the whole story
Stacking a **model drafter + ngram** is worth more than either alone. Comma-separated
`--spec-type` composes them: the drafter handles novel text, ngram completes literal
repetition wholesale. Gains over the drafter alone: **+9 % prose, +53 % code, +9 % long-doc.**

---

## 3. Negative results (the expensive half)

- **Speculation is net-negative on sparse MoE.** On a 284B/13B-active model every
  variant lost (−16 % to −61 %): each verified position routes to its own experts, so
  the verify batch reads their *union* — width costs bytes, which is exactly the
  amortisation speculation depends on. Dense weights are one shared read, so the same
  tricks multiply. **The identical ngram setting: −61 % on MoE, +53 % on dense.**
- **DFlash2 alone is a wash** (higher acceptance, same throughput) — it only pays
  *stacked with ngram*, and only at long context.
- **Don't tune the drafter's block size.** Every explicit `n-max`/`p-min` setting was
  worse than model defaults. For the MTP drafter, `n4` beat n5 and n8 decisively.
- **`-ub` is model-specific, not a universal.** 256 costs the MoE **35 %** of prefill
  and is noise (±5 %) on the dense model. Use 2048 for both.
- **Max reasoning effort buys nothing agentic.** `xhigh` gained +7 pts on hard
  knowledge questions (at 6× the tokens and 5.7× the wall clock) but went **0-for-8**
  against `medium` on terminal-agent tasks, losing one.
- **Don't max the context.** Every search-result truncation from 1200 chars to
  uncapped found the target fact; latency scaled 3.7 s → 14.1 s. Retrieval quality is
  in the ranking, not the tail.
- **Thinking does *not* collapse under distractors** (contra the inverse-scaling
  literature, at least for this model): 180/180 on a distractor suite, and it took a
  multi-hop suite from 45/72 to **72/72** for only 18 % more wall clock.

---

## 4. Setup

```bash
cp systemd/*.service ~/.config/systemd/user/     # edit %HOME% and model paths
printf 'OLLAMA_KEY=<your key>\n' > ~/.config/rts-proxy.env && chmod 600 ~/.config/rts-proxy.env
systemctl --user daemon-reload
systemctl --user enable --now llama-server a3b-extractor rts-proxy
```
Point your client's OpenAI base URL at `http://127.0.0.1:8090/v1`.
Single-model mode: set `EXTRACT_ENDPOINT` equal to `ANSWER_ENDPOINT`.

**Config (env):** `SEARCH_MODE=auto|always|off` · `MAX_DOC_CHARS=1200` ·
`ANSWER_THINK=medium|off|xhigh` · `ANSWER_ENDPOINT` · `EXTRACT_ENDPOINT` · `OLLAMA_KEY`

### Thinking belongs on the answer call only
Run the answering server with **`-rea off`** and let the proxy opt in per request.
Extraction and the sufficiency check stay thinking-off (copy and classify jobs).
Thinking-medium on the grounded answer is the best accuracy-per-second lever measured
here: a multi-hop retrieval suite went **45/72 → 72/72 for +18 % wall clock**.

**Do not set the server default to `-rea auto`.** It makes *every* turn think, so a
short request with a small `max_tokens` spends its whole budget on the reasoning trace
and returns **empty content** — a silent failure that looks like the model breaking.

If you use Open WebUI, disable its built-in web search — the proxy owns that path now.

---

## 5. Reproducing the numbers

`bench/` builds two suites with **deterministic ground truth**:

- `build_rag_corpus.py` — QA with **same-answer-type** hard distractors (a year
  question distracted by other years). Gold is always present: this measures *using*
  a context, not retrieving one.
- `build_longreason.py` — multi-hop over **invented entities** (`the Kestrel survey was
  completed in 1962`), so parametric knowledge can't help. Two task types: compare five
  scattered facts, and count against a threshold.
- `kbench_rts.py` — runs `direct` vs `rts` arms at 3 context lengths; `RTS_EXTRACT_EP`
  points extraction at a second model for the cross-model arm.

**Method notes learned the hard way:** check whether any run touched its `max_tokens`
ceiling before comparing (a cap sized for terse models silently penalises verbose ones);
measure prefill on a *salted* prompt or the cache reports thousands of t/s; and verify
label distributions aren't degenerate before trusting a score.

## License
MIT.

"""Retrieve-then-Solve middleware for a local web-search assistant.

Drops in between "search tool returned results" and "ask the local LLM to answer".
Both calls go to the SAME local model — no extra model, no extra service.

Why: long inputs degrade answer quality even when the needed fact is present and
retrieval is perfect (13.9-85% depending on task; a large part of the drop lands
within 7k tokens). Compressing to the few relevant lines converts a long-context
task into a short-context one, which is where these models are strong.

Measured on Qwen3.8-27B-Q4 locally (2026-08-20):
  extraction output was a median 28 tokens against an 8,305-token input (0.3%),
  so call 2 is nearly free and the overhead is a roughly CONSTANT ~1.5s:
      28 tok ctx  -> +144%      211 tok -> +94%      8.3k tok -> +5%
  i.e. negligible exactly where you need it, painful on short inputs. Hence
  MIN_CTX_TOKENS below.
"""
from __future__ import annotations
import json, re, urllib.request

# Two endpoints (measured 2026-08-24): a cheap high-prefill EXTRACTOR reads the
# long context, the ANSWERER reasons over the ~30-token extract. On Strix Halo
# A3B (938 t/s prefill) extract + q38 answer = 3.4x faster cold-context at 16k,
# with EQUAL-or-better accuracy (A3B is the stronger copy model). Point both at
# the same endpoint to run single-model.
ANSWER_ENDPOINT  = "http://127.0.0.1:8080"      # q38 daily driver
EXTRACT_ENDPOINT = "http://127.0.0.1:18094"     # A3B extractor (fallback: ANSWER_ENDPOINT)
MODEL = "local"
MIN_CTX_TOKENS = 800      # below this, skip RTS — the flat ~1.5s isn't worth it
NO_EVIDENCE = "NONE"


def _chat(messages, max_tokens, temperature=0.0, timeout=600, endpoint=None):
    body = {
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        # extraction is a copy task; thinking adds latency and, per the 2026
        # distractor literature, can make noise-robustness worse, not better
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        (endpoint or ANSWER_ENDPOINT).rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


def format_results(results) -> str:
    """results: [{'title':..., 'url':..., 'snippet'/'content':...}, ...]
    Numbered docs so the extractor can carry a citation handle through."""
    out = []
    for i, r in enumerate(results, 1):
        body = r.get("content") or r.get("snippet") or ""
        out.append(f"[{i}] {r.get('title','')} — {r.get('url','')}\n{body}")
    return "\n\n".join(out)


def extract_evidence(question: str, context: str) -> str:
    """Call 1: verbatim copy of the relevant lines. NO escape hatch — see below.

    MEASURED 2026-08-20: offering "reply NONE if nothing is relevant" in THIS call
    is model-dependent and dangerous. DeepSeek-V4-Flash took the escape hatch on
    73% of inputs where the answer was verbatim in doc 1 (Qwen3.8 2.2%, A3B 0%).
    The failure is invisible: you still get a fluent answer, sourced from the
    model's memory instead of your search results.

    So extraction is now unconditional, and sufficiency is a SEPARATE call
    (evidence_is_sufficient). Never let one call decide both "what is relevant"
    and "is anything relevant" — a model can duck the first by answering the second.

    Verbatim matters too: asking for a summary gives a second chance to
    hallucinate and costs you the ability to check the quote against the page.
    """
    prompt = (
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Do NOT answer the question. Copy out, word for word, the line(s) above "
        f"most relevant to it, each prefixed with its [n] source number. At most 5 "
        f"lines. Do not summarise, do not rephrase, do not add anything of your own."
    )
    return _chat([{"role": "user", "content": prompt}], max_tokens=400,
                 endpoint=EXTRACT_ENDPOINT)


def evidence_is_sufficient(question: str, evidence: str) -> bool:
    """Call 1b: the abstention check, asked on its own and on a SHORT context.

    Cheap (evidence is ~30 tokens) and far more reliable than folding it into the
    extraction call. This is what preserves the "retrieval failed" signal.
    """
    verdict = _chat([{"role": "user", "content":
        f"Evidence:\n{evidence}\n\nQuestion: {question}\n\n"
        f"Does the evidence above actually answer the question? Reply YES or NO."}],
        max_tokens=8, endpoint=EXTRACT_ENDPOINT)
    return verdict.strip().upper().startswith("Y")


def answer_from_evidence(question: str, evidence: str, history=None) -> str:
    """Call 2: short context — evidence only, never the raw dump."""
    msgs = list(history or [])
    msgs.append({"role": "user", "content":
                 f"Evidence:\n{evidence}\n\nQuestion: {question}\n\n"
                 f"Answer using only the evidence above. Cite the [n] source for "
                 f"each claim. If the evidence is insufficient, say so plainly."})
    return _chat(msgs, max_tokens=1024)


def retrieve_then_solve(question: str, search_results, history=None):
    """Returns (answer, evidence, used_rts).

    `evidence == NONE` is a free retrieval-failure signal you do not get from the
    direct path (now produced by the separate sufficiency check, not by letting
    the extractor opt out): the model is telling you the search did not contain the answer,
    BEFORE it has committed to a fluent wrong answer. Re-search or say "I don't
    know" instead of letting it improvise.
    """
    context = format_results(search_results)
    if len(context) // 4 < MIN_CTX_TOKENS:            # ~4 chars/token
        return answer_from_evidence(question, context, history), context, False

    evidence = extract_evidence(question, context)
    if not evidence.strip() or not evidence_is_sufficient(question, evidence):
        return None, NO_EVIDENCE, True                # caller decides: re-search / abstain
    return answer_from_evidence(question, evidence, history), evidence, True


# --- multi-turn note -------------------------------------------------------
# Store the EVIDENCE in conversation history, never the raw search dump. Raw
# results accumulate across turns and push you into the degradation zone within
# a few exchanges; the extract is ~0.3% of the size and carries the citations.
#
#   history.append({"role": "assistant", "content": answer})
#   history.append({"role": "system", "content": f"Evidence used: {evidence}"})

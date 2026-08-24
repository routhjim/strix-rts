#!/usr/bin/env python3
"""Retrieve/Plan-then-Solve benchmark. THINKING OFF everywhere.

Rationale (see the 2026-08-20 research pass):
 - Closed-book MMLU/TriviaQA measure the capability web search replaces. Retired.
 - What matters with search on: using a context you were GIVEN, under noise and
   length, and structuring hard problems before solving them.

SUITE ragqa  -- the gold fact is ALWAYS present, so this is not a retrieval test.
  L0 clean : gold only
  L1 noisy : gold + 8 same-answer-type hard distractors   -> isolates DISTRACTION
  L2 long  : L1 + wiki filler to ~8k tokens               -> isolates LENGTH
  arms: direct | rts (2 calls: extract evidence, then answer from evidence ALONE)

SUITE hard   -- questions the model already got WRONG closed-book (its direct
  baseline is 0 by construction, so only the pts arm is run).
  arm: pts (2 calls: state an approach without answering, then answer using it)
  Comparable to the 2026-08-19 thinking-recovery numbers, but with thinking OFF.
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbench import jload, jappend, chat, mmlu_prompt, LETTERS
D = os.path.dirname(os.path.abspath(__file__))
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
# RTS_THINK=1 flips the battery's thinking-off default. Used only to fill the
# 2x2 cell "plan-then-solve WITH thinking" against the 2026-08-19 thinking-on
# baseline (42/92) and the thinking-off plan-then-solve result (24/92).
THINK = {"reasoning_effort": "medium",
         "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}}

def _mt(mt):
    # with thinking ON the trace spends the same completion budget as the answer;
    # a 64-token cap would truncate every trace into a fake zero (the recurring
    # cap bug). Give thinking calls real room.
    return max(mt, 4096) if os.environ.get("RTS_THINK") else mt

def _chat(ep, model, prompt, mt, timeout=900):
    mt = _mt(mt)
    on = os.environ.get("RTS_THINK")
    os.environ["KBENCH_EXTRA_JSON"] = json.dumps(THINK if on else NOTHINK)
    return chat(ep, model, prompt, mt, timeout)

# ---------------- context construction ----------------
def build_context(item, level, filler, target_tok=8000):
    facts = [item["gold_fact"]] + item["distractor_facts"][:8] if level != "L0" else [item["gold_fact"]]
    import random
    random.Random(hash(item["id"]) & 0xffff).shuffle(facts)
    blocks = [f"[doc {i+1}] {f}" for i, f in enumerate(facts)]
    if level == "L2":
        need = target_tok * 4                       # ~4 chars/token
        got = sum(len(b) for b in blocks); k = 0
        while got < need and k < len(filler):
            blocks.insert(random.Random(k).randrange(len(blocks) + 1),
                          f"[doc {len(blocks)+1}] {filler[k]}")
            got += len(filler[k]); k += 1
    return "\n\n".join(blocks)

def build_lr_context(item, level, filler, target_tok):
    import random
    blocks = item["facts"] + item["distractor_facts"]
    random.Random(hash(item["id"]) & 0xffff).shuffle(blocks)
    if level != "L1":
        need = target_tok * 4; got = sum(len(b) for b in blocks); k = 0
        while got < need and k < len(filler):
            blocks.insert(random.Random(k).randrange(len(blocks) + 1), filler[k])
            got += len(filler[k]); k += 1
    return "\n\n".join(blocks)

def grade_lr(reply, item):
    r = (reply or "").strip()
    if item["kind"] == "COUNT":
        # take the LAST standalone small integer: models that show their working
        # ("- Teal: 1949 - yes ... 3") must not be graded on an intermediate value
        cands = [x for x in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", r) if int(x) <= 5]
        return bool(cands) and cands[-1] == item["answer"]
    hits = [(r.lower().find(n.lower()), n) for n in item["asked"] if n.lower() in r.lower()]
    return bool(hits) and min(hits)[1] == item["answer"]   # first name mentioned must be right

def run_longreason(a):
    items = jload(f"{D}/longreason.jsonl")[:a.n]
    filler = json.load(open(f"{D}/filler.json"))
    out = f"{D}/rts-{a.label}-longreason.jsonl"
    done = {(r["id"], r["level"], r["arm"]) for r in jload(out)} if os.path.exists(out) else set()
    for level, tgt in (("L1", 0), ("L2", 8000), ("L3", 16000)):
        for arm in ("direct", "rts"):
            for it in items:
                if (it["id"], level, arm) in done: continue
                ctx = build_lr_context(it, level, filler, tgt)
                t0 = time.time(); ev = ""
                try:
                    if arm == "direct":
                        p = f"{ctx}\n\n{it['ask']}"
                        rep, _ = _chat(a.endpoint, a.model, p, 400)
                    else:
                        p1 = (f"{ctx}\n\nTask: {it['ask']}\n\nDo NOT answer. Copy out, word for "
                              f"word, the Record lines that mention any of these surveys: "
                              f"{', '.join(it['asked'])}. List them exactly, one per line.")
                        ev, _ = _chat(os.environ.get("RTS_EXTRACT_EP", a.endpoint), a.model, p1, 400)
                        p2 = f"Records:\n{ev}\n\n{it['ask']}"
                        rep, _ = _chat(a.endpoint, a.model, p2, 400)
                except Exception as e:
                    rep = f"ERROR {e}"
                jappend(out, {"id": it["id"], "kind": it["kind"], "level": level, "arm": arm,
                              "reply": rep, "evidence": ev[:600], "ok": grade_lr(rep, it),
                              "ctx_chars": len(ctx), "secs": round(time.time() - t0, 1)})
        print(f"[{a.label}] longreason {level} done", flush=True)

def grade_trivia(reply, aliases):
    # normalise away spacing/punctuation: a reply of "1 9 7 9" must match "1979"
    n = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    r = n(reply)
    if not r: return False
    return any(n(a) in r for a in aliases if a)

def letter_of(txt):
    t = (txt or "").strip()
    m = re.match(r"^\**\(?([ABCD])\)?\b", t)
    if m: return m.group(1)
    ms = re.findall(r"(?i)answer\s*(?:is)?\s*:?\s*\**\(?([ABCD])\)?\b", t)
    return ms[-1].upper() if ms else None

# ---------------- suites ----------------
def run_ragqa(a):
    items = jload(f"{D}/ragqa.jsonl")[:a.n]
    filler = json.load(open(f"{D}/filler.json"))
    out = f"{D}/rts-{a.label}-ragqa.jsonl"
    done = {(r["id"], r["level"], r["arm"]) for r in jload(out)} if os.path.exists(out) else set()
    for level in ("L0", "L1", "L2"):
        for arm in ("direct", "rts"):
            for it in items:
                if (it["id"], level, arm) in done: continue
                ctx = build_context(it, level, filler)
                t0 = time.time(); ev = ""
                try:
                    if arm == "direct":
                        p = (f"Use ONLY the documents below to answer.\n\n{ctx}\n\n"
                             f"Question: {it['question']}\nAnswer with just the answer:")
                        rep, _ = _chat(a.endpoint, a.model, p, 64)
                    else:
                        p1 = (f"{ctx}\n\nQuestion: {it['question']}\n\n"
                              f"Do NOT answer the question. Copy out, word for word, the "
                              f"[doc n] line(s) above that are most relevant to it.")
                        ev, _ = _chat(os.environ.get("RTS_EXTRACT_EP", a.endpoint), a.model, p1, 256)
                        p2 = (f"Evidence:\n{ev}\n\nQuestion: {it['question']}\n"
                              f"Answer with just the answer:")
                        rep, _ = _chat(a.endpoint, a.model, p2, 64)
                except Exception as e:
                    rep = f"ERROR {e}"
                jappend(out, {"id": it["id"], "level": level, "arm": arm, "reply": rep,
                              "evidence": ev[:500], "ok": grade_trivia(rep, it["aliases"]),
                              "ctx_chars": len(ctx), "secs": round(time.time() - t0, 1)})
        print(f"[{a.label}] {level} done", flush=True)

def run_hard(a):
    qs = {q["id"]: q for q in jload(f"{D}/mmlu.jsonl")}
    base = jload(f"{D}/results-{a.base_label}-mmlu.jsonl")
    wrong = [qs[r["id"]] for r in base
             if not ((l := letter_of(r["reply"])) and LETTERS.index(l) == qs[r["id"]]["answer"])]
    out = f"{D}/rts-{a.label}-hard.jsonl"
    done = {r["id"] for r in jload(out)} if os.path.exists(out) else set()
    todo = [q for q in wrong if q["id"] not in done]
    print(f"[{a.label}] hard: {len(todo)} previously-wrong questions (plan-then-solve)", flush=True)
    for n, q in enumerate(todo):
        ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
        t0 = time.time()
        try:
            p1 = (f"Question about {q['subject'].replace('_',' ')}:\n{q['question']}\n{ch}\n\n"
                  f"Do NOT give the answer. In at most 4 short lines, state the method or "
                  f"principle needed to decide between these options.")
            plan, _ = _chat(a.endpoint, a.model, p1, 2048 if os.environ.get("RTS_THINK") else 256)
            p2 = (f"Question:\n{q['question']}\n{ch}\n\nApproach:\n{plan}\n\n"
                  f"Applying that approach, reply with ONLY the letter (A, B, C, or D).")
            rep, _ = _chat(a.endpoint, a.model, p2, 2048 if os.environ.get("RTS_THINK") else 16)
        except Exception as e:
            plan, rep = "", f"ERROR {e}"
        l = letter_of(rep)
        jappend(out, {"id": q["id"], "plan": plan[:400], "reply": rep,
                      "ok": bool(l and LETTERS.index(l) == q["answer"]),
                      "secs": round(time.time() - t0, 1)})
        if (n + 1) % 20 == 0: print(f"  [{a.label}] {n+1}/{len(todo)}", flush=True)

def score(a):
    for suite in ("ragqa", "longreason", "hard"):
        f = f"{D}/rts-{a.label}-{suite}.jsonl"
        if not os.path.exists(f): continue
        rows = jload(f)
        if suite == "longreason":
            print(f"\n{a.label} / longreason   (5 facts + 8 same-shape distractors; invented entities)")
            print(f"  {'level':6} {'direct':>12} {'rts':>12}   {'ctx tok':>8}")
            for lv in ("L1", "L2", "L3"):
                cells = {}
                for arm in ("direct", "rts"):
                    rs = [r for r in rows if r["level"] == lv and r["arm"] == arm]
                    cells[arm] = f"{sum(r['ok'] for r in rs)}/{len(rs)}" if rs else "-"
                ct = [r["ctx_chars"] for r in rows if r["level"] == lv]
                print(f"  {lv:6} {cells['direct']:>12} {cells['rts']:>12}   "
                      f"{(sorted(ct)[len(ct)//2]//4 if ct else 0):>8}")
            for kind in ("EARLIEST", "COUNT"):
                rs = [r for r in rows if r["kind"] == kind]
                if rs: print(f"    by kind {kind:9}: {sum(r['ok'] for r in rs)}/{len(rs)}")
        elif suite == "ragqa":
            print(f"\n{a.label} / ragqa   (gold fact ALWAYS present)")
            print(f"  {'level':6} {'direct':>12} {'rts':>12}   {'ctx tok':>8}")
            for lv in ("L0", "L1", "L2"):
                cells = {}
                for arm in ("direct", "rts"):
                    rs = [r for r in rows if r["level"] == lv and r["arm"] == arm]
                    cells[arm] = f"{sum(r['ok'] for r in rs)}/{len(rs)}" if rs else "-"
                ct = [r["ctx_chars"] for r in rows if r["level"] == lv]
                print(f"  {lv:6} {cells['direct']:>12} {cells['rts']:>12}   "
                      f"{(sorted(ct)[len(ct)//2]//4 if ct else 0):>8}")
        else:
            ok = sum(r["ok"] for r in rows)
            secs = sum(r["secs"] for r in rows) / max(1, len(rows))
            print(f"\n{a.label} / hard (plan-then-solve on previously-WRONG): "
                  f"{ok}/{len(rows)} = {100*ok/max(1,len(rows)):.1f}%  ({secs:.0f}s per question)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ragqa", "longreason", "hard"):
        p = sub.add_parser(name)
        p.add_argument("--endpoint", required=True); p.add_argument("--model", default="x")
        p.add_argument("--label", required=True); p.add_argument("--n", type=int, default=30)
        if name == "hard": p.add_argument("--base-label", required=True)
    s = sub.add_parser("score"); s.add_argument("--label", required=True)
    a = ap.parse_args()
    {"ragqa": run_ragqa, "longreason": run_longreason,
     "hard": run_hard, "score": score}[a.cmd](a)

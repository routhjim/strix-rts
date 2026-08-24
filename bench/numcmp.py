#!/usr/bin/env python3
"""Isolate NUMERIC COMPARISON from everything else.

V4-pruned picks the SECOND-earliest survey in 14/20 of its EARLIEST errors, with
complete extraction and correctly-attributed years — so the failure is selecting
a minimum over 5 values it already holds. Its errors concentrate where the top
two years are close (median gap 8y wrong vs 12y correct). Hypothesis: low-bit
quantization blurs fine numeric discrimination.

This strips away context, distractors and retrieval entirely: 5 labelled years,
one question, controlled minimum gap. Run the SAME model at Q4 and Q8 to separate
bit-width from model identity.
"""
import argparse, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbench import chat, jappend, jload
D = os.path.dirname(os.path.abspath(__file__))
NAMES = "Kestrel Marlin Osprey Petrel Tanager Vireo Merlin Bittern Godwit Curlew Avocet Dunlin".split()
GAPS = (1, 3, 8, 20, 50)          # gap between the smallest and 2nd-smallest year

def build(n_per_gap=12, seed=5):
    rng = random.Random(seed); out = []
    for gap in GAPS:
        for i in range(n_per_gap):
            base = rng.randrange(1900, 1950)
            ys = [base, base + gap]
            while len(ys) < 5:
                y = base + gap + rng.randrange(5, 60)
                if y not in ys: ys.append(y)
            names = rng.sample(NAMES, 5)
            pairs = list(zip(names, sorted(ys)))
            rng.shuffle(pairs)
            out.append({"id": f"nc-{gap}-{i}", "gap": gap,
                        "answer": min(pairs, key=lambda p: p[1])[0],
                        "pairs": pairs})
    with open(f"{D}/numcmp.jsonl", "w") as f:
        for o in out: f.write(json.dumps(o) + "\n")
    print(f"numcmp.jsonl: {len(out)} items, gaps {GAPS}, {n_per_gap} each")

VARIANTS = {
    # the phrasing used in longreason, verbatim
    "plain":    "Of these surveys — {asked} — which was completed EARLIEST? "
                "Reply with just the survey name.",
    # removes the temporal word entirely: pure numeric minimum
    "numeric":  "Which of these surveys — {asked} — has the LOWEST completion year? "
                "Reply with just the survey name.",
    # forces the comparison to be written out before committing
    "explicit": "For each of {asked}, write its completion year on one line. "
                "Then on a final line write only the name of the one with the smallest year.",
}

def prompt(it, variant="plain"):
    lines = "\n".join(f"The {n} survey was completed in {y}." for n, y in it["pairs"])
    asked = ", ".join(n for n, _ in it["pairs"])
    return f"{lines}\n\n" + VARIANTS[variant].format(asked=asked)

def run(a):
    items = jload(f"{D}/numcmp.jsonl")
    out = f"{D}/numcmp-{a.label}.jsonl"
    done = {r["id"] for r in jload(out)} if os.path.exists(out) else set()
    os.environ["KBENCH_EXTRA_JSON"] = json.dumps({"chat_template_kwargs": {"enable_thinking": False}})
    for it in items:
        if it["id"] in done: continue
        try: rep, dt = chat(a.endpoint, a.model, prompt(it, a.variant), 400)
        except Exception as e: rep, dt = f"ERROR {e}", 0
        names = [n for n, _ in it["pairs"]]
        hits = [(rep.lower().find(n.lower()), n) for n in names if n.lower() in rep.lower()]
        named = (max(hits)[1] if a.variant == "explicit" else min(hits)[1]) if hits else None
        order = [n for n, _ in sorted(it["pairs"], key=lambda p: p[1])]
        jappend(out, {"id": it["id"], "gap": it["gap"], "reply": rep[:200],
                      "ok": named == it["answer"],
                      "rank": order.index(named) + 1 if named in order else 0,
                      "secs": round(dt, 1)})
    print(f"[{a.label}] done")

def score(a):
    for lbl in a.label:
        rows = jload(f"{D}/numcmp-{lbl}.jsonl")
        if not rows: continue
        print(f"\n{lbl}: {sum(r['ok'] for r in rows)}/{len(rows)}")
        print("   gap:  " + "".join(f"{g:>8}y" for g in GAPS))
        print("   acc:  " + "".join(
            f"{sum(r['ok'] for r in rows if r['gap']==g)}/{len([r for r in rows if r['gap']==g]):<7}"
            for g in GAPS))
        import collections
        rk = collections.Counter(r["rank"] for r in rows if not r["ok"])
        print(f"   wrong picks by true rank (2=runner-up): {dict(sorted(rk.items()))}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run"); r.add_argument("--endpoint", required=True)
    r.add_argument("--model", default="x"); r.add_argument("--label", required=True)
    r.add_argument("--variant", default="plain", choices=list(VARIANTS))
    s = sub.add_parser("score"); s.add_argument("--label", nargs="+")
    a = ap.parse_args()
    {"build": lambda a: build(), "run": run, "score": score}[a.cmd](a)

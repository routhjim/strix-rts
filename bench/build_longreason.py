#!/usr/bin/env python3
"""Long-context REASONING items. The answer cannot come from one sentence, and it
cannot come from parametric knowledge -- the entities are invented. That is the
point: with web search on, what matters is reasoning over a context you were
GIVEN, not what you memorised.

Each item plants N records of the form
    "Record <id>: the <NAME> survey was completed in <YEAR>."
among SAME-SHAPE distractor records (different names, different years) and real
wiki prose filler.

  EARLIEST : "of A,B,C,D,E which was completed earliest?" -> answer is a NAME,
             so the model must compare years AND map back to the right label.
  COUNT    : "how many of A..E were completed before YYYY?" -> integer; missing
             any single record changes the answer, so partial retrieval fails.

Distractor records are asked-about-adjacent: same sentence shape, plausible years,
names NOT in the question. Grabbing the nearest record yields a wrong answer.
"""
import json, os, random, re
D = os.path.dirname(os.path.abspath(__file__))

NAMES = """Kestrel Marlin Osprey Petrel Tanager Vireo Merlin Bittern Godwit Curlew
Avocet Dunlin Redshank Sanderling Turnstone Whimbrel Brambling Chiffchaff Fieldfare
Goldcrest Hawfinch Linnet Nuthatch Redpoll Siskin Twite Wheatear Wryneck Yellowhammer
Bullfinch Crossbill Firecrest Greenshank Jackdaw Kittiwake Lapwing Nightjar Pochard
Quail Razorbill Shelduck Teal Wigeon Woodlark Corncrake Dotterel Egret Fulmar Gannet""".split()

def build(n_items=24, seed=11, n_planted=5, n_distract=8):
    rng = random.Random(seed)
    wiki = open(f"{D}/../wiki.test.raw", encoding="utf-8", errors="replace").read()
    wiki = wiki.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    wiki = re.sub(r"<unk>", "thing", wiki)
    paras = [p.strip() for p in wiki.split("\n")
             if 400 < len(p.strip()) < 2500 and not p.strip().startswith("=")]
    out = []
    for i in range(n_items):
        names = rng.sample(NAMES, n_planted + n_distract)
        planted = [{"name": n, "year": rng.randrange(1908, 2011)} for n in names[:n_planted]]
        # unique years so EARLIEST/COUNT are unambiguous
        seen = set()
        for p in planted:
            while p["year"] in seen: p["year"] = rng.randrange(1908, 2011)
            seen.add(p["year"])
        distract = [{"name": n, "year": rng.randrange(1908, 2011)} for n in names[n_planted:]]
        asked = [p["name"] for p in planted]
        if i % 2 == 0:
            kind = "EARLIEST"
            ans = min(planted, key=lambda p: p["year"])["name"]
            ask = (f"Of these surveys — {', '.join(asked)} — which was completed EARLIEST? "
                   f"Reply with just the survey name.")
        else:
            kind = "COUNT"
            # BUG FIXED 2026-08-20: cutting at the MEDIAN made the answer always
            # exactly 2 -- zero label variance, so "always say 2" scored 100%.
            # Pick the cut so the true count is uniform over 0..n_planted.
            ys = sorted(p["year"] for p in planted)
            k = rng.randrange(0, n_planted + 1)          # target count
            lo = ys[k-1] + 1 if k > 0 else 1900
            hi = ys[k] if k < n_planted else 2015
            cut = rng.randrange(lo, hi + 1) if lo <= hi else (ys[k-1] + 1 if k > 0 else 1900)
            ans = str(sum(1 for p in planted if p["year"] < cut))
            ask = (f"Of these surveys — {', '.join(asked)} — how many were completed "
                   f"BEFORE {cut}? Reply with just the number.")
        rec = lambda p, k: f"Record {k:04d}: the {p['name']} survey was completed in {p['year']}."
        out.append({
            "id": f"lr-{i}", "kind": kind, "answer": ans, "ask": ask, "asked": asked,
            "facts": [rec(p, 1000 + j) for j, p in enumerate(planted)],
            "distractor_facts": [rec(p, 2000 + j) for j, p in enumerate(distract)],
        })
    with open(f"{D}/longreason.jsonl", "w") as f:
        for o in out: f.write(json.dumps(o) + "\n")
    if not os.path.exists(f"{D}/filler.json"):
        json.dump(rng.sample(paras, min(400, len(paras))), open(f"{D}/filler.json", "w"))
    print(f"longreason.jsonl: {len(out)} items "
          f"({sum(o['kind']=='EARLIEST' for o in out)} EARLIEST / "
          f"{sum(o['kind']=='COUNT' for o in out)} COUNT), "
          f"{n_planted} planted + {n_distract} same-shape distractors each")
    for ex in out[:2]:
        print(f"\n[{ex['kind']}] answer={ex['answer']}\n  ask: {ex['ask'][:150]}")
        print(f"  fact: {ex['facts'][0]}")
        print(f"  DISTRACTOR: {ex['distractor_facts'][0]}")

if __name__ == "__main__":
    build()

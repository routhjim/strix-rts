#!/usr/bin/env python3
"""Build a deterministic noisy-retrieval corpus that implements the two findings
that actually matter for a web-search workload:

  1. HARD DISTRACTORS, not random noise. The Distracting Effect (arXiv 2505.06914):
     passages topically related but NOT containing the answer degrade accuracy;
     random unrelated passages do not. So distractors here are answers to OTHER
     trivia questions selected by keyword overlap with the target question --
     semantic near-misses, the kind a strong retriever surfaces.
  2. LENGTH HURTS EVEN WITH PERFECT RETRIEVAL (arXiv 2510.05381): degradation
     starts within ~7k tokens. So we sweep length with the gold passage ALWAYS
     present -- this is not a retrieval test, it is a "can you use what you were
     given" test.

Filler is real prose (wiki.test.raw), not lorem, so attention has plausible
competing material.
"""
import json, os, random, re, sys
D = os.path.dirname(os.path.abspath(__file__))

def jload(p):
    return [json.loads(l) for l in open(p) if l.strip()]

STOP = set("the a an of in on at to for and or is are was were which what who whom whose "
           "how why when where did does do this that these those with from by as it its "
           "his her their name named called first last most largest".split())

def atype(a):
    a = a.strip()
    if re.fullmatch(r"1[0-9]{3}|20[0-9]{2}", a):        return "year"
    if re.fullmatch(r"[\d,.]+", a):                     return "number"
    if len(a.split()) <= 3 and a[:1].isupper():         return "name"
    return "phrase"

def keywords(q):
    return {w for w in re.findall(r"[a-z]{4,}", q.lower()) if w not in STOP}

def build(n_items=40, seed=7):
    rng = random.Random(seed)
    trivia = jload(f"{D}/trivia.jsonl")
    # shortest alias = the crisp answer string to plant
    for t in trivia:
        t["gold"] = min((a for a in t["aliases"] if a), key=len)
        t["kw"] = keywords(t["question"])
    pool = [t for t in trivia if len(t["gold"]) <= 40 and len(t["kw"]) >= 2]
    picks = rng.sample(pool, n_items)

    wiki = open(f"{D}/../wiki.test.raw", encoding="utf-8", errors="replace").read()
    wiki = wiki.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    wiki = re.sub(r"<unk>", "thing", wiki)
    paras = [p.strip() for p in wiki.split("\n")            # wikitext uses single \n
             if 400 < len(p.strip()) < 2500 and not p.strip().startswith("=")]

    out = []
    for t in picks:
        # HARD distractors: same ANSWER TYPE (a year is distracted by other years,
        # not by an unrelated person) AND >=2 shared keywords. Type+topic overlap is
        # what makes a passage a semantic near-miss rather than obvious noise.
        cand = [u for u in pool
                if u["id"] != t["id"]
                and atype(u["gold"]) == atype(t["gold"])
                and len(u["kw"] & t["kw"]) >= 2
                and u["gold"].lower() != t["gold"].lower()]
        if len(cand) < 4:      # relax topic overlap before giving up type match
            cand = [u for u in pool
                    if u["id"] != t["id"] and atype(u["gold"]) == atype(t["gold"])
                    and (u["kw"] & t["kw"]) and u["gold"].lower() != t["gold"].lower()]
        rng.shuffle(cand)
        dis = cand[:12]
        out.append({
            "id": t["id"], "question": t["question"], "aliases": t["aliases"],
            "gold_fact": f"{t['question'].rstrip('?')}? The answer is {t['gold']}.",
            "distractor_facts": [f"{u['question'].rstrip('?')}? The answer is {u['gold']}."
                                 for u in dis],
            "n_shared_kw": [len(u["kw"] & t["kw"]) for u in dis],
        })
    with open(f"{D}/ragqa.jsonl", "w") as f:
        for o in out: f.write(json.dumps(o) + "\n")
    # filler snapshot so contexts are reproducible across models/runs
    rng2 = random.Random(seed + 1)
    with open(f"{D}/filler.json", "w") as f:
        json.dump(rng2.sample(paras, min(400, len(paras))), f)
    med = sorted(len(o["distractor_facts"]) for o in out)[len(out)//2]
    print(f"ragqa.jsonl: {len(out)} items, median {med} hard distractors available")
    print(f"filler.json: {min(400,len(paras))} wiki paragraphs "
          f"(mean {sum(len(p) for p in paras[:400])//400} chars)")
    ex = out[0]
    print(f"\nEXAMPLE\n  Q: {ex['question']}\n  gold: {ex['gold_fact']}")
    for d in ex["distractor_facts"][:3]: print(f"  dist: {d}")

if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 40)

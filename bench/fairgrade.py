#!/usr/bin/env python3
"""Bidirectional alias grader.

The strict matcher only asked "is an alias a substring of the reply?", which
rejected correct answers that were *shorter* or reordered relative to the alias
("Eva Cassidy" vs alias "Eva Marie Cassidy", "DNA structure" vs "The molecular
structure of DNA", "violin" vs "The Violin"). This checks both directions plus a
token-subset test, scored against the reply's final line (models often list
candidates before committing).
"""
import json, os, re, sys

STOP = {"the","a","an","of","and","in","on","at","to","for","is","was","by","its","it"}

def norm(s):  return re.sub(r"[^a-z0-9]", "", (s or "").lower())
def toks(s):  return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP}

def verdict_text(rep):
    """Models often enumerate options then commit; trust the last substantive line."""
    lines = [l.strip() for l in (rep or "").strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""

def hit(alias, rep):
    na, nr = norm(alias), norm(rep)
    if not na or not nr:
        return False
    if na in nr or nr in na:          # bidirectional containment
        return True
    ta, tr = toks(alias), toks(rep)
    return bool(ta) and bool(tr) and (ta <= tr or tr <= ta)   # token-subset either way

def is_correct(aliases, reply):
    return any(hit(a, reply) for a in aliases) or \
           any(hit(a, verdict_text(reply)) for a in aliases)

def score(path, gold):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = ok = 0
    lat = []
    for r in rows:
        g = gold.get(r.get("id") or r.get("q"))
        if not g:
            continue
        n += 1
        ok += is_correct(g, r.get("reply", ""))
        if r.get("secs"):
            lat.append(r["secs"])
    lat.sort()
    return n, ok, lat

if __name__ == "__main__":
    gold = {}
    here = os.path.dirname(os.path.abspath(__file__))
    for l in open(os.environ.get("TRIVIA", os.path.join(here, "trivia.jsonl"))):
        d = json.loads(l)
        gold[d["id"]] = d["aliases"]; gold[d["question"]] = d["aliases"]
    for p in sys.argv[1:]:
        try:
            n, ok, lat = score(p, gold)
        except FileNotFoundError:
            print(f"{p}: missing"); continue
        med = lat[len(lat)//2] if lat else 0
        mean = sum(lat)/len(lat) if lat else 0
        p95 = lat[int(len(lat)*0.95)-1] if len(lat) >= 20 else (lat[-1] if lat else 0)
        print(f"{p:34s} {ok:3d}/{n:<3d} = {100*ok/max(n,1):5.1f}%   "
              f"median {med:5.1f}s  mean {mean:5.1f}s  p95 {p95:5.1f}s")

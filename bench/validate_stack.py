#!/usr/bin/env python3
"""Best-of-both-worlds validation: accuracy AND latency, same 120 questions.

Ground truth: trivia.jsonl (TriviaQA subset with alias lists) — factual questions
where closed-book q38 previously scored 69.7%. If search+RTS works, accuracy should
rise sharply; the question is what it costs in seconds.

Arms:
  A closed-book       q38 direct, no search, no thinking      (the old daily driver)
  B closed+thinking   q38 direct, thinking medium             (thinking alone)
  C full stack        the RTS proxy: search -> A3B extract -> q38 answer + thinking
"""
import json, os, re, sys, time, urllib.request
D = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get("N", 120))
def jload(p): return [json.loads(l) for l in open(p) if l.strip()]
def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def ask(url, q, think, timeout=300):
    b = {"model":"x","messages":[{"role":"user","content":
         f"Answer this question with just the answer — no explanation.\n\nQ: {q}\nA:"}],
         "max_tokens": 2048 if think else 64, "temperature":0}
    b["chat_template_kwargs"] = {"enable_thinking": bool(think),
                                 **({"reasoning_effort":"medium"} if think else {})}
    if think: b["reasoning_effort"]="medium"
    r = urllib.request.Request(url+"/v1/chat/completions", data=json.dumps(b).encode(),
                               headers={"Content-Type":"application/json"})
    t0=time.time()
    j=json.load(urllib.request.urlopen(r, timeout=timeout))
    return (j["choices"][0]["message"].get("content") or ""), time.time()-t0

def run(label, url, think):
    out=f"{D}/val-{label}.jsonl"
    done={r["id"] for r in jload(out)} if os.path.exists(out) else set()
    qs=jload(f"{D}/trivia.jsonl")[:N]
    for q in qs:
        if q["id"] in done: continue
        try: rep,dt = ask(url, q["question"], think)
        except Exception as e: rep,dt = f"ERR {e}", 0
        ok = any(norm(a) and norm(a) in norm(rep) for a in q["aliases"])
        with open(out,"a") as f:
            f.write(json.dumps({"id":q["id"],"reply":rep[:400],"ok":ok,"secs":round(dt,2)})+"\n")
    rows=jload(out)
    acc=sum(r["ok"] for r in rows); tot=sum(r["secs"] for r in rows)
    import statistics
    print(f"  {label:18} {acc:3}/{len(rows)} = {100*acc/len(rows):5.1f}%   "
          f"median {statistics.median(r['secs'] for r in rows):5.1f}s   total {tot/60:5.1f} min", flush=True)

if __name__=="__main__":
    which=sys.argv[1] if len(sys.argv)>1 else "all"
    print(f"validating on {N} TriviaQA questions (closed-book q38 baseline was 69.7%)\n")
    if which in ("all","A"): run("A-closedbook",      "http://127.0.0.1:8080", False)
    if which in ("all","B"): run("B-closed+think",    "http://127.0.0.1:8080", True)
    if which in ("all","C"): run("C-fullstack",       "http://127.0.0.1:8090", False)  # cross-model: A3B extracts
    if which in ("all","D"): run("D-singlemodel",     "http://127.0.0.1:8091", False)  # q38 extracts for itself

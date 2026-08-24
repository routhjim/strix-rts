#!/usr/bin/env python3
"""RTS Proxy — an OpenAI-compatible endpoint that does search + cross-model
retrieve-then-solve, transparently, for ANY client (Open WebUI, OpenClaw, curl).

Flow per user turn:
  client ──/v1/chat/completions──▶ RTS proxy
     1. decide: does this turn need a web search?   (SEARCH_MODE valve)
     2. if yes → Ollama web_search  (query = last user msg)
     3. EXTRACT: A3B copies the lines relevant to the query   (938 t/s reader)
     4. SUFFICIENCY: A3B says YES/NO — NONE ⇒ answer says so, no hallucinated cite
     5. ANSWER: q38 reasons over the ~30-tok extract, STREAMS back to client
  if no search (or search/extract yields nothing) → straight pass-through to q38.

Measured 2026-08-24: A3B-extract + q38-answer = 3.4x faster cold-context @16k,
equal-or-better accuracy, vs q38 doing its own extraction. Falls back to
single-model automatically if the A3B endpoint is down.

Config via env:
  RTS_PORT           listen port (default 8090)
  ANSWER_ENDPOINT    q38 daily driver   (default http://127.0.0.1:8080)
  EXTRACT_ENDPOINT   A3B reader         (default http://127.0.0.1:18094; falls back to ANSWER)
  OLLAMA_KEY         Ollama cloud web_search bearer key (required for search)
  SEARCH_MODE        auto | always | off   (default auto)
  MIN_SEARCH_CHARS   auto-mode: only search turns longer than this (default 12)
"""
import json, os, re, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

PORT     = int(os.environ.get("RTS_PORT", 8090))
ANSWER   = os.environ.get("ANSWER_ENDPOINT",  "http://127.0.0.1:8080").rstrip("/")
EXTRACT  = os.environ.get("EXTRACT_ENDPOINT", "http://127.0.0.1:18094").rstrip("/")
OLLAMA_KEY = os.environ.get("OLLAMA_KEY", "")
SEARCH_MODE = os.environ.get("SEARCH_MODE", "auto")
# Thinking on the ANSWER call only. Measured 2026-08-22: thinking=medium took the
# multi-hop retrieval suite from 45/72 to 72/72 at ALL context lengths for +18%
# wall clock, and scored 180/180 under hard distractors — no inverse scaling.
# Extraction/sufficiency stay thinking-OFF: they are copy/classify jobs, and off
# is faster. xhigh is deliberately NOT used: +7pts on hard knowledge but 6x the
# tokens and 0-for-8 on agentic tasks.
ANSWER_THINK  = os.environ.get("ANSWER_THINK", "medium")   # medium | off | xhigh
MIN_SEARCH_CHARS = int(os.environ.get("MIN_SEARCH_CHARS", 12))
MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", 2400))  # per result; 6x2400~=3.6k tok
NO_EVIDENCE = "NONE"

def _post_json(url, payload, timeout=600, key=None):
    hdr = {"Content-Type": "application/json"}
    if key: hdr["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def _chat(endpoint, messages, max_tokens, temperature=0.0, think=False):
    body = {"model": "local", "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": bool(think),
                                     **({"reasoning_effort": "medium"} if think else {})}}
    j = _post_json(endpoint + "/v1/chat/completions", body)
    return (j["choices"][0]["message"].get("content") or "").strip()

def _endpoint_up(ep):
    try:
        urllib.request.urlopen(ep + "/health", timeout=2); return True
    except Exception:
        try: urllib.request.urlopen(ep + "/v1/models", timeout=2); return True
        except Exception: return False

# ---- pipeline stages ----
def web_search(query, n=6):
    if not OLLAMA_KEY: return []
    try:
        j = _post_json("https://ollama.com/api/web_search",
                       {"query": query, "max_results": n}, timeout=30, key=OLLAMA_KEY)
        return j.get("results", [])
    except Exception as e:
        print(f"[rts] search error: {e}"); return []

def format_docs(results):
    return "\n\n".join(f"[{i+1}] {r.get('title','')} — {r.get('url','')}\n{(r.get('content','') or '')[:MAX_DOC_CHARS]}"
                       for i, r in enumerate(results))

def extract(query, context, ep):
    prompt = (f"{context}\n\nQuestion: {query}\n\n"
              f"Do NOT answer the question. Copy out, word for word, the sentence(s) "
              f"from the documents above most relevant to it, each prefixed with its "
              f"[n] source number. At most 6 sentences. No summary, no rephrasing.")
    return _chat(ep, [{"role": "user", "content": prompt}], 400)

def sufficient(query, evidence, ep):
    v = _chat(ep, [{"role": "user", "content":
        f"Evidence:\n{evidence}\n\nQuestion: {query}\n\n"
        f"Does the evidence above actually answer the question? Reply YES or NO."}], 8)
    return v.strip().upper().startswith("Y")

def wants_search(text):
    if SEARCH_MODE == "off": return False
    if SEARCH_MODE == "always": return len(text.strip()) >= 1
    # auto: skip trivial/chatty turns; search substantive info-seeking ones
    t = text.strip()
    if len(t) < MIN_SEARCH_CHARS: return False
    if re.search(r"\b(who|what|when|where|which|why|how|latest|current|price|news|"
                 r"today|compare|vs\.?|release|version|score|benchmark)\b", t, re.I): return True
    return t.endswith("?")

# ---- HTTP ----
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _passthrough(self, raw):
        """No RTS: forward the original request to q38 verbatim, streaming."""
        req = urllib.request.Request(ANSWER + "/v1/chat/completions", data=raw,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1200) as up:
                self.send_response(200)
                for h in ("Content-Type",):
                    if up.headers.get(h): self.send_header(h, up.headers.get(h))
                self.end_headers()
                while True:
                    chunk = up.read(4096)
                    if not chunk: break
                    self.wfile.write(chunk); self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code); self.end_headers(); self.wfile.write(e.read())

    def do_GET(self):
        # expose model list so clients see one model
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"object":"list","data":[{"id":"rts-q38","object":"model","owned_by":"rts"}]}).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.end_headers(); self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try: body = json.loads(raw)
        except Exception: return self._passthrough(raw)
        msgs = body.get("messages", [])
        user = next((m["content"] for m in reversed(msgs) if m.get("role")=="user"), "")
        if not isinstance(user, str) or not wants_search(user):
            return self._passthrough(raw)

        # RTS path
        results = web_search(user)
        if not results:
            print("[rts] no results -> passthrough"); return self._passthrough(raw)
        ctx = format_docs(results)
        print(f"[rts] search={len(results)} ctx~{len(ctx)//4}tok")
        ext_ep = EXTRACT if _endpoint_up(EXTRACT) else ANSWER
        try:
            evidence = extract(user, ctx, ext_ep)
        except Exception as e:
            print(f"[rts] EXTRACT FAILED ({e}) -> passthrough"); return self._passthrough(raw)
        have = bool(evidence.strip()) and not evidence.strip().upper().startswith(NO_EVIDENCE) \
               and sufficient(user, evidence, ext_ep)
        print(f"[rts] evidence={len(evidence)}ch sufficient={have}")

        # Build the answer request: replace the user turn with evidence-grounded prompt.
        if have:
            grounded = (f"Use ONLY the evidence below to answer. Cite the [n] source for "
                        f"each claim. If it is insufficient, say so.\n\nEvidence:\n{evidence}"
                        f"\n\nQuestion: {user}")
        else:
            grounded = (f"A web search was run for this question but returned nothing that "
                        f"answers it. Say plainly that you couldn't find a reliable source, "
                        f"then answer from your own knowledge if you can, flagging uncertainty."
                        f"\n\nQuestion: {user}")
        ans_body = dict(body)
        ans_body["messages"] = msgs[:-1] + [{"role":"user","content":grounded}]
        if ANSWER_THINK != "off":
            kw = dict(ans_body.get("chat_template_kwargs") or {})
            kw.update({"enable_thinking": True, "reasoning_effort": ANSWER_THINK})
            ans_body["chat_template_kwargs"] = kw
            ans_body["reasoning_effort"] = ANSWER_THINK
            # a trace needs room; don't let a small client cap truncate the answer
            if int(ans_body.get("max_tokens") or 0) < 2048:
                ans_body["max_tokens"] = 2048
        print(f"[rts] answering (think={ANSWER_THINK})")
        return self._passthrough(json.dumps(ans_body).encode())

class S(ThreadingMixIn, HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    print(f"[rts] listening :{PORT}  answer={ANSWER}  extract={EXTRACT}  "
          f"search={SEARCH_MODE}  key={'set' if OLLAMA_KEY else 'MISSING'}")
    S(("127.0.0.1", PORT), H).serve_forever()

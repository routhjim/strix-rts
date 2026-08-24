#!/usr/bin/env bash
# Isolated BACKEND comparison. The earlier salted.sh arms differed in three ways
# at once (backend AND drafter AND llama.cpp version), so its 30.7-vs-15.6 gap
# could not be attributed to the backend. Here build (b10433) and drafter
# (MTP n4 + ngram) are held constant and only Vulkan-vs-HIP varies.
set -u
M=/home/jrouth/models; PORT=18093; R=/home/jrouth/therock/7.14.0
S=$M/dflash2/backend-clean-status.txt; : > "$S"
exec >>$M/dflash2/backend-clean.log 2>&1
arm(){ local label="$1" bin="$2"; shift 2
  for p in $(pgrep -x llama-server); do kill -9 $p 2>/dev/null; done; sleep 4
  env "$@" "$bin" -m $M/Qwen3.8-27B-Q4_K_S.gguf -md $M/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99 \
    --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-ngram-mod-n-min 24 \
    -ngl 99 -fa on --jinja --reasoning-format deepseek -rea off \
    --chat-template-file $M/qwen38-fixed-template.jinja \
    -c 32768 -b 4096 -ub 2048 -np 1 --host 127.0.0.1 --port $PORT > $M/dflash2/bc-$label.log 2>&1 &
  local SRV=$!
  for i in $(seq 1 90); do curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 5; done
  curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || { echo "  $label FAILED to start" >>"$S"; return 1; }
  grep -aq "Qwen3.8-27B-Q4_K_S.gguf" $M/dflash2/bc-$label.log || { echo "  $label ABORT wrong model" >>"$S"; kill -9 $SRV; return 1; }
  python3 - "$PORT" "$label" <<'PY' >> "$S" 2>&1
import json,urllib.request,sys,re
port,label=sys.argv[1:3]
base=json.load(open('/home/jrouth/models/longdoc-payload.json'))
rates=[];accs=[]
for i in (1,2,3):
    p=json.loads(json.dumps(base)); p['max_tokens']=400
    p['chat_template_kwargs']={"enable_thinking":False}
    p['messages'][0]['content']=f"[variant {label}-{i}] "+p['messages'][0]['content']
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    j=json.load(urllib.request.urlopen(r,timeout=900))
    rates.append((j.get('timings',{}) or {}).get('predicted_per_second',0))
    w=re.findall(r"[A-Za-z']+",j['choices'][0]['message']['content']); accs.append(len(set(w))/max(len(w),1))
print(f"  {label:14} salted x3: "+", ".join(f"{x:.1f}" for x in rates)
      +f"   median {sorted(rates)[1]:.1f}   uniq {min(accs):.2f}")
PY
  grep -aoE "draft acceptance = [0-9.]+" $M/dflash2/bc-$label.log | tail -1 | sed "s/^/  $label /" >> "$S"
  kill -9 $SRV 2>/dev/null; sleep 4
}
arm vk-mtp  /home/jrouth/llama.cpp-b10433/build/bin/llama-server     _=_
arm hip-mtp /home/jrouth/llama.cpp-b10433/build-hip/bin/llama-server LD_LIBRARY_PATH=$R/lib
echo DONE >> "$S"

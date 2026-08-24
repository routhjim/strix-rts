#!/usr/bin/env bash
# Qwen3.8-27B Q4_K_S + official MTP drafter — the throughput-winning config.
# Benched 2026-08-15: 27.6 t/s short-form / 29.9 t/s long-doc (AR is 13.2).
# Quality is ~= Q8 (MMLU 81.6 vs 81.0, hard tier 6/6 both) at +37% speed.
#
# LOAD-BEARING FLAGS -- do not "clean these up":
#   -fa on / f16 KV        q8 KV + MTP on RADV = Vulkan device-lost (llama.cpp #27076)
#   --chat-template-file   official 3.8 template CRASHES on enable_thinking:false
#   --spec-type draft-mtp  exact spelling; "mtp" silently no-ops
#   NO --spec-draft-p-min  gating lost EVERY arm ever tested on this box
#   -ngld 99               drafter fully on GPU
# n-max 4 is the Q4 optimum (Q8's is 5). -ub 256 is an UNTESTED possible +12% pp.
set -u
M=%HOME%/models
# 2026-08-22 DEFAULT: PR-27342 build + DFlash2 drafter + ngram-mod stack.
#   longdoc 30->38.6 t/s (+28%, 2-rep confirmed); freeform/code ~wash vs MTP stack.
#   DFlash2 uses its model-default block config — do NOT pass --spec-draft-n-max
#   (n7/n12/p-min all measured WORSE). MTP fallback: BIN=...b10433... DRAFT=mtp-...
BIN=${BIN:-%HOME%/llama.cpp-dflash2/build/bin/llama-server}
MODEL=${MODEL:-$M/Qwen3.8-27B-Q4_K_S.gguf}
DRAFT=${DRAFT:-$M/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}
TMPL=${TMPL:-$M/qwen38-fixed-template.jinja}
NMAX=${NMAX:-4}
UB=${UB:-2048}
PORT=${PORT:-8080}

# 2026-08-22: ngram-mod STACKED on the MTP drafter — strict win, no regressions:
# freeform 26.5->28.8, code-edit 43.6->66.5, longdoc 29.0->31.5 (n-min 24).
# n12 (KyaniteLabs) craters freeform to ~14 — keep NMAX=4.
case "$DRAFT" in
  *DFlash2*) SPEC=(-md "$DRAFT" -ngld 99 --spec-type draft-dflash,ngram-mod --spec-ngram-mod-n-min 24) ;;
  *)         SPEC=(-md "$DRAFT" -ngld 99 --spec-type draft-mtp,ngram-mod --spec-draft-n-max "$NMAX" --spec-ngram-mod-n-min 24) ;;
esac
[ -n "${NODRAFT:-}" ] && SPEC=()

exec %HOME%/bin/model-run qwen38 \
  "$BIN" -m "$MODEL" \
  "${SPEC[@]}" \
  -ngl 99 --no-mmap -fa on --jinja \
  --chat-template-file "$TMPL" \
  --reasoning-format deepseek \
  -c 65536 -b 4096 -ub "$UB" -np 1 -t 8 \
  --host 127.0.0.1 --port "$PORT" "$@"

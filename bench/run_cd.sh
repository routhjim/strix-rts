#!/usr/bin/env bash
# SEQUENTIAL by design: C and D share the q38 endpoint, so running them
# concurrently would measure GPU contention instead of architecture.
cd "$(dirname "$(readlink -f "$0")")"
echo "=== C: cross-model (A3B extracts, q38 answers) ==="
N=120 python3 -u validate_stack.py C
echo "=== D: single-model (q38 extracts for itself) ==="
N=120 python3 -u validate_stack.py D
echo "=== DONE ==="

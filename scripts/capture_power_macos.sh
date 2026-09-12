#!/bin/sh
set -eu

# Run interactively on macOS. powermetrics requires administrator authorization.
# The output is telemetry, not cross-device-calibrated power evidence.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT="$ROOT/results/m4-stage2/power"
mkdir -p "$OUT"

sudo -v
sudo powermetrics \
  --sample-rate 200 \
  --sample-count 80 \
  --samplers cpu_power,gpu_power,thermal \
  --show-usage-summary \
  --output-file "$OUT/powermetrics.txt" &
SAMPLER_PID=$!
trap 'sudo kill -TERM "$SAMPLER_PID" 2>/dev/null || true' EXIT INT TERM
sleep 1

"$ROOT/third_party/llama.cpp/build/bin/llama-bench" \
  --offline \
  -m "$ROOT/models/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
  -p 0 -n 512 -r 5 --delay 1 -t 8 -b 2048 -ub 512 -ngl 99 -o json \
  > "$OUT/q4_metal_decode.json"

wait "$SAMPLER_PID"
trap - EXIT INT TERM
echo "Wrote $OUT/powermetrics.txt and $OUT/q4_metal_decode.json"

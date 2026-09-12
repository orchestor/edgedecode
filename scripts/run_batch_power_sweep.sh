#!/bin/bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
OUT_DIR="$ROOT/results/m4-stage2/power/batch-sweep"
mkdir -p "$OUT_DIR"

MODEL_PATH="$ROOT/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
METAL_BIN="$ROOT/third_party/llama.cpp/build/bin/llama-bench"
CPU_BIN="$ROOT/third_party/llama.cpp/build-cpu/bin/llama-bench"

BATCHES=(1 2 4 8 16)
PROMPT=128
GEN=64
THREADS=8
REPEATS=3

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -x "$METAL_BIN" || ! -x "$CPU_BIN" ]]; then
  echo "Expected llama-bench binaries not found." >&2
  echo "Check the pinned builds under third_party/llama.cpp" >&2
  exit 1
fi

if sudo -n true >/dev/null 2>&1; then
  POWER_ON=1
else
  POWER_ON=0
  echo "Power capture disabled: sudo requires an interactive password prompt."
  echo "The benchmark will still run, but power samples will be skipped."
fi

for batch in "${BATCHES[@]}"; do
  for backend in cpu metal; do
    case "$backend" in
      cpu)
        BIN="$CPU_BIN"
        NGL=0
        ;;
      metal)
        BIN="$METAL_BIN"
        NGL=99
        ;;
      *)
        echo "Unknown backend: $backend" >&2
        exit 1
        ;;
    esac

    LABEL="batch-${batch}-${backend}"
    POWER_FILE="$OUT_DIR/${LABEL}.powermetrics.txt"
    BENCH_FILE="$OUT_DIR/${LABEL}.bench.json"
    STDERR_FILE="$OUT_DIR/${LABEL}.stderr.txt"

    echo "==> $LABEL"

    if [[ "$POWER_ON" -eq 1 ]]; then
      echo "Starting powermetrics capture..."
      sudo powermetrics \
        --sample-rate 200 \
        --sample-count 80 \
        --samplers cpu_power,gpu_power,thermal \
        --show-usage-summary \
        --output-file "$POWER_FILE" &
      POW_PID=$!
      sleep 1
      trap 'sudo kill -TERM "$POW_PID" 2>/dev/null || true' EXIT INT TERM
    else
      POW_PID=""
    fi

    "$BIN" \
      --offline \
      -m "$MODEL_PATH" \
      -p "$PROMPT" \
      -n "$GEN" \
      -r "$REPEATS" \
      --delay 1 \
      -t "$THREADS" \
      -b "$batch" \
      -ub "$batch" \
      -ngl "$NGL" \
      -o json \
      > "$BENCH_FILE" 2> "$STDERR_FILE"

    if [[ -n "${POW_PID:-}" ]]; then
      wait "$POW_PID" || true
      trap - EXIT INT TERM
    fi

    echo "Saved benchmark output to $BENCH_FILE"
    if [[ "$POWER_ON" -eq 1 ]]; then
      echo "Saved power output to $POWER_FILE"
    fi
  done
done

echo "Batch sweep complete. Results are in $OUT_DIR"

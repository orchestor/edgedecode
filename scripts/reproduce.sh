#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
"$PYTHON" scripts/build_native.py
"$PYTHON" -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  "$PYTHON" -m edgedecode.experiment --out results/local --backends cpu --repeats 9
"$PYTHON" -m edgedecode.architecture --out results/local/architecture.json
"$PYTHON" -m edgedecode.npu_contract > results/local/npu-contract.json

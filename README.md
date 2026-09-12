# EdgeDecode

EdgeDecode is a research prototype for studying where and why LLM inference shifts between CPU and GPU-like execution paths on edge-class systems.

It focuses on a simple but important question:

> When does a model's prefill phase behave differently from its decode phase, and how do model scale, context length, and concurrency change the hardware boundary?

This repository combines three layers of evidence:

- real model execution with GGUF + llama.cpp
- reference operator measurements on CPU and Metal
- synthetic architecture exploration to reason about future hardware choices

Documentation: [中文项目说明](docs/PROJECT.zh-CN.md) · [Working paper](paper/main.pdf) · [Reproduction guide](docs/REPRODUCIBILITY.md) · [Evidence ledger](docs/EVIDENCE.md) · [Format notes](docs/FORMAT.md)

## Why this project exists

The central issue is not "CPU vs GPU" in the abstract.

The real problem is that LLM inference is not one uniform workload. It has at least two distinct phases:

- Prefill: process a long prompt or context in bulk
- Decode: generate tokens one by one

These phases stress hardware differently. In practice, the best execution path depends on:

- model size
- prompt/context length
- generation length
- service concurrency
- software blocking and scheduling behavior
- quantization format and weight packing

EdgeDecode is designed to make this visible and auditable instead of hiding it behind a single headline claim.

## What the project measures

The repository contains evidence for several layers of the stack:

- real GGUF execution on local Qwen models via llama.cpp
- CPU and Metal reference kernel measurements
- packed low-precision weight experiments
- synthetic composed-network architecture exploration
- stage-2 summaries that separate measured results from hypothetical scenarios

On the M4 Pro host used in this project, the measured results show that decode often benefits more strongly from low-precision formats, while prefill can favor a higher-precision path depending on the backend. The key point is not that one format is universally best; it is that the winning choice depends on which phase is dominating the workload.

## Repository status

This project intentionally keeps evidence levels separate:

- actual measurements
- quality pilots and model-level checks
- hypothetical architecture scenarios

The last category is explicitly labeled as a design-analysis tool, not as a direct product prediction or a measured silicon result.

## Quick start

Requirements: Python 3.10+, NumPy, and a local C/C++ toolchain.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
sh scripts/reproduce.sh
```

This runs the local reproduction flow and writes outputs under `results/` without overwriting the committed benchmark evidence.

## Stage 2 real-model workflow

The real-model benchmark protocol is configured in `configs/stage2.json`.

After obtaining the required model files, datasets, and a local llama.cpp checkout, run:

```bash
python scripts/run_stage2.py --mode all
python -m edgedecode.stage2_analysis
```

The repository checks input SHA-256 values before execution and retains raw logs and measurements in `results/m4-stage2/`.

## Optional Metal experiment

```bash
python scripts/build_native.py --metal
python -m edgedecode.experiment --out results/local-metal --backends cpu metal
```

A missing device is recorded as skipped rather than silently counted as a successful run. The project is careful not to overstate Apple Metal as a general Arm GPU result.

## Minimal operator entry point

You can run a single tensor-level operator check with a local `.npz` file containing FP32-compatible arrays:

```bash
python -m edgedecode.experiment --tensors my_operator.npz \
  --skip-search --out results/local --backends cpu
```

This is an operator test, not a full LLM benchmark.

## Project layout

```text
edgedecode/          Core benchmark and analysis code
native/              Native CPU/Metal reference implementations
configs/             Benchmark and scenario configuration files
results/             Measured and synthetic evidence outputs
scripts/             Reproducibility and experiment orchestration
tests/               Validation and regression checks
paper/               LaTeX paper source and generated PDF
docs/                Project design, evidence, and formatting notes
models/              Local GGUF model files used by the benchmark flow
third_party/         Pinned llama.cpp checkout
```

## Research message

This project does not claim a universal statement like "quantization always helps" or "GPU always wins."

Its more defensible conclusion is:

- prefill and decode are different workloads
- the execution boundary moves with model size, prompt length, and service concurrency
- low-precision formats can help decode strongly, while prefill may still favor higher-precision or different execution paths
- architectural recommendations should therefore be phase-aware, not one-size-fits-all

## Paper and reproduction

```bash
python -m pip install -r requirements-paper.txt
make paper
```

The manuscript is maintained in `paper/main.tex` and compiled locally with LaTeX.

## License

MIT license for the project code and documentation.

The repository intentionally does not bundle third-party model weights or datasets. Any external artifacts must be obtained separately as required by their licenses.

## Important note

This is a research artifact, not a product claim. It is designed for auditability, careful reproduction, and clear separation between measured results and hypothetical architecture reasoning.

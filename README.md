# EdgeDecode

EdgeDecode is a research artifact for phase-aware LLM inference on edge-class systems.

It studies how model scale, prompt length, generation length, and scheduling behavior change the CPU/GPU boundary during inference.

## At a glance

This repository is built around measured evidence from a local Apple M4 Pro host and a Qwen2.5-1.5B-Instruct GGUF benchmark path.

| Metric | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| Model size (bytes) | 3,560,416,288 | 1,894,532,128 | 1,117,320,736 |
| Compression vs F16 | 1.0x | 1.88x | 3.19x |
| WikiText-2 perplexity | 8.9573 | 8.9524 | 9.3102 |
| HellaSwag accuracy | 61.5% | 61.5% | 60.0% |
| CPU decode throughput (tg128) | 40.95 tok/s | 70.81 tok/s | 113.99 tok/s |
| Metal decode throughput (tg128) | 62.78 tok/s | 98.76 tok/s | 121.29 tok/s |
| CPU prefill throughput (pp128) | 325.69 tok/s | 449.84 tok/s | 423.20 tok/s |
| Metal prefill throughput (pp128) | 1629.82 tok/s | 1591.19 tok/s | 1406.20 tok/s |

## Key finding

The measured results do not support a universal recommendation of one quantized format for all phases.

- On CPU decode, Q4_K_M is the fastest path.
- On Metal decode, Q4_K_M is also the fastest path.
- On Metal prefill, F16 remains the fastest path.
- Q8_0 matches F16 quality in the measured pilot while reducing model size substantially.
- Q4_K_M has a measurable quality penalty, but it is still highly attractive for decode-heavy workloads.

This is exactly the phase-sensitive behavior that motivates the project.

## Figure gallery

The project includes figure assets and manuscript output that summarize the measured and synthetic evidence.

- [Architecture view](paper/figures/architecture.pdf)
- [Operator view](paper/figures/operator.pdf)
- [Search-space view](paper/figures/search.pdf)
- [Stage-2 performance summary](paper/figures/stage2-performance.pdf)
- [Working paper](paper/main.pdf)

## Measured result snapshots

### CPU performance

| Phase | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| pp128 | 325.69 tok/s | 449.84 tok/s | 423.20 tok/s |
| pp2048 | 516.13 tok/s | 419.67 tok/s | 357.34 tok/s |
| tg128 | 40.95 tok/s | 70.81 tok/s | 113.99 tok/s |

### Metal performance

| Phase | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| pp128 | 1629.82 tok/s | 1591.19 tok/s | 1406.20 tok/s |
| pp2048 | 1872.16 tok/s | 1530.36 tok/s | 1287.24 tok/s |
| tg128 | 62.78 tok/s | 98.76 tok/s | 121.29 tok/s |

## Quality and model fidelity

| Format | WikiText-2 PPL | Delta vs F16 | HellaSwag Acc | Delta vs F16 |
|---|---:|---:|---:|---:|
| F16 | 8.9573 | 0.00% | 61.5% | 0.0 pt |
| Q8_0 | 8.9524 | -0.05% | 61.5% | 0.0 pt |
| Q4_K_M | 9.3102 | +3.94% | 60.0% | -1.5 pt |

## Research interpretation

The project does not claim that one backend or one quantization format is universally optimal.

Instead, the measured evidence supports a more conservative conclusion:

> Precision choice is phase- and backend-dependent.

The practical implication is to separate the architecture into at least two design targets:

1. a dense, high-throughput prefill path
2. a separate decode path that is optimized for weight streaming and token generation

This is the design signal that emerges from the data, rather than a universal low-precision rule.

## Evidence scope and limits

This project is intentionally explicit about what it does and does not claim.

- Measured on one Apple M4 Pro host
- One 1.5B model family
- Real GGUF + llama.cpp execution path
- No measured Arm GPU/NPU results
- No measured silicon area claims
- No product-level power recommendation from this artifact alone

The repository keeps measured results, quality pilots, and hypothetical architecture scenarios distinct.

## Repository structure

```text
edgedecode/          Core analysis and benchmark logic
native/              CPU and Metal reference code
configs/             Benchmark and scenario configuration files
results/             Measured and synthetic evidence outputs
scripts/             Reproduction and experiment orchestration
tests/               Validation and regression checks
paper/               LaTeX paper source and generated PDF
docs/                Evidence, reproduction, and format notes
models/              Local GGUF model files used in the benchmark flow
third_party/         Pinned llama.cpp checkout
```

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
sh scripts/reproduce.sh
```

Optional Metal evaluation:

```bash
python scripts/build_native.py --metal
python -m edgedecode.experiment --out results/local-metal --backends cpu metal
```

## Reproduction notes

The project includes a real-model workflow using the GGUF benchmark stack and logs the measured outputs under `results/m4-stage2/`.

For the paper build:

```bash
python -m pip install -r requirements-paper.txt
make paper
```

## License

MIT.

This project intentionally does not package third-party weights or datasets. Any external model or data artifacts must be obtained separately under their respective licenses.

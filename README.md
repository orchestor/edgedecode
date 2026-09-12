# EdgeDecode

**An auditable research prototype connecting low-precision model quality, execution phases, and architecture decisions.**

[中文项目说明](docs/PROJECT.zh-CN.md) · [Working paper](paper/main.pdf) · [Reproduction](docs/REPRODUCIBILITY.md) · [Evidence ledger](docs/EVIDENCE.md) · [Format](docs/FORMAT.md)

## What is actually demonstrated?

- A pinned Qwen2.5-1.5B-Instruct experiment with official F16, Q8_0 and Q4_K_M GGUF files.
- 18 end-to-end llama.cpp measurements: CPU-only and Metal, prefill at 128/2048 tokens and 128-token generation, seven repetitions each.
- Model-level quality pilots: 32 WikiText-2 chunks and 200 fixed-seed HellaSwag tasks.
- Real packed 4-bit and 8-bit weights consumed by native CPU and Metal reference kernels, plus 56 operator configurations measured on this host.
- Exhaustive, development-quality-constrained selection over 216 precision/group configurations on each of three synthetic composed networks.
- 108 **uncalibrated hypothetical** architecture scenarios, with explicit throughput, energy and area-budget assumptions.
- Stage 2 evidence and derivations are machine-readable under `results/m4-stage2/`.

On the measured M4 Pro, Q4_K_M improves 128-token decode throughput over F16 by **1.95x on CPU** and **2.25x on Metal**. Metal prefill shows the opposite ranking: F16 is fastest. Q8_0 matches the F16 quality pilot; Q4_K_M raises WikiText-2 perplexity by 3.94% and is 1.5 HellaSwag points lower, while the 200-task confidence intervals overlap. **There are no measured Arm GPU/NPU results, energy results, or synthesized silicon area in this release.** Apple Metal is used as a real unified-memory accelerator study, not relabeled as an Arm GPU.

## Research question

When do smaller quantization groups justify their metadata and execution costs, and when could a dedicated decode path be worth architectural investment?

The artifact keeps three evidence levels separate: actual execution measurements; synthetic-network quality proxies; and hypothetical architecture scenarios. The latter are not fitted to the native reference kernel and must not be cited as M4 or Arm product predictions.

## Quick start

Python 3.10+, NumPy, and Clang. Use a source checkout; native libraries are built locally and are not distributed as wheels.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
sh scripts/reproduce.sh
```

This builds the CPU backend, runs tests, reruns the pilot into `results/local`, and writes the scenario and contract outputs. It never overwrites the committed `results/pilot` evidence. No model download, API key, GPU, or internet access is required after dependency installation.

The real-model Stage 2 protocol is configured in `configs/stage2.json`. Model weights, datasets and the pinned llama.cpp checkout are intentionally ignored. After acquiring the exact files listed in the config and building llama.cpp, run:

```bash
python scripts/run_stage2.py --mode all
python -m edgedecode.stage2_analysis
```

Every input SHA-256 is checked before execution. Commands, raw logs, individual timing samples and limitations are retained in `results/m4-stage2/`.

Verified generation smoke test (local 7B path): the repository contains a split Qwen2.5-7B Q4_K_M GGUF in `models/`, and a local llama.cpp build successfully generated a one-sentence answer from the prompt: "Write a one-sentence description of EdgeDecode." The produced output was: "EdgeDecode is an edge computing solution that decodes and processes media content locally, reducing latency and improving performance for real-time applications." This confirms the actual generation stack works; the lightweight `qwen3_agent` config remains a harness and is not a substitute for a real GGUF + llama.cpp benchmark run.

Optional macOS GPU experiment:

```bash
python scripts/build_native.py --metal
python -m edgedecode.experiment --out results/local-metal --backends cpu metal
```

A missing GPU device is recorded as skipped, not counted as a successful measurement. The Apple GPU is not an Arm GPU. The CPU kernel contains no explicit SME/NEON intrinsics; compiler-generated code has not been instruction-profiled.

## A real tensor entry point

Supply a local `.npz` with finite FP32-compatible `weight[M,K]` and `activation[N,K]` arrays:

```bash
python -m edgedecode.experiment --tensors my_operator.npz \
  --skip-search --out results/local --backends cpu
```

The file is loaded with `allow_pickle=False`; its SHA-256 is recorded. This tests one operator, not an entire LLM. Never call the synthetic pilot a checkpoint-derived workload. See the [next-evidence plan](docs/NEXT_EXPERIMENTS.md) for model-level validation.

## Repository map

| Path | Purpose |
|---|---|
| `edgedecode/quant.py` | Format, packing, unpacking and error metric |
| `native/cpu.c` | Fused packed-weight reference operator and F32 baseline |
| `native/linear.metal`, `native/metal.m` | Optional Metal source and persistent-buffer host wrapper |
| `edgedecode/experiment.py` | Operator measurements and composed-network search |
| `edgedecode/architecture.py` | Ideal-overlap throughput and energy scenarios |
| `edgedecode/npu_contract.py` | Static hypothetical compatibility check, not a compiler |
| `configs/` | Fully disclosed hypothetical parameters |
| `results/pilot/` | Synthetic operator/search evidence and assumptions |
| `results/m4-stage2/` | Real-model CPU/Metal performance and quality evidence |
| `paper/` | English working paper, LaTeX source and generated vector figures |
| `tests/` | Format, native numerical, constraint and model-unit checks |

## Paper and figures

```bash
python -m pip install -r requirements-paper.txt
make paper
```

A TeX distribution with `pdflatex` is required. Figures and tables are generated from committed JSON; no numerical results are manually invented. The manuscript is an unattributed project working draft. Set genuine author names and affiliations before an external submission. It has not been submitted to arXiv or peer-reviewed.

## Architecture verdict from Stage 2

The result rejects a universal low-precision execution recommendation. On this workload, weight compression strongly benefits single-stream decode, while Metal prefill favors its F16 path. The defensible architecture proposal is therefore phase-specific: preserve a dense, high-throughput prefill path and separately optimize weight-streaming decode. This is a hypothesis for future Arm CPU/GPU/NPU validation, not an M4-to-Arm product extrapolation.

## License and provenance

Original code and documentation: MIT. External papers are referenced, not redistributed. No third-party model weights or datasets are included. Prepared with AI coding/writing assistance; human review is needed before making scientific or application claims. No Arm affiliation or endorsement is implied.

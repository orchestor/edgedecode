# Validation record

## Stage 2 M4 Pro real-model experiment

- Host: Apple M4 Pro, 12 CPU cores, 16-core Apple GPU, 24 GB unified memory.
- Software: pinned llama.cpp commit and source-archive hash in `configs/stage2.json`.
- Inputs: official Qwen2.5-1.5B-Instruct F16/Q8_0/Q4_K_M GGUF variants; all model and dataset hashes verified before execution.
- Performance: 18 cells, seven samples each, spanning CPU-only/Metal, 128/2048-token prefill and 128-token generation.
- Quality: 32 WikiText-2 chunks and 200 fixed-seed HellaSwag tasks per model.
- Derived summary: regenerated from raw JSON with `python -m edgedecode.stage2_analysis`.
- Power: unavailable in the automated run because `powermetrics` requires interactive administrator authorization. `scripts/capture_power_macos.sh` provides a bounded manual capture.
- Area and NPU: not measured.

Validated on September 10, 2026 in the supplied local execution environment.

## Code and evidence

- Native CPU and optional Metal libraries compiled with Apple Clang 17.0.0.
- `python -m unittest discover -s tests -v`: 23 tests executed; 22 passed and one optional Metal runtime test skipped because the process had no accessible Metal device.
- The packed CPU kernel was compared against dequantized NumPy references across both bitwidths, two group sizes, tail dimensions, and multiple batch sizes.
- The committed pilot contains 28 CPU timing records, 28 explicit Metal skips, 15 reported network selections, every candidate from three 216-point searches, and 108 hypothetical architecture rows.
- The exhaustive-selection test independently checks that each reported hardware-aware choice minimizes the disclosed proxy over its feasible candidate set.

## Paper

- `paper/main.tex` compiled twice with pdfLaTeX into a nine-page PDF.
- The final LaTeX log contains no unresolved citation/reference, overfull-box, or underfull-box warning.
- The PDF was reopened with pypdf; all nine pages yielded extractable text and the principal evidence statements were present.
- All nine pages were rendered to PNG. A contact-sheet review and full-page review of the result and limitation pages found no clipping, overlap, broken tables, or illegible figures.

## Boundaries

The host architecture is observed as arm64. The Apple M4 Pro and 24 GB description is user-reported because the restricted process could not query those system fields. No GPU execution, NPU compilation, model-checkpoint evaluation, power measurement, or RTL synthesis was completed. Those omissions are stated in the README, evidence ledger, and paper.

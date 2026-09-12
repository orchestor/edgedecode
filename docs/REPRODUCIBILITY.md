# Reproducibility protocol

## Scope

Committed results describe one local, hot-input pilot. They are not cross-machine comparable peak-throughput claims. The supplied run used NumPy 2.3.5 and Python 3.12; exact Python/OS/compiler data are recorded in the manifest. Device branding is user-reported.

## Commands

From the repository root, install NumPy (`pip install -e .`) and run `sh scripts/reproduce.sh`. For the exact tested NumPy version use `pip install -r requirements-experiment.txt`. Native CPU build requires Clang; GPU build additionally requires macOS Metal and Foundation.

CPU tests run on GitHub Actions Linux. GPU absence causes an explicit skip. A CPU build is required before native tests; standalone test invocation without a build may skip those tests, so use `make test` or the reproduction script for acceptance.

## Data and statistics

Operator weights are synthetic Gaussian matrices with every 31st input column amplified by 4. Four shapes: (M,K,N)=(256,512,1),(256,512,16),(1024,1024,1),(1024,1024,16). Seven formats per shape: F32 and 4/8-bit with G=32,64,128. Configuration blocks are shuffled with a fixed seed; each block has two warmups and nine timed hot-input repetitions. Report median and interquartile range, not a confidence interval. Thermal state, core affinity and DVFS were not controlled. C and NumPy have different reduction orders; correctness tolerances accommodate FP32 accumulation differences.

Network weights have seeds 11,23,37 and shapes (192,256),(128,192),(64,128), with tanh after the first two layers. Development inputs: 96 rows with seed+1000. Test inputs: 256 rows with seed+2000. Relative MSE uses full final outputs. Threshold 0.005 and storage <=30% of F32 are fixed before search. There is no checkpoint training and this is not a language task.

Search enumerates 6^3=216 full-network configurations per seed. The additive latency proxy uses measured N=1 CPU per-layer times; final composed times include overhead absent from this proxy. Five selections are reported: quality-only, Q4/G128, Q8/G64, hardware-aware, random feasible. Identical configurations can appear under different labels and have separately measured times; their differences indicate measurement variation, not different algorithms. The random control is sampled from quality-and-memory feasible candidates, not exactly byte-matched.

## Provenance

`manifest.json` records source SHA-256 values and sanitized build commands. `operators.json` records skipped attempts as well as results. `candidates.json` contains the complete search ledger. Scenario inputs are embedded in the architecture result. `scripts/paper_assets.py` generates tables and figures directly from these files. No weights or private local paths are needed in a public repository.

## Model tensors

The NPZ adapter accepts operator weights and activations only, not arbitrary pickled objects. Document checkpoint ID/revision/license, tensor name, activation provenance, shape and train/dev/test splits separately. The adapter records a file hash; it cannot establish provenance on its own.

## Publication checklist

Review claims, add real authors/affiliations, select a repository URL, and update citation metadata before submitting externally. The existing paper is a local working draft, not an arXiv submission. Verify every claim against the evidence ledger; do not replace missing measurements with scenario numbers.

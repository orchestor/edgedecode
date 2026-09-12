# Contributing

Run `make test`. Add correctness tests for new formats, including zero groups, tails and invalid input. Preserve explicit measurement scope and data provenance. New hardware results must include build information, numerical validation, raw samples and skipped/failed runs. Separate measured results from calibrated predictions and uncalibrated scenarios.

Do not silently regenerate committed pilot results or paper claims. Add a new run directory and explain the experimental change. No model weights, credentials, private datasets, binaries, or unrelated personal paths belong in a contribution. CI is correctness-focused and does not enforce performance thresholds across machines.

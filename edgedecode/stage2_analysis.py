"""Derive reviewable Stage 2 metrics from committed llama.cpp evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_ORDER = ("F16", "Q8_0", "Q4_K_M")
BACKEND_ORDER = ("cpu", "metal")


def _bench_key(row: dict) -> str:
    if row["n_prompt"] and not row["n_gen"]:
        return f"pp{row['n_prompt']}"
    if row["n_gen"] and not row["n_prompt"]:
        return f"tg{row['n_gen']}"
    raise ValueError("benchmark row is neither pure prompt processing nor generation")


def summarize(benchmark: dict, quality: dict, model_files: dict) -> dict:
    if isinstance(model_files, list):
        model_files = {item["label"]: item for item in model_files}
    rows = benchmark["rows"]
    lookup = {
        (r["requested_backend"], r["model_label"], _bench_key(r)): r for r in rows
    }
    expected = 2 * 3 * 3
    if len(rows) != expected or len(lookup) != expected:
        raise ValueError(f"expected {expected} unique benchmark rows")

    f16_bytes = model_files["F16"]["bytes"]
    models = {}
    ppl = {r["model_label"]: r for r in quality["perplexity"]}
    hs = {r["model_label"]: r for r in quality["hellaswag"]}
    for model in MODEL_ORDER:
        models[model] = {
            "file_bytes": model_files[model]["bytes"],
            "size_fraction_of_f16": model_files[model]["bytes"] / f16_bytes,
            "compression_vs_f16": f16_bytes / model_files[model]["bytes"],
            "perplexity": ppl[model]["ppl"],
            "perplexity_delta_percent_vs_f16":
                100 * (ppl[model]["ppl"] / ppl["F16"]["ppl"] - 1),
            "hellaswag_acc_norm_percent": hs[model]["acc_norm_percent"],
            "hellaswag_delta_points_vs_f16":
                hs[model]["acc_norm_percent"] - hs["F16"]["acc_norm_percent"],
            "hellaswag_ci95_percent": hs[model]["ci95_percent"],
        }

    performance = {}
    for backend in BACKEND_ORDER:
        performance[backend] = {}
        for test in ("pp128", "pp2048", "tg128"):
            baseline = lookup[(backend, "F16", test)]["avg_ts"]
            performance[backend][test] = {
                model: {
                    "tokens_per_second": lookup[(backend, model, test)]["avg_ts"],
                    "stddev_tokens_per_second": lookup[(backend, model, test)]["stddev_ts"],
                    "speedup_vs_f16": lookup[(backend, model, test)]["avg_ts"] / baseline,
                    "repetitions": len(lookup[(backend, model, test)]["samples_ts"]),
                }
                for model in MODEL_ORDER
            }

    return {
        "experiment_id": benchmark["experiment_id"],
        "models": models,
        "performance": performance,
        "decision": {
            "observation": (
                "Precision choice is phase- and backend-dependent: Q4_K_M is fastest "
                "for single-stream decode, while F16 is fastest for Metal prefill."
            ),
            "quality_gate": (
                "Q8_0 matches the F16 pilot; Q4_K_M has 3.94% higher WikiText-2 "
                "perplexity and 1.5-point lower HellaSwag accuracy, with overlapping "
                "HellaSwag intervals at 200 tasks."
            ),
            "architecture_implication": (
                "Do not propose one universal low-precision path. Preserve a high-throughput "
                "dense prefill path and separately optimize weight-streaming decode."
            ),
        },
        "evidence_limits": [
            "One 1.5B model family and one Apple M4 Pro host.",
            "Apple Metal results are not Arm GPU results.",
            "No measured power, silicon area, Arm NPU, or long-context KV-cache sweep.",
            "The WikiText-2 run uses 32 chunks and HellaSwag uses 200 tasks.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.json")
    parser.add_argument("--results", default="results/m4-stage2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / args.config).read_text())
    result_dir = root / args.results
    summary = summarize(
        json.loads((result_dir / "benchmark.json").read_text()),
        json.loads((result_dir / "quality.json").read_text()),
        cfg["models"],
    )
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(result_dir / "summary.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a systematic model-scale × concurrency × context sweep using a pinned llama.cpp build.

This script measures prompt processing (prefill) and single-stream decode throughput
across a small but disciplined matrix of model size, prompt context, and batch size.
The outputs are saved under results/m4-stage2/scale-sweep and are designed to support
phase-boundary analysis for the final architecture claim.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "m4-stage2" / "scale-sweep"

MODEL_SPECS = {
    "qwen2.5-1.5b": {
        "path": ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "contexts": [128, 512, 2048],
        "batches": [1, 2, 4, 8, 16],
        "gen_tokens": 128,
    },
    "qwen2.5-7b": {
        "path": ROOT / "models" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "contexts": [128, 512, 1024],
        "batches": [1, 2, 4, 8],
        "gen_tokens": 128,
    },
}

BACKENDS = {
    "cpu": {"binary": ROOT / "third_party" / "llama.cpp" / "build-cpu" / "bin" / "llama-bench", "ngl": 0},
    "metal": {"binary": ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-bench", "ngl": 99},
}


def build_case_table() -> list[dict]:
    """Return a single matrix of benchmark cases for prompt-processing and generation."""
    cases: list[dict] = []
    for model_name, spec in MODEL_SPECS.items():
        for backend_name, backend in BACKENDS.items():
            for context in spec["contexts"]:
                for batch in spec["batches"]:
                    cases.append(
                        {
                            "model": model_name,
                            "backend": backend_name,
                            "phase": "prefill",
                            "context": context,
                            "batch": batch,
                            "gen_tokens": 0,
                            "binary": str(backend["binary"]),
                            "ngl": backend["ngl"],
                            "model_path": str(spec["path"]),
                        }
                    )
            for batch in spec["batches"]:
                cases.append(
                    {
                        "model": model_name,
                        "backend": backend_name,
                        "phase": "decode",
                        "context": 128,
                        "batch": batch,
                        "gen_tokens": spec["gen_tokens"],
                        "binary": str(backend["binary"]),
                        "ngl": backend["ngl"],
                        "model_path": str(spec["path"]),
                    }
                )
    return cases


def _run_case(case: dict) -> dict:
    binary = Path(case["binary"])
    if not binary.exists():
        raise FileNotFoundError(f"missing llama-bench binary: {binary}")
    if not Path(case["model_path"]).exists():
        raise FileNotFoundError(f"missing model: {case['model_path']}")

    command = [
        str(binary),
        "--offline",
        "-m",
        case["model_path"],
        "-p",
        str(case["context"]),
        "-n",
        str(case["gen_tokens"]),
        "-r",
        "3",
        "--delay",
        "1",
        "-t",
        "8",
        "-b",
        str(case["batch"]),
        "-ub",
        str(case["batch"]),
        "-ngl",
        str(case["ngl"]),
        "-o",
        "json",
    ]

    out_name = (
        f"{case['model']}-{case['backend']}-{case['phase']}-"
        f"p{case['context']}-b{case['batch']}-g{case['gen_tokens']}.json"
    )
    out_path = OUT_DIR / out_name
    err_path = OUT_DIR / out_name.replace(".json", ".stderr")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    out_path.write_text(result.stdout)
    err_path.write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"benchmark failed for {case}: return code {result.returncode}\n{result.stderr}")
    payload = json.loads(result.stdout)
    return {
        "case": case,
        "payload": payload,
        "output": str(out_path),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = build_case_table()
    all_rows = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case['model']} {case['backend']} {case['phase']} context={case['context']} batch={case['batch']}")
        result = _run_case(case)
        all_rows.append(result)

    summary = {
        "experiment": "scale-context-concurrency-sweep",
        "cases": [entry["case"] for entry in all_rows],
        "rows": [entry["payload"] for entry in all_rows],
        "notes": [
            "Each case uses 3 llama-bench repetitions with 1 s inter-run delay.",
            "Prefill measures prompt processing (n_gen=0) with context in {128,512,2048}.",
            "Decode measures generation throughput at n_gen=128 and context fixed at 128.",
            "The matrix is designed to expose the model-scale × concurrency × context crossover.",
        ],
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

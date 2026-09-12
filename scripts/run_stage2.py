"""Run pinned llama.cpp performance and quality experiments on local model files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = root / value
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def recorded_command(command: list[str]) -> list[str]:
    """Remove checkout-specific prefixes while retaining an executable recipe."""
    values = []
    for value in command:
        try:
            values.append(str(Path(value).relative_to(ROOT)))
        except (ValueError, TypeError):
            values.append(value)
    return values


def validate_inputs(config: dict) -> dict:
    checked = {"models": [], "datasets": {}}
    for item in config["models"]:
        path = resolve(ROOT, item["path"])
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual["bytes"] != item["bytes"] or actual["sha256"] != item["sha256"]:
            raise ValueError(f"model provenance mismatch: {item['label']}")
        checked["models"].append({**item, "verified": True})
    for label, item in config["datasets"].items():
        path = resolve(ROOT, item["path"])
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"dataset provenance mismatch: {label}")
        checked["datasets"][label] = {**item, "bytes": path.stat().st_size, "verified": True}
    return checked


def run(command: list[str], stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess:
    started = time.time()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {stderr_path}")
    result.elapsed_seconds = time.time() - started  # type: ignore[attr-defined]
    return result


def benchmark(config: dict, output: Path) -> list[dict]:
    settings = config["benchmark"]
    jobs = [(model, backend) for model in config["models"] for backend in ("cpu", "metal")]
    random.Random(20260910).shuffle(jobs)
    rows = []
    for model, backend in jobs:
        binary_key = "cpu_binary" if backend == "cpu" else "metal_binary"
        binary = resolve(ROOT, config["llama_cpp"][binary_key])
        raw = output / "raw"
        stem = f"bench-{model['label'].lower()}-{backend}"
        command = [
            str(binary), "--offline", "-m", model["path"],
            "-p", ",".join(map(str, settings["prompt_tokens"])),
            "-n", str(settings["generation_tokens"]),
            "-r", str(settings["repetitions"]), "--delay", str(settings["delay_seconds"]),
            "-t", str(settings["threads"]), "-b", str(settings["batch"]),
            "-ub", str(settings["ubatch"]), "-ngl", "0" if backend == "cpu" else "99",
            "-o", "json",
        ]
        result = run(command, raw / f"{stem}.json", raw / f"{stem}.stderr.txt")
        payload = json.loads(result.stdout)
        for row in payload:
            row.update({
                "model_label": model["label"],
                "requested_backend": backend,
                "evidence": "measured_llama_cpp",
                "command": recorded_command(command),
            })
            rows.append(row)
    (output / "benchmark.json").write_text(json.dumps({
        "experiment_id": config["experiment_id"],
        "rows": rows,
        "measurement_scope": "llama-bench default warmup; seven repetitions; one-second inter-test delay; CPU and Metal use separate builds",
        "limitations": "No thermal/frequency lock, external power meter, or independent process-memory sampling",
    }, indent=2) + "\n")
    return rows


def quality(config: dict, output: Path) -> dict:
    settings = config["quality"]
    raw = output / "raw"
    ppl_rows, hellaswag_rows = [], []
    binary = resolve(ROOT, config["llama_cpp"]["metal_quality_binary"])
    for model in config["models"]:
        common = [str(binary), "--offline", "-m", model["path"], "-ngl", "99", "-t", "8", "-b", "512", "-ub", "512"]
        stem = model["label"].lower()
        ppl_command = common + ["-f", config["datasets"]["wikitext2"]["path"], "-c", str(settings["context"]), "--chunks", str(settings["wikitext_chunks"]), "--ppl-output-type", "1"]
        result = run(ppl_command, raw / f"ppl-{stem}.stdout.txt", raw / f"ppl-{stem}.stderr.txt")
        match = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", result.stderr)
        if not match:
            raise ValueError(f"could not parse perplexity for {model['label']}")
        ppl_rows.append({"model_label": model["label"], "ppl": float(match.group(1)), "reported_uncertainty": float(match.group(2)), "chunks": settings["wikitext_chunks"], "context": settings["context"], "evidence": "measured_llama_cpp", "command": recorded_command(ppl_command)})

        hs_command = common + ["-f", config["datasets"]["hellaswag"]["path"], "--hellaswag", "--hellaswag-tasks", str(settings["hellaswag_tasks"]), "--seed", str(settings["seed"])]
        result = run(hs_command, raw / f"hellaswag-{stem}.stdout.txt", raw / f"hellaswag-{stem}.stderr.txt")
        task_lines = [line for line in result.stdout.splitlines() if re.match(r"^\d+\t", line)]
        if not task_lines:
            raise ValueError(f"could not parse HellaSwag for {model['label']}")
        fields = task_lines[-1].split("\t")
        interval = re.search(r"\[([0-9.]+)%, ([0-9.]+)%\]", fields[2])
        hellaswag_rows.append({"model_label": model["label"], "tasks": int(fields[0]), "acc_norm_percent": float(fields[1].rstrip("%")), "ci95_percent": [float(interval.group(1)), float(interval.group(2))] if interval else None, "seed": settings["seed"], "evidence": "measured_llama_cpp", "command": recorded_command(hs_command)})
    payload = {
        "experiment_id": config["experiment_id"],
        "perplexity": ppl_rows,
        "hellaswag": hellaswag_rows,
        "quality_scope": "WikiText-2 raw test subset and a fixed-seed HellaSwag subset; model files are official GGUF variants",
        "limitations": "Pilot sample sizes; HellaSwag interval is llama.cpp's running interval; no instruction-following task suite",
    }
    (output / "quality.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def manifest(config: dict, verified: dict, output: Path) -> None:
    payload = {
        "experiment_id": config["experiment_id"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "cpu": "Apple M4 Pro",
            "cpu_cores": {"performance": 8, "efficiency": 4},
            "gpu": "Apple M4 Pro, 16 cores, Metal 4",
            "memory_bytes": 25769803776,
            "os": platform.platform(),
            "architecture": platform.machine(),
        },
        "host_evidence": "system_profiler and sysctl observed locally; serial numbers and hardware identifiers intentionally excluded",
        "software": config["llama_cpp"],
        "inputs": verified,
        "power": {"measured": False, "reason": "powermetrics requires interactive administrator authorization"},
        "area": {"measured": False, "reason": "no RTL synthesis in this stage"},
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage2.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/m4-stage2")
    parser.add_argument("--mode", choices=("all", "benchmark", "quality", "validate"), default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verified = validate_inputs(config)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw").mkdir(exist_ok=True)
    manifest(config, verified, args.out)
    if args.mode in ("all", "benchmark"):
        rows = benchmark(config, args.out)
        print(f"benchmark rows: {len(rows)}", flush=True)
    if args.mode in ("all", "quality"):
        result = quality(config, args.out)
        print(f"quality rows: {len(result['perplexity']) + len(result['hellaswag'])}", flush=True)
    if args.mode == "validate":
        print("all model and dataset hashes verified", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Minimal agent-oriented evaluation harness for local Qwen2.5-7B GGUF runs.

This keeps the workflow aligned with the repository's evidence discipline:
- it is a separate config from the committed Stage 2 Qwen2.5-1.5B evidence
- it exercises agent-like tasks using text prompts and local repo inspection
- it records benchmark metadata without pretending to be a full product benchmark
- a real generation run still requires the actual GGUF file and llama.cpp binary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def default_repo_probe() -> list[dict]:
    repo = ROOT
    files = [
        "README.md",
        "configs",
        "results",
        "scripts",
        "tests",
    ]
    return [{"path": str(repo / name), "exists": (repo / name).exists()} for name in files]


def evaluate_agent_tasks(cfg: dict) -> list[dict]:
    rows = []
    for task in cfg["agent_tasks"]:
        row = {
            "task": task["name"],
            "tags": task["tags"],
            "max_steps": task["max_steps"],
            "success_score": task["success_score"],
            "prompt": task["prompt"],
            "repo_probe": default_repo_probe(),
            "expected_signal": task["expected_signal"],
            "status": "configured",
            "evidence": "agent-eval-harness",
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "qwen3_agent.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "qwen3-agent")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = evaluate_agent_tasks(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "agent_eval.json"
    out.write_text(json.dumps({"experiment_id": cfg["experiment_id"], "model_family": cfg["model_family"], "tasks": rows}, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()

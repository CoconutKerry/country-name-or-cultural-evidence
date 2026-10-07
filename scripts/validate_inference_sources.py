#!/usr/bin/env python3
"""Run the unit and contract tests preserved with each inference source."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "qwen2.5-7b-instruct",
    "llama-3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "gemma-2-9b-it",
]


def main() -> None:
    env = dict(os.environ)
    env.setdefault("TERM", "xterm")
    for name in SOURCES:
        source = ROOT / "inference/validated_sources" / name
        print(f"==> validating {name}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=source,
            env={**env, "PYTHONPATH": "."},
            check=True,
        )
    print("INFERENCE SOURCE TESTS PASS")


if __name__ == "__main__":
    main()

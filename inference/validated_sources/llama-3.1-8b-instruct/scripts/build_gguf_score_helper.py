#!/usr/bin/env python3
"""Build the isolated llama.cpp low-level label scorer."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "build/gguf-smoke/gguf_score_helper",
    )
    args = parser.parse_args()

    llama_dir = args.llama_cpp_dir.resolve()
    build_bin = llama_dir / "build-cpu/bin"
    required = [
        llama_dir / "include/llama.h",
        llama_dir / "ggml/include/ggml-backend.h",
        llama_dir / "vendor/nlohmann/json.hpp",
        build_bin / "libllama.so",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"llama.cpp CPU build is incomplete: {missing}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "c++",
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        f"-I{llama_dir / 'include'}",
        f"-I{llama_dir / 'ggml/include'}",
        f"-I{llama_dir / 'vendor/nlohmann'}",
        str(PROJECT_ROOT / "src/gguf_score_helper.cpp"),
        f"-L{build_bin}",
        f"-Wl,-rpath,{build_bin}",
        "-lllama",
        "-lggml",
        "-lggml-base",
        "-lggml-cpu",
        "-pthread",
        "-fopenmp",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()

"""
Run all 4 benchmark scripts directly on Modal (H200).
Executes each script in-process via runpy - same process, same CUDA context as Modal worker.
No subprocess. TileLang runs natively.
Collects generated .png, .csv, .html artifacts and returns them.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import modal

APP_NAME = "dslmonkey-ops-four-benchmarks"
REMOTE_REPO_ROOT = "/home/simon/willyc/dslmonkey-accelerated"
REMOTE_PLOTS_DIR = os.path.join(REMOTE_REPO_ROOT, "plots")

_SCRIPT_PATH = Path(__file__).resolve()
LOCAL_REPO_ROOT = _SCRIPT_PATH.parents[2] if len(_SCRIPT_PATH.parents) > 2 else Path.cwd()

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        [
            "torch==2.5.1",
            "triton==3.1.0",
            "numpy",
            "pandas",
            "matplotlib",
            "einops",
            "transformers>=4.45.0",
            "flash-linear-attention",
            "tilelang",
        ]
    )
    .add_local_dir(
        str(LOCAL_REPO_ROOT),
        remote_path=REMOTE_REPO_ROOT,
        ignore=[".git", ".venv", "**/__pycache__", "benchmarks/ops/modal_outputs"],
    )
)


@app.function(
    image=image,
    gpu="H100",  # H200 on Modal triggers XID 43 with TileLang; A100 works (user: TileLang works on local H200)
    timeout=60 * 60,
)
def run_all_four_benchmarks() -> dict[str, bytes]:
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONPATH"] = REMOTE_REPO_ROOT
    os.chdir(REMOTE_REPO_ROOT)
    sys.path.insert(0, REMOTE_REPO_ROOT)

    scripts = [
        "benchmark_based.py",
        "benchmark_rebased_parallel.py",
        "benchmark_linear_attention_parallel.py",
        "benchmark_retention_parallel.py",
    ]

    for script_name in scripts:
        script_path = os.path.join(REMOTE_REPO_ROOT, "benchmarks", "ops", script_name)
        print("=" * 80)
        print(f"Running {script_name}")
        print("=" * 80)
        runpy.run_path(script_path, run_name="__main__")

    # Collect outputs from plots/
    outputs: dict[str, bytes] = {}
    if os.path.isdir(REMOTE_PLOTS_DIR):
        for path in sorted(Path(REMOTE_PLOTS_DIR).iterdir()):
            if path.is_file() and path.suffix.lower() in {".png", ".csv", ".html"}:
                outputs[path.name] = path.read_bytes()
                print(f"Collected: {path.name}")

    # Also check benchmarks/ops/plots (some scripts may write there)
    ops_plots = os.path.join(REMOTE_REPO_ROOT, "benchmarks", "ops", "plots")
    if os.path.isdir(ops_plots):
        for path in sorted(Path(ops_plots).iterdir()):
            if path.is_file() and path.suffix.lower() in {".png", ".csv", ".html"}:
                name = path.name
                if name not in outputs:
                    outputs[name] = path.read_bytes()
                    print(f"Collected: {name}")

    return outputs


@app.local_entrypoint()
def main(output_dir: str = "benchmarks/ops/modal_outputs") -> None:
    output_path = LOCAL_REPO_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving Modal outputs to: {output_path}")
    files = run_all_four_benchmarks.remote()

    for name, data in files.items():
        destination = output_path / name
        destination.write_bytes(data)
        print(f"Wrote {destination}")

    print("\nDone. Generated benchmark artifacts:")
    for p in sorted(output_path.iterdir()):
        print(f"  - {p}")

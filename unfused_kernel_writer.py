#!/usr/bin/env python3
"""
Script to substitute TileLang kernels into an original PyTorch file using Gemini-3-pro-preview.

Usage:
    python unfused_kernel_writer.py <original_pytorch_file.py> <tilelang_kernel1.py> [tilelang_kernel2.py ...]

    Or with a directory of kernels:
    python unfused_kernel_writer.py <original_pytorch_file.py> --kernels-dir <dir_with_tilelang_kernels>

Requires:
    - GEMINI_API_KEY environment variable set
    - google-generativeai package installed (pip install google-generativeai)
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not installed. Run: pip install google-generativeai")
    sys.exit(1)


SYSTEM_PROMPT = """You are an expert at integrating TileLang GPU kernels into PyTorch code.

Your task is to take an original PyTorch program and a set of TileLang kernel implementations,
then produce a new version of the PyTorch program that uses the TileLang kernels wherever possible.

TileLang kernels are defined using the tilelang library with:
- `@T.prim_func` decorated functions that define the GPU kernel
- `tilelang.compile()` to compile the kernel
- Usually wrapped in a PyTorch nn.Module or functional interface

When substituting TileLang kernels:
1. Identify which parts of the original PyTorch code can be replaced by the provided TileLang kernels
2. Keep the overall structure and interface of the original code
3. Import tilelang and tilelang.language as T at the top
4. Include the TileLang kernel build functions in the output
5. Replace the corresponding PyTorch operations with calls to the compiled TileLang kernels
6. Handle kernel caching (compile once, reuse) for efficiency
7. Ensure tensor contiguity before passing to TileLang kernels
8. Preserve any parts that don't have TileLang replacements as original PyTorch

IMPORTANT:
- The output should be a complete, runnable Python file
- Maintain the same function/class signatures as the original
- Add comments indicating which TileLang kernel is being used for each substitution
- If a TileLang kernel only covers PART of an operation, use it for that part and keep PyTorch for the rest
- Handle dtype conversions appropriately (TileLang often uses bfloat16/float16)

HARD REQUIREMENT:
- Everything must run in fp16 end-to-end: inputs are fp16, ModelNew parameters (if any) are fp16, and TileLang kernels operate on fp16."""


def _reference_dimension_contract_text(original_code: str) -> str:
    # Keep this conservative: we cannot reliably infer all shapes statically.
    # The most important contract (for chunked linear attention) is that V/output head_dim
    # may differ from Q/K feature_dim, so the generated code must not assume they match.
    hints = []
    if re.search(r"v\s*=\s*v\.view\([^\)]*,-1\)", original_code):
        hints.append(
            "- The reference reshapes V with `-1` (e.g. `v.view(..., -1)`), so V's last "
            "dimension is allowed to differ from Q/K's last dimension."
        )
    if re.search(r"o\s*=\s*o\.view\([^\)]*,-1\)", original_code):
        hints.append(
            "- The reference reshapes output with `-1` (e.g. `o.view(..., -1)`), so output "
            "last dimension must match V's last dimension (head_dim), not necessarily Q/K (feature_dim)."
        )
    extra = "\n".join(hints).strip()
    return (
        "## Reference Dimension Contract (must match exactly)\n"
        "You MUST preserve the exact tensor shape semantics of the reference program.\n\n"
        "- Let `feature_dim = q.shape[-1]` and `feature_dim = k.shape[-1]`.\n"
        "- Let `head_dim = v.shape[-1]`.\n"
        "- **Do NOT assume `feature_dim == head_dim`.** This is a common linear-attention setup.\n"
        "- Output MUST have shape `(B, H, T, head_dim)` and MUST match the reference numerically.\n"
        "- Any intermediate `view/reshape` of tensors involving V/output MUST use `head_dim` (or `-1`)\n"
        "  and must never hardcode `feature_dim` where `head_dim` belongs.\n"
        "\n"
        + (extra + "\n" if extra else "")
    )


def build_prompt(
    original_code: str,
    original_path: str,
    tilelang_kernels: list[tuple[str, str]],
    attempt_index: int,
    failure_log: str,
) -> str:
    """Build the prompt with original code and all TileLang kernels."""
    
    kernels_section = ""
    for path, code in tilelang_kernels:
        kernels_section += f"\n### TileLang Kernel: {path}\n```python\n{code}\n```\n"
    
    failure_section = ""
    if failure_log.strip():
        failure_section = f"""\n\n## Previous Attempt Failures (Modal / compilation / correctness logs)
{failure_log}
"""

    dim_contract = _reference_dimension_contract_text(original_code)

    return f"""I have an original PyTorch program and several TileLang kernel implementations.
Please create a new version of the original program that substitutes in the TileLang kernels
wherever they can replace PyTorch operations.

This is attempt #{attempt_index}.

## Original PyTorch File: {original_path}
```python
{original_code}
```

## Available TileLang Kernels
{kernels_section}

{dim_contract}

## Instructions
1. Analyze which parts of the original code can be replaced by the TileLang kernels
2. Create a new version that uses TileLang kernels where applicable.
   - You are fully allowed to use PyTorch for glue / reshapes / masking / cumsum / einsum, etc.
   - You are also allowed to **modify/adapt the provided TileLang kernels** (e.g. tensor shapes,
     kernel signatures, tiling constants, output layout) as needed to exactly match the reference.
   - However, you must keep and use the provided kernels (do not ignore them). If a kernel needs to
     be generalized (e.g. support `feature_dim != head_dim`), do so.
3. Keep PyTorch for any operations not covered by the TileLang kernels
4. The output should be a single, complete Python file
5. Add comments to indicate which TileLang kernel is being used for each substitution
6. Maintain the same external interface (function signatures, class API)
7. The output MUST define `class ModelNew(nn.Module)` (KernelBench expects this name for the candidate).
8. Everything MUST run in fp16 end-to-end.
9. **Shape contract is non-negotiable**: match the reference's expected dimensions exactly, especially
   the `feature_dim` (Q/K) vs `head_dim` (V/output) distinction for chunked linear attention.
{failure_section}

Output the complete modified Python file:"""


def call_gemini(
    original_code: str,
    original_path: str,
    tilelang_kernels: list[tuple[str, str]],
    api_key: str,
    attempt_index: int,
    failure_log: str,
) -> str:
    """Call Gemini-3-pro-preview with the substitution prompt."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-pro-preview")

    full_prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + build_prompt(original_code, original_path, tilelang_kernels, attempt_index, failure_log)
    )

    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=32000,
        ),
    )
    
    return response.text


def extract_code(response_text: str) -> str:
    """Extract Python code from the Gemini response."""
    import re
    
    # Look for code blocks
    code_blocks = re.findall(r'```python\s*\n(.*?)```', response_text, re.DOTALL)
    
    if code_blocks:
        # Return the largest code block (likely the main output)
        return max(code_blocks, key=len).strip()
    
    # Fallback: return everything after removing markdown
    return response_text.strip()


RUN_AND_CHECK_PATH = (
    Path(__file__).resolve().parents[1]
    / "dsl-monkeys"
    / "KernelBench"
    / "scripts"
    / "run_and_check.py"
)


def _parse_run_and_check_output(stdout: str, stderr: str) -> tuple[bool, bool]:
    combined = f"{stdout}\n{stderr}"
    compiled = bool(re.search(r"\bcompiled\s*=\s*True\b|\bcompiled=True\b", combined))
    correctness = bool(re.search(r"\bcorrectness\s*=\s*True\b|\bcorrectness=True\b", combined))
    return compiled, correctness


def run_modal_check(
    ref_path: Path,
    candidate_path: Path,
    gpu: str,
    backend: str,
    precision: str,
    num_correct_trials: int,
    num_perf_trials: int,
    timeout: int,
    verbose: bool,
    check_kernel: bool,
) -> tuple[bool, str]:
    if not RUN_AND_CHECK_PATH.exists():
        return False, f"run_and_check.py not found at: {RUN_AND_CHECK_PATH}"

    cmd = [
        sys.executable,
        str(RUN_AND_CHECK_PATH),
        "ref_origin=local",
        f"ref_arch_src_path={ref_path}",
        f"kernel_src_path={candidate_path}",
        "eval_mode=modal",
        f"gpu={gpu}",
        f"backend={backend}",
        f"precision={precision}",
        f"num_correct_trials={num_correct_trials}",
        f"num_perf_trials={num_perf_trials}",
        f"timeout={timeout}",
        f"verbose={str(verbose)}",
        f"measure_performance=False",
        f"check_kernel={str(check_kernel)}",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    compiled, correctness = _parse_run_and_check_output(stdout, stderr)

    success = (proc.returncode == 0) and compiled and correctness
    log = (
        f"[run_and_check returncode={proc.returncode}]\n"
        f"[compiled={compiled} correctness={correctness}]\n"
        f"--- STDOUT ---\n{stdout}\n"
        f"--- STDERR ---\n{stderr}\n"
    )
    return success, log


def load_tilelang_kernels(kernel_paths: list[Path], kernels_dir: Path | None) -> list[tuple[str, str]]:
    """Load TileLang kernel files and return list of (path, code) tuples."""
    kernels = []
    
    # Load from explicit paths
    for path in kernel_paths:
        if path.exists():
            kernels.append((str(path), path.read_text()))
        else:
            print(f"Warning: Kernel file not found: {path}")
    
    # Load from directory if specified
    if kernels_dir and kernels_dir.exists():
        for py_file in sorted(kernels_dir.glob("*.py")):
            content = py_file.read_text()
            # Check if it looks like a TileLang kernel (has tilelang imports)
            if "tilelang" in content or "T.prim_func" in content:
                kernels.append((str(py_file), content))
    
    return kernels


def main():
    parser = argparse.ArgumentParser(
        description="Substitute TileLang kernels into a PyTorch file using Gemini"
    )
    parser.add_argument("original_file", type=Path, help="Original PyTorch .py file")
    parser.add_argument("kernel_files", type=Path, nargs="*", help="TileLang kernel .py files")
    parser.add_argument(
        "--kernels-dir", "-d", type=Path, default=None,
        help="Directory containing TileLang kernel files"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output file path (default: <original>_tilelang.py)"
    )
    parser.add_argument(
        "--gpu", type=str, default="L40S",
        help="Modal GPU type (e.g., L40S, H100, H200, A100)"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=6,
        help="Maximum number of Gemini attempts before giving up"
    )
    parser.add_argument(
        "--backend", type=str, default="tilelang",
        help="KernelBench backend for evaluation (tilelang recommended)"
    )
    parser.add_argument(
        "--num-correct-trials", type=int, default=5,
        help="Number of correctness trials (KernelBench eval)"
    )
    parser.add_argument(
        "--num-perf-trials", type=int, default=1,
        help="Number of perf trials (kept small; not used when measure_performance=False)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout seconds for KernelBench eval"
    )
    parser.add_argument(
        "--no-static-check", action="store_true",
        help="Disable KernelBench static checker"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Print output without writing to file"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Gemini API key (defaults to GEMINI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set and --api-key not provided")
        sys.exit(1)
    
    # Read original file
    if not args.original_file.exists():
        print(f"Error: Original file not found: {args.original_file}")
        sys.exit(1)
    
    original_code = args.original_file.read_text()
    print(f"Read original file: {args.original_file} ({len(original_code)} bytes)")
    
    # Load TileLang kernels
    tilelang_kernels = load_tilelang_kernels(args.kernel_files, args.kernels_dir)
    
    if not tilelang_kernels:
        print("Error: No TileLang kernel files provided or found")
        sys.exit(1)
    
    print(f"Loaded {len(tilelang_kernels)} TileLang kernel(s):")
    for path, _ in tilelang_kernels:
        print(f"  - {path}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.original_file.with_stem(args.original_file.stem + "_tilelang")

    if args.dry_run:
        print("\nCalling Gemini-3-pro-preview (dry-run)...")
        response_text = call_gemini(
            original_code,
            str(args.original_file),
            tilelang_kernels,
            api_key,
            attempt_index=1,
            failure_log="",
        )
        candidate_code = extract_code(response_text)
        print(f"\n{'='*60}")
        print(f"Would write to: {output_path}")
        print(f"{'='*60}")
        print(candidate_code)
        return

    precision = "fp16"
    check_kernel = not args.no_static_check

    failure_log = ""
    last_candidate_code = ""
    for attempt in range(1, args.max_attempts + 1):
        if attempt == 1:
            print("\nCalling Gemini-3-pro-preview...")
        else:
            print(f"\nRetrying Gemini-3-pro-preview (attempt {attempt}/{args.max_attempts})...")

        response_text = call_gemini(
            original_code,
            str(args.original_file),
            tilelang_kernels,
            api_key,
            attempt_index=attempt,
            failure_log=failure_log,
        )
        print(f"Received response ({len(response_text)} chars)")

        candidate_code = extract_code(response_text)
        if not candidate_code.strip():
            failure_log += f"\n[Attempt {attempt}] Empty model output.\n"
            continue

        if "class ModelNew" not in candidate_code:
            failure_log += f"\n[Attempt {attempt}] Output did not define class ModelNew.\n"
            continue

        last_candidate_code = candidate_code

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_attempt{attempt}.py",
            prefix=output_path.stem + "_",
            dir=str(output_path.parent),
            delete=False,
        ) as tmp:
            tmp.write(candidate_code)
            candidate_path = Path(tmp.name)

        print(f"Evaluating candidate on Modal (gpu={args.gpu}, precision={precision}, backend={args.backend})...")
        ok, log = run_modal_check(
            ref_path=args.original_file,
            candidate_path=candidate_path,
            gpu=args.gpu,
            backend=args.backend,
            precision=precision,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            timeout=args.timeout,
            verbose=False,
            check_kernel=check_kernel,
        )

        if ok:
            output_path.write_text(candidate_code)
            print(f"\nCORRECT candidate found on attempt {attempt}. Saved: {output_path}")
            try:
                candidate_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        failure_log += f"\n\n=== Attempt {attempt} Modal Check Failure ===\n{log}\n"
        # Keep the prompt from growing without bound
        if len(failure_log) > 20000:
            failure_log = failure_log[-20000:]
        print(f"Candidate failed on attempt {attempt}; appending logs and retrying.")
        try:
            candidate_path.unlink(missing_ok=True)
        except Exception:
            pass

    print("\nNo correct candidate produced within max attempts.")
    if last_candidate_code.strip():
        output_path.write_text(last_candidate_code)
    else:
        output_path.write_text("# No candidate produced. See stderr/logs from unfused_kernel_writer.py\n")
    print(f"Last candidate saved for inspection: {output_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()

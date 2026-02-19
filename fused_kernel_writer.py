#!/usr/bin/env python3
""" 
Script to fuse TileLang kernels inside a single Python file using Gemini-3-pro-preview.

Given an *unfused* TileLang/PyTorch program (reference), this script asks Gemini to
aggressively fuse as many TileLang kernels as possible into fewer kernel launches,
while preserving the reference program's exact semantics and shape contracts.

It then uses KernelBench's scripts/run_and_check.py to validate the fused candidate
against the input reference on Modal (fp16), retrying with failure logs appended
until success (or max attempts).

Usage:
    python fused_kernel_writer.py <unfused_reference.py>

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


SYSTEM_PROMPT = """You are an expert GPU kernel engineer specializing in TileLang.

You will be given ONE Python program that already contains a working (but unfused)
implementation using PyTorch + many TileLang kernels.

Your job:
- Produce a new Python file that is functionally identical to the reference, but
  aggressively reduces kernel-launch overhead by fusing TileLang work into as few
  kernels as possible.

Hard constraints:
- Preserve the exact external interface expected by KernelBench: the output MUST
  define `class ModelNew(nn.Module)`.
- Preserve all semantic behavior and tensor shape contracts of the reference.
- Everything must run in fp16 end-to-end.

Fusing requirements:
- Aggressively fuse TileLang compute into fewer kernels:
  - Prefer combining multiple stages into a single `@T.prim_func` when possible.
  - Prefer producing multiple outputs from one kernel if it removes launches.
  - Prefer in-kernel accumulation/state to avoid intermediate global memory.
- You are allowed to keep PyTorch for glue (reshapes, masks, cumsum, etc.) ONLY
  when it does not block fusion, but you should strive to move work into fused
  TileLang kernels.

TileLang kernel correctness requirements:
- Ensure correct indexing, causal masking semantics, and exact same dtype casting.
- Ensure correct contiguity requirements and any needed `.contiguous()` calls.
- If the reference supports `feature_dim != head_dim` (common for linear attention),
  the fused kernels MUST also support it.

CRITICAL TileLang compiler constraints (do NOT violate):
- **Do NOT reuse the same `T.alloc_fragment(...)` buffer as both**:
  - a GEMM input operand (A or B), and
  - a GEMM accumulator/output (C)
  inside the same kernel.
  This commonly triggers TileLang LayoutInference failures like:
  `Get different layout for acc_s/state` (conflicting fragment layouts/replication).
  If you need a recurrent state across chunks, keep it in `T.alloc_shared(...)` or global
  memory and use separate fragment accumulators, OR use separate fragment buffers for
  operand-vs-accumulator roles.
- GEMM operand dtypes must match: TileLang GEMM requires `A.dtype == B.dtype`.
  Do not call `T.gemm(fp16_A, fp32_B, ...)`. If you want fp32 accumulation, accumulate
  into a fp32 accumulator fragment/output but keep GEMM input operands same dtype.

Output requirements:
- Output a complete, runnable Python file.
- Prefer outputting only a single ```python code block.
"""


def _reference_dimension_contract_text(reference_code: str) -> str:
    hints = []
    if re.search(r"v\s*=\s*v\.view\([^\)]*,-1\)", reference_code):
        hints.append(
            "- The reference reshapes V with `-1` (e.g. `v.view(..., -1)`), so V's last "
            "dimension is allowed to differ from Q/K's last dimension."
        )
    if re.search(r"o\s*=\s*o\.view\([^\)]*,-1\)", reference_code):
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
        "- **Do NOT assume `feature_dim == head_dim`.**\n"
        "- Output MUST have shape `(B, H, T, head_dim)` and MUST match the reference numerically.\n"
        "- Any intermediate `view/reshape` of tensors involving V/output MUST use `head_dim` (or `-1`).\n"
        "\n"
        + (extra + "\n" if extra else "")
    )


def build_prompt(
    reference_code: str,
    reference_path: str,
    attempt_index: int,
    failure_log: str,
) -> str:
    dim_contract = _reference_dimension_contract_text(reference_code)

    failure_section = ""
    if failure_log.strip():
        failure_section = f"""\n\n## Previous Attempt Failures (Modal / compilation / correctness logs)
{failure_log}
"""

    return f"""You are given a SINGLE Python program that is correct and acts as the reference.
Your goal is to output a new Python program that is functionally identical but aggressively fuses
TileLang kernels into fewer kernel launches.

This is attempt #{attempt_index}.

## Reference Program Path
{reference_path}

## Reference Program
```python
{reference_code}
```

{dim_contract}

## Requirements
- Preserve exact semantics and outputs vs the reference.
- Everything fp16 end-to-end.
- Aggressively fuse TileLang kernels:
  - Reduce the number of `tilelang.compile(...)` calls.
  - Reduce the number of kernel invocations in `forward`.
  - Prefer fewer, larger kernels that do more work per launch.
- It is allowed to keep some PyTorch glue, but do not regress fusion unnecessarily.
- The output MUST define `class ModelNew(nn.Module)`.
- The output should remain a single file and be runnable.

## Output format
- Output ONLY the complete fused Python file (prefer a single ```python code block).
{failure_section}
"""


def call_gemini(
    reference_code: str,
    reference_path: str,
    api_key: str,
    model_name: str,
    attempt_index: int,
    failure_log: str,
) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    full_prompt = SYSTEM_PROMPT + "\n\n" + build_prompt(
        reference_code=reference_code,
        reference_path=reference_path,
        attempt_index=attempt_index,
        failure_log=failure_log,
    )

    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=32000,
        ),
    )

    return response.text


def extract_code(response_text: str) -> str:
    code_blocks = re.findall(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if code_blocks:
        return max(code_blocks, key=len).strip()
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
        "measure_performance=False",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse TileLang kernels inside a single Python file using Gemini, validate with KernelBench run_and_check on Modal."
    )
    parser.add_argument("reference_file", type=Path, help="Input reference .py file (unfused implementation)")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output fused file path (default: <reference>_fused.py)"
    )
    parser.add_argument("--gpu", type=str, default="H100", help="Modal GPU type (e.g., L40S, H100, H200, A100)")
    parser.add_argument("--max-attempts", type=int, default=1, help="Maximum number of Gemini attempts")
    parser.add_argument("--backend", type=str, default="tilelang", help="KernelBench backend (tilelang recommended)")
    parser.add_argument("--num-correct-trials", type=int, default=5, help="Number of correctness trials")
    parser.add_argument("--num-perf-trials", type=int, default=1, help="Number of perf trials (unused when measure_performance=False)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout seconds for KernelBench eval")
    parser.add_argument("--no-static-check", action="store_true", help="Disable KernelBench static checker")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Generate once and print (no Modal eval, no write)")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key (defaults to GEMINI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview", help="Gemini model name")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set and --api-key not provided")
        sys.exit(1)

    if not args.reference_file.exists():
        print(f"Error: reference file not found: {args.reference_file}")
        sys.exit(1)

    reference_code = args.reference_file.read_text()
    print(f"Read reference file: {args.reference_file} ({len(reference_code)} bytes)")

    if args.output:
        output_path = args.output
    else:
        output_path = args.reference_file.with_stem(args.reference_file.stem + "_fused")

    precision = "fp16"
    check_kernel = not args.no_static_check

    failure_log = ""
    last_candidate_code = ""

    for attempt in range(1, args.max_attempts + 1):
        if attempt == 1:
            print("\nCalling Gemini-3-pro-preview for fused candidate...")
        else:
            print(f"\nRetrying Gemini (attempt {attempt}/{args.max_attempts})...")

        response_text = call_gemini(
            reference_code=reference_code,
            reference_path=str(args.reference_file),
            api_key=api_key,
            model_name=args.model,
            attempt_index=attempt,
            failure_log=failure_log,
        )

        candidate_code = extract_code(response_text)
        if not candidate_code.strip():
            failure_log += f"\n[Attempt {attempt}] Empty model output.\n"
            continue

        if "class ModelNew" not in candidate_code:
            failure_log += f"\n[Attempt {attempt}] Output did not define class ModelNew.\n"
            continue

        last_candidate_code = candidate_code

        if args.dry_run:
            print(f"\n{'='*60}\nDRY RUN OUTPUT (attempt {attempt})\n{'='*60}")
            print(candidate_code)
            return

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_attempt{attempt}.py",
            prefix=output_path.stem + "_",
            dir=str(output_path.parent),
            delete=False,
        ) as tmp:
            tmp.write(candidate_code)
            candidate_path = Path(tmp.name)

        print(f"Evaluating fused candidate on Modal (gpu={args.gpu}, precision={precision}, backend={args.backend})...")
        ok, log = run_modal_check(
            ref_path=args.reference_file,
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
            print(f"\nCORRECT fused candidate found on attempt {attempt}. Saved: {output_path}")
            try:
                candidate_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        failure_log += f"\n\n=== Attempt {attempt} Modal Check Failure ===\n{log}\n"
        if len(failure_log) > 20000:
            failure_log = failure_log[-20000:]
        print(f"Candidate failed on attempt {attempt}; appending logs and retrying.")

        try:
            candidate_path.unlink(missing_ok=True)
        except Exception:
            pass

    print("\nNo correct fused candidate produced within max attempts.")
    if last_candidate_code.strip():
        output_path.write_text(last_candidate_code)
    else:
        output_path.write_text("# No candidate produced. See stderr/logs from fused_kernel_writer.py\n")
    print(f"Last candidate saved for inspection: {output_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()

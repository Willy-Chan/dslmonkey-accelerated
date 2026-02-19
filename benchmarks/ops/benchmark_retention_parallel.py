"""
Benchmark comparing Retention Parallel implementations.

Compares:
- naive_retention_parallel: reference PyTorch implementation
- naive_retention_parallel (compiled): torch.compile wrapper
- triton_retention_parallel: Triton implementation from fla
- tilelang_retention_parallel: TileLang implementation

Includes correctness checks and performance benchmarks across different sequence lengths.
"""

import importlib.util
import os

import torch
import triton

# Import naive Retention implementation from RETENTION_REFERENCES
spec_naive = importlib.util.spec_from_file_location(
    "naive_retention",
    "/home/simon/willyc/dslmonkey-accelerated/RETENTION_REFERENCES/11_retention_parallel.py",
)
module_naive = importlib.util.module_from_spec(spec_naive)
spec_naive.loader.exec_module(module_naive)
NaiveRetentionParallel = module_naive.Model

# Import Triton Retention implementation from fla
from fla.ops.retention import parallel_retention


class TritonRetentionParallel:
    """Wrapper to match benchmark interface (B, H, T, D) -> (B, H, T, D)."""

    def __init__(self):
        pass

    def __call__(self, q, k, v, scale=None):
        # fla.ops.retention.parallel expects (B, T, H, D)
        q_bthd = q.transpose(1, 2).contiguous()
        k_bthd = k.transpose(1, 2).contiguous()
        v_bthd = v.transpose(1, 2).contiguous()
        o_bthd, _ = parallel_retention(q_bthd, k_bthd, v_bthd, scale=scale)
        return o_bthd.transpose(1, 2).contiguous()


try:
    spec_tilelang = importlib.util.spec_from_file_location(
        "tilelang_retention",
        "/home/simon/willyc/dslmonkey-accelerated/RETENTION_SOLUTIONS/lv22-s3-k3.py",
    )
    module_tilelang = importlib.util.module_from_spec(spec_tilelang)
    spec_tilelang.loader.exec_module(module_tilelang)
    TileLangRetentionParallelImpl = module_tilelang.ModelNew
except Exception:
    TileLangRetentionParallelImpl = None


class TileLangRetentionParallel:
    """Wrapper to match benchmark interface (B, H, T, D) -> (B, H, T, D)."""

    def __init__(self):
        if TileLangRetentionParallelImpl is None:
            raise RuntimeError("TileLang retention implementation not available")
        self._model = TileLangRetentionParallelImpl()

    def __call__(self, q, k, v):
        return self._model(q, k, v)


naive_retention_compiled = torch.compile(NaiveRetentionParallel(), mode="max-autotune")

_correctness_checked = set()
_compiled_warmup = set()

_tilelang_model = None


def _get_tilelang_model():
    global _tilelang_model
    if _tilelang_model is None:
        _tilelang_model = TileLangRetentionParallel()
    return _tilelang_model


def check_correctness_retention(T=512, atol=1e-2, rtol=1e-2):
    """Compare TileLang implementation against naive PyTorch reference."""

    if T in _correctness_checked:
        return True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    B, H, D = 4, 8, 64

    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)

    naive_model = NaiveRetentionParallel().to(device)
    with torch.no_grad():
        out_naive = naive_model(q, k, v)

    print(f"\nCorrectness Check for T={T}:")
    print(f"  Naive output shape: {out_naive.shape}")
    print(f"  Naive output range: [{out_naive.min():.4f}, {out_naive.max():.4f}]")

    all_correct = True

    try:
        tilelang_model = _get_tilelang_model()
        with torch.no_grad():
            out_tilelang = tilelang_model(q, k, v)

        diff = (out_naive.float() - out_tilelang.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        is_close = torch.allclose(out_naive.float(), out_tilelang.float(), atol=atol, rtol=rtol)

        print(
            f"  TileLang vs Naive: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={is_close}"
        )
        all_correct &= is_close
    except Exception as e:
        print(f"  TileLang check failed: {e}")
        all_correct = False

    _correctness_checked.add(T)
    return all_correct


def _plot_with_markers(df, save_path: str, plot_name: str):
    import matplotlib.pyplot as plt

    x = df["T"].astype(float)
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    ax.set_xscale("log", base=2)
    for col in df.columns:
        if col == "T":
            continue
        color = None
        if "triton" in col.lower():
            color = "cyan"
        elif "tilelang" in col.lower():
            color = "green"
        ax.plot(x, df[col].astype(float), marker="o", linewidth=2.5, label=col, color=color)
    ax.set_title("Parallel RetNet Linear Attention", fontsize=16)
    ax.set_xlabel("Sequence Length (T)", fontsize=14)
    ax.set_ylabel("Execution Time (ms)", fontsize=14)
    ax.legend(fontsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.savefig(os.path.join(save_path, f"{plot_name}.png"))
    plt.close()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T"],
        # x_vals=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        x_vals=[256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=[
            "naive_torch",
            "naive_torch_compiled",
            "triton_retention",
            "tilelang_retention",
        ],
        line_names=["Naive PyTorch", "Naive PyTorch (compiled)",  'Hand-written Triton', 'DSLMonkey-generated TileLang'],
        x_log=True,
        styles=[("red", "-"), ("orange", "-"), ("blue", "-"), ("green", "-")],
        xlabel="Sequence Length (T)",
        ylabel="Execution Time (ms)",
        plot_name="Retention Parallel Latency Comparison",
        args={},
    ),
)
def benchmark_retention_parallel(T, provider):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    B, H, D = 4, 8, 64

    quantiles = [0.5, 0.2, 0.8]
    results = (0, 0, 0)

    if T <= 4096:
        check_correctness_retention(T)

    if provider == "naive_torch":
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)

        model = NaiveRetentionParallel().to(device)
        return triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)

    if provider == "naive_torch_compiled":
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)

        if T not in _compiled_warmup and device == "cuda":
            _compiled_warmup.add(T)
            for _ in range(3):
                naive_retention_compiled(q, k, v)
            torch.cuda.synchronize()

        return triton.testing.do_bench(
            lambda: naive_retention_compiled(q, k, v), quantiles=quantiles
        )

    if provider == "triton_retention":
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)

        model = TritonRetentionParallel()
        return triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)

    if provider == "tilelang_retention":
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)

        model = _get_tilelang_model()
        return triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)

    return results


def run_manual_correctness_test():
    print("=" * 60)
    print("MANUAL CORRECTNESS TESTS")
    print("=" * 60)

    test_lengths = [256, 512, 1024, 2048]
    all_passed = True

    for T in test_lengths:
        passed = check_correctness_retention(T, atol=1e-2, rtol=1e-2)
        all_passed &= passed
        print(f"T={T}: {'PASS' if passed else 'FAIL'}")

    print(f"\nOverall correctness: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    print("Retention Parallel Benchmark")
    print("=" * 40)

    correctness_passed = run_manual_correctness_test()
    if not correctness_passed:
        print("\nWARNING: Some correctness tests failed!")
        print("Proceeding with benchmark anyway...")

    save_path = "./plots/"
    os.makedirs(save_path, exist_ok=True)

    print(f"\nRunning full benchmark suite...")
    print("Results will be saved to:", save_path)

    df = benchmark_retention_parallel.run(save_path=save_path, print_data=True, return_df=True)
    _plot_with_markers(df, save_path=save_path, plot_name="Retention Parallel Latency Comparison")

    print(f"\nBenchmark complete! Check {save_path} for plots.")

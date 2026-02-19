"""Benchmark comparing Linear Attention Parallel implementations.

This benchmarks the repo's "linear attention" reference in LINEAR_ATTN_REFERENCES,
which is chunked attention with:
- inter-chunk state accumulation via per-chunk K^T V
- intra-chunk causal attention within each chunk

Compares:
- naive_torch: reference PyTorch implementation
- naive_torch_compiled: torch.compile of the reference
- triton_linear_attn: (if available) Triton implementation found in repo
- tilelang_linear_attn: TileLang implementation

All benchmarks use float16.
"""

import importlib.util
import os

import torch
import triton

# Import naive reference implementation
spec_naive = importlib.util.spec_from_file_location(
    "naive_linear_attn",
    "/home/simon/willyc/dslmonkey-accelerated/LINEAR_ATTN_REFERENCES/8_linear_attention_parallel.py",
)
module_naive = importlib.util.module_from_spec(spec_naive)
spec_naive.loader.exec_module(module_naive)
NaiveLinearAttentionParallel = module_naive.Model

# Import TileLang implementation
spec_tilelang = importlib.util.spec_from_file_location(
    "tilelang_linear_attn",
    "/home/simon/willyc/dslmonkey-accelerated/LINEAR_ATTN_SOLUTIONS/lv22_s3_k3.py",
)
module_tilelang = importlib.util.module_from_spec(spec_tilelang)
spec_tilelang.loader.exec_module(module_tilelang)
TileLangLinearAttentionParallel = module_tilelang.ModelNew

# Try to find a Triton implementation in the repo.
# Note: fla/ops/linear_attn is "linear attention" but does not implement the exact
# chunked formulation in LINEAR_ATTN_REFERENCES/8_linear_attention_parallel.py.
# We leave this as optional and disable by default.
try:
    from fla.ops.linear_attn import fused_chunk_linear_attn as _triton_chunk_linear_attn

    class TritonLinearAttentionParallel:
        def __init__(self, chunk_size: int = 64):
            self.chunk_size = chunk_size

        def __call__(self, q, k, v, scale=None, normalize: bool = False):
            # fla.ops.linear_attn expects [B, T, H, D]
            o, _ = _triton_chunk_linear_attn(q, k, v, scale=scale, normalize=normalize)
            return o

except Exception:
    TritonLinearAttentionParallel = None


_chunk_size = 64

naive_model = NaiveLinearAttentionParallel(chunk_size=_chunk_size)
naive_compiled = torch.compile(naive_model.forward, mode="max-autotune")

_tilelang_model = TileLangLinearAttentionParallel(chunk_size=_chunk_size)

_correctness_checked = set()
_compiled_warmup = set()


def check_correctness_linear_attn(T=512, atol=2e-1, rtol=1e-2):
    if T in _correctness_checked:
        return True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    B, H, D = 4, 8, 64

    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)

    naive_model_local = naive_model.to(device)
    tilelang_model_local = _tilelang_model.to(device)

    

    with torch.no_grad():
        out_naive = naive_model_local(q, k, v, scale=None, normalize=False)
        out_tilelang = tilelang_model_local(q, k, v, scale=None, normalize=False)

    diff = (out_naive.float() - out_tilelang.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    is_close = torch.allclose(out_naive.float(), out_tilelang.float(), atol=atol, rtol=rtol)

    print(f"\nCorrectness Check for T={T}:")
    print(f"  Naive output shape: {out_naive.shape}")
    print(f"  Naive output range: [{out_naive.min():.4f}, {out_naive.max():.4f}]")
    print(f"  TileLang vs Naive: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={is_close}")

    _correctness_checked.add(T)
    return is_close


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
    ax.set_title("Chunked Linear Attention", fontsize=16)
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
        x_vals=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        # x_vals=[256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["naive_torch", "naive_torch_compiled", "triton_linear_attn", "tilelang_linear_attn"],
        line_names=["Naive PyTorch", "Naive PyTorch (compiled)",  'Hand-written Triton', 'DSLMonkey-generated TileLang'],
        x_log=True,
        styles=[("red", "-"), ("orange", "-"), ("blue", "-"), ("green", "-")],
        xlabel="Sequence Length (T)",
        ylabel="Execution Time (ms)",
        plot_name="Chunked Linear Attention Latency Comparison",
        args={},
    ),
)
def benchmark_linear_attention_parallel(T, provider):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    B, H, D = 4, 8, 64

    quantiles = [0.5, 0.2, 0.8]
    results = (0, 0, 0)

    if T <= 4096:
        check_correctness_linear_attn(T)

    q = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)

    if provider == "naive_torch":
        model = naive_model.to(device)
        return triton.testing.do_bench(lambda: model(q, k, v, scale=None, normalize=False), quantiles=quantiles)

    if provider == "naive_torch_compiled":
        model = naive_compiled
        if T not in _compiled_warmup and device == "cuda":
            _compiled_warmup.add(T)
            for _ in range(3):
                model(q, k, v, scale=None, normalize=False)
            torch.cuda.synchronize()
        return triton.testing.do_bench(lambda: model(q, k, v, scale=None, normalize=False), quantiles=quantiles)

    if provider == "triton_linear_attn":
        if TritonLinearAttentionParallel is None:
            print(f"{provider}: Implementation not available")
            return results
        model = TritonLinearAttentionParallel(chunk_size=_chunk_size)
        return triton.testing.do_bench(lambda: model(q, k, v, scale=None, normalize=False), quantiles=quantiles)

    if provider == "tilelang_linear_attn":
        model = _tilelang_model.to(device)
        return triton.testing.do_bench(lambda: model(q, k, v, scale=None, normalize=False), quantiles=quantiles)

    return results


def run_manual_correctness_test():
    print("=" * 60)
    print("MANUAL CORRECTNESS TESTS")
    print("=" * 60)

    test_lengths = [256, 512, 1024, 2048]
    all_passed = True

    for T in test_lengths:
        passed = check_correctness_linear_attn(T, atol=2e-1, rtol=1e-2)
        all_passed &= passed
        print(f"T={T}: {'PASS' if passed else 'FAIL'}")

    print(f"\nOverall correctness: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    print("Linear Attention Parallel Benchmark")
    print("=" * 40)

    correctness_passed = run_manual_correctness_test()
    if not correctness_passed:
        print("\nWARNING: Some correctness tests failed!")
        print("Proceeding with benchmark anyway...")

    save_path = "./plots/"
    os.makedirs(save_path, exist_ok=True)

    print(f"\nRunning full benchmark suite...")
    print("Results will be saved to:", save_path)

    df = benchmark_linear_attention_parallel.run(save_path=save_path, print_data=True, return_df=True)
    _plot_with_markers(df, save_path=save_path, plot_name="Linear Attention Parallel Latency Comparison")

    print(f"\nBenchmark complete! Check {save_path} for plots.")

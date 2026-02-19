"""
Benchmark comparing ReBASED Parallel implementations.

ReBASED (Refined BASED) uses squared attention scores instead of Taylor expansion,
making it simpler but still effective for linear attention.

Compares:
- naive_rebased_parallel: pure PyTorch implementation
- triton_rebased_parallel: hand-optimized Triton (placeholder)
- tilelang_rebased_parallel: TileLang implementation (placeholder)

Includes correctness checks and performance benchmarks across different sequence lengths.
"""
import os

import torch
import triton
import matplotlib.pyplot as plt
import sys
import importlib.util

# Import naive ReBASED implementation from BASED_REFERENCES
spec_naive = importlib.util.spec_from_file_location(
    "naive_rebased", 
    "/home/simon/willyc/dslmonkey-accelerated/REBASED_REFERENCES/9_rebased_parallel.py"
)
module_naive = importlib.util.module_from_spec(spec_naive)
spec_naive.loader.exec_module(module_naive)
NaiveRebasedParallel = module_naive.Model

# Import Triton ReBASED implementation from fla
from fla.ops.rebased import parallel_rebased

class TritonRebasedParallel:
    """Wrapper class for Triton ReBASED implementation to match benchmark interface."""
    def __init__(self):
        pass
    
    def __call__(self, q, k, v, scale=None, use_norm=True):
        # fla.ops.rebased.parallel expects head_first=False by default
        # Our benchmark uses (B, H, T, D) format which is head_first=True
        return parallel_rebased(q, k, v, eps=1e-6, use_scale=True, use_normalize=use_norm, head_first=True)

# TODO: Replace with actual TileLang implementation path
try:
    spec_tilelang = importlib.util.spec_from_file_location(
        "tilelang_rebased", 
        "/home/simon/willyc/dslmonkey-accelerated/REBASED_SOLUTIONS/lv5-s3-k5.py"
    )
    module_tilelang = importlib.util.module_from_spec(spec_tilelang)
    spec_tilelang.loader.exec_module(module_tilelang)
    TileLangRebasedParallel = module_tilelang.ModelNew
    # TileLangRebasedParallel = None
except:
    TileLangRebasedParallel = None

# Compiled versions for better performance
naive_rebased_compiled = torch.compile(NaiveRebasedParallel().forward, mode="max-autotune")

_correctness_checked = set()
_compiled_warmup = set()

def get_flops(batch, seqlen, headdim, nheads):
    """
    Calculate FLOPs for ReBASED attention.
    
    ReBASED operations:
    1. Q @ K^T: 2 * batch * nheads * seqlen * seqlen * headdim
    2. Squared attention: batch * nheads * seqlen * seqlen
    3. Causal masking: negligible
    4. A @ V: 2 * batch * nheads * seqlen * seqlen * headdim
    5. Normalization (if used): batch * nheads * seqlen * (seqlen + headdim)
    """
    # Main attention computation
    qk_flops = 2 * batch * nheads * seqlen * seqlen * headdim
    square_flops = batch * nheads * seqlen * seqlen
    av_flops = 2 * batch * nheads * seqlen * seqlen * headdim
    norm_flops = batch * nheads * seqlen * (seqlen + headdim)
    
    return qk_flops + square_flops + av_flops + norm_flops

def check_correctness_rebased(T=512, atol=1e-2, rtol=1e-2):
    """
    Compare optimized implementations against naive PyTorch reference.
    Uses float16 for correctness checking to match typical usage.
    """
    if T in _correctness_checked:
        return True
        
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, D = 4, 8, 64
    
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    
    # Reference output from naive implementation
    naive_model = NaiveRebasedParallel()
    naive_model = naive_model.to(device)
    with torch.no_grad():
        out_naive = naive_model(q, k, v)
    
    print(f"\nCorrectness Check for T={T}:")
    print(f"  Naive output shape: {out_naive.shape}")
    print(f"  Naive output range: [{out_naive.min():.4f}, {out_naive.max():.4f}]")
    
    all_correct = True
    
    # Check Triton implementation if available
    if TritonRebasedParallel is not None:
        try:
            triton_model = TritonRebasedParallel()
            with torch.no_grad():
                out_triton = triton_model(q, k, v)
            
            max_diff = (out_naive - out_triton).abs().max().item()
            mean_diff = (out_naive - out_triton).abs().mean().item()
            is_close = torch.allclose(out_naive, out_triton, atol=atol, rtol=rtol)
            
            print(f"  Triton vs Naive: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={is_close}")
            all_correct &= is_close
        except Exception as e:
            print(f"  Triton check failed: {e}")
            all_correct = False
    else:
        print("  Triton implementation not available")
    
    # Check TileLang implementation if available
    if TileLangRebasedParallel is not None:
        try:
            tilelang_model = TileLangRebasedParallel()
            with torch.no_grad():
                out_tilelang = tilelang_model(q, k, v)
            
            max_diff = (out_naive - out_tilelang).abs().max().item()
            mean_diff = (out_naive - out_tilelang).abs().mean().item()
            is_close = torch.allclose(out_naive, out_tilelang, atol=atol, rtol=rtol)
            
            print(f"  TileLang vs Naive: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={is_close}")
            all_correct &= is_close
        except Exception as e:
            print(f"  TileLang check failed: {e}")
            all_correct = False
    else:
        print("  TileLang implementation not available")
    
    # print(f"out_naive = {out_naive[:1]}")
    # print(f"out_tilelang = {out_tilelang[:1]}")

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
    ax.set_title("Parallel ReBASED Linear Attention", fontsize=16)
    ax.set_xlabel("Sequence Length (T)", fontsize=14)
    ax.set_ylabel("Execution Time (ms)", fontsize=14)
    ax.legend(fontsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.savefig(os.path.join(save_path, f"{plot_name}.png"))
    plt.close()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[256, 512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['naive_torch', 'naive_torch_compiled', 'triton_rebased', 'tilelang_rebased'],
        line_names=['Naive PyTorch', 'Naive PyTorch (compiled)', 'Hand-written Triton', 'DSLMonkey-generated TileLang'],
        x_log=True,
        styles=[('red', '-'), ('orange', '-'), ('blue', '-'), ('green', '-')],
        xlabel='Sequence Length (T)',
        ylabel="Execution Time (ms)",
        plot_name="ReBASED Latency Comparison",
        args={},
    ),
)
def benchmark_rebased_parallel(T, provider):
    """Benchmark ReBASED parallel implementations across different sequence lengths."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    B, H, D = 4, 8, 64
    
    quantiles = [0.5, 0.2, 0.8]
    results = (0, 0, 0)
    
    # Run correctness check for this sequence length
    if T <= 4096:  # Skip correctness for very long sequences to save time
        check_correctness_rebased(T)
    
    if provider == 'naive_torch':
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        
        model = NaiveRebasedParallel().to(device)
        results = triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)
        flops = get_flops(B, T, D, H)
        tflops_s = flops / (results[0] * 1e-3) / 1e12
        print(f"{provider} TFLOP/s, T={T}: {tflops_s:.3f}")
        return results
    
    elif provider == 'naive_torch_compiled':
        # if T > 8192:  # Skip compiled version for very long sequences
        #     return results
            
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        
        # Warmup compiled version
        if T not in _compiled_warmup:
            _compiled_warmup.add(T)
            for _ in range(3):
                naive_rebased_compiled(q, k, v)
            torch.cuda.synchronize()

        results = triton.testing.do_bench(lambda: naive_rebased_compiled(q, k, v), quantiles=quantiles)
        flops = get_flops(B, T, D, H)
        tflops_s = flops / (results[0] * 1e-3) / 1e12
        print(f"{provider} TFLOP/s, T={T}: {tflops_s:.3f}")
        return results
    
    elif provider == 'triton_rebased':
        if TritonRebasedParallel is None:
            print(f"{provider}: Implementation not available")
            return results
            
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        
        model = TritonRebasedParallel()
        results = triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)
        flops = get_flops(B, T, D, H)
        tflops_s = flops / (results[0] * 1e-3) / 1e12
        print(f"{provider} TFLOP/s, T={T}: {tflops_s:.3f}")
        return results
    
    elif provider == 'tilelang_rebased':
        if TileLangRebasedParallel is None:
            print(f"{provider}: Implementation not available")
            return results
            
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        
        model = TileLangRebasedParallel()
        results = triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)
        flops = get_flops(B, T, D, H)
        tflops_s = flops / (results[0] * 1e-3) / 1e12
        print(f"{provider} TFLOP/s, T={T}: {tflops_s:.3f}")
        return results
    
    return results

def run_manual_correctness_test():
    """Run manual correctness tests for different sequence lengths."""
    print("="*60)
    print("MANUAL CORRECTNESS TESTS")
    print("="*60)
    
    test_lengths = [256, 512, 1024, 2048]
    all_passed = True
    
    for T in test_lengths:
        passed = check_correctness_rebased(T, atol=1e-2, rtol=1e-2)
        all_passed &= passed
        print(f"T={T}: {'PASS' if passed else 'FAIL'}")
    
    print(f"\nOverall correctness: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

def run_performance_comparison():
    """Run a quick performance comparison."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    B, H, D = 4, 8, 64
    T = 2048
    
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    
    # Warmup
    naive_model = NaiveRebasedParallel().to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = naive_model(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            _ = naive_model(q, k, v)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    avg_time = sum(times) / len(times)
    flops = get_flops(B, T, D, H)
    tflops = flops / (avg_time * 1e-3) / 1e12
    
    print(f"Naive PyTorch (T={T}):")
    print(f"  Average time: {avg_time:.3f} ms")
    print(f"  TFLOP/s: {tflops:.3f}")
    print(f"  Total FLOPs: {flops / 1e12:.3f} T")

if __name__ == '__main__':
    import os
    
    print("ReBASED Parallel Benchmark")
    print("=" * 40)
    
    # Run correctness tests
    correctness_passed = run_manual_correctness_test()
    
    if not correctness_passed:
        print("\nWARNING: Some correctness tests failed!")
        print("Proceeding with benchmark anyway...")
    
    # Run quick performance test
    run_performance_comparison()
    
    # Run full benchmark and save plots
    save_path = '/home/simon/willyc/dslmonkey-accelerated/plots'
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\nRunning full benchmark suite...")
    print("Results will be saved to:", save_path)
    
    # This will generate the benchmark plot
    df = benchmark_rebased_parallel.run(save_path=save_path, print_data=True, return_df=True)
    _plot_with_markers(df, save_path=save_path, plot_name="ReBASED Parallel Latency Comparison")
    
    print(f"\nBenchmark complete! Check {save_path} for plots.")

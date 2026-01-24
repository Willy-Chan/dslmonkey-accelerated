"""
Minimal benchmark: TileLang vs Torch (naive) at small sequence lengths
"""
import torch
import triton

from fla.ops.based.naive import naive_parallel_based
from benchmarks.ops.tilelang_based import tilelang_based

# LOCAL H200 BENCHMARK ON RICHARD


def check_correctness(T=512, atol=1e-2, rtol=1e-2):
    """
    Compare TileLang output against naive PyTorch reference.
    Note: They use different feature dims (torch uses 16, tilelang uses D=128),
    so we need to run them with matching dimensions for a fair comparison.
    """
    from fla.utils import device
    dtype = torch.bfloat16
    B, H, D = 8, 16, 128
    
    # Use same dimensions for both
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    
    # Get outputs
    out_torch = naive_parallel_based(q, k, v)
    out_tilelang = tilelang_based(q, k, v)
    
    # Compare
    max_diff = (out_torch - out_tilelang).abs().max().item()
    mean_diff = (out_torch - out_tilelang).abs().mean().item()
    
    # Check if close
    is_close = torch.allclose(out_torch, out_tilelang, atol=atol, rtol=rtol)
    
    print(f"Correctness check (T={T}):")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Shapes match: {out_torch.shape == out_tilelang.shape}")
    print(f"  torch.allclose(atol={atol}, rtol={rtol}): {is_close}")
    
    if not is_close:
        print("  WARNING: Outputs differ significantly!")
    
    return is_close


# Track which T values have been correctness-checked
_correctness_checked = set()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[64 * 2 ** i for i in range(0, 8)],
        line_arg='provider',
        line_vals=['torch', 'tilelang'],
        line_names=['torch_fwd', 'tilelang_fwd'],
        styles=[('blue', '-'), ('magenta', '-')],
        ylabel="Execution Time (ms)",
        plot_name="TileLang_vs_Torch",
        args={},
    ),
)
def benchmark(T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    B, H, D = 8, 16, 128
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        q = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        # Warmup runs
        for _ in range(3):
            naive_parallel_based(q, k, v)
        torch.cuda.synchronize()
        result = triton.testing.do_bench(lambda: naive_parallel_based(q, k, v), quantiles=quantiles)
        
        # Run correctness check after benchmarking this T value (only once per T)
        if T not in _correctness_checked:
            _correctness_checked.add(T)
            check_correctness(T=T)
        
        return result
    elif provider == 'tilelang':
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        # Warmup runs
        for _ in range(3):
            tilelang_based(q, k, v)
        torch.cuda.synchronize()
        return triton.testing.do_bench(lambda: tilelang_based(q, k, v), quantiles=quantiles)


if __name__ == '__main__':
    # Correctness checks run automatically after each T value is benchmarked
    benchmark.run(print_data=True, show_plots=True, save_path='./plots/')

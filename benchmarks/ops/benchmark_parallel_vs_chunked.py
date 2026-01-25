"""
Benchmark comparing PARALLEL vs CHUNKED implementations of BASED attention.

PARALLEL: Good for low sequence lengths, materializes full attention matrix.
  - parallel_based: hand-optimized Triton
  - naive_parallel_based: pure PyTorch
  - TileLangParallelNew: our TileLang implementation

CHUNKED: Process sequence in chunks, better memory efficiency for longer sequences.
  - fused_chunk_based: hand-optimized Triton
  - naive_chunk_based: pure PyTorch

Produces two graphs:
1. Parallel comparison (up to 16k seq len, naive torch stops at 8k)
2. Chunked comparison (8k to very long sequences)
"""
import torch
import triton
import matplotlib.pyplot as plt

from fla.ops.based.naive import naive_parallel_based, naive_chunk_based
from fla.ops.based import parallel_based, fused_chunk_based

# Import TileLang implementations
import sys
import importlib.util
spec = importlib.util.spec_from_file_location(
    "5_21_ZZZ", 
    "/home/simon/willyc/dsl-monkeys/runs/lv5-gemini-3-pro-preview-BIGRUNREAL/archive/kernels/level5/5_21_ZZZ.py"
)
module_5_21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_5_21)
TileLangParallelNew = module_5_21.ModelNew

# Import TileLang chunked implementation
spec_chunked = importlib.util.spec_from_file_location(
    "based_la_chunked", 
    "/home/simon/willyc/dslmonkey-accelerated/MSMD_KERNELS/based_la_chunked.py"
)
module_chunked = importlib.util.module_from_spec(spec_chunked)
spec_chunked.loader.exec_module(module_chunked)
TileLangChunkedNew = module_chunked.ModelNew


def check_correctness(T=512, atol=1e-2, rtol=1e-2):
    """
    Compare TileLangParallelNew against naive PyTorch reference.
    Uses float16 for correctness checking to avoid bf16 precision issues.
    """
    from fla.utils import device

    dtype = torch.float16
    B, H, D = 8, 16, 128
    
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    
    # Reference output
    out_torch = naive_parallel_based(q, k, v)
    
    # TileLangParallelNew
    model = TileLangParallelNew()
    out_tilelang = model(q, k, v)
    max_diff = (out_torch - out_tilelang).abs().max().item()
    mean_diff = (out_torch - out_tilelang).abs().mean().item()
    is_close = torch.allclose(out_torch, out_tilelang, atol=atol, rtol=rtol)
    
    print(f"Correctness check (T={T}):")
    print(f"  TileLangParallelNew vs naive_parallel_based: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={is_close}")
    
    if not is_close:
        print("  WARNING: TileLangParallelNew outputs differ significantly!")
    
    return is_close


_correctness_checked = set()

# ============================================================================
# PARALLEL BENCHMARK (low sequence lengths, up to 16k)
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        # 256, 512, 1k, 2k, 4k, 8k, 16k
        # x_vals=[256, 512, 1024, 2048, 4096, 8192, 16384],
        x_vals=[256, 512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        # line_vals=['naive_torch', 'parallel_triton', 'tilelang_parallel'],
        # line_names=['Naive PyTorch', 'Triton (parallel_based)', 'TileLang (ours)'],
        line_vals=['parallel_triton', 'tilelang_parallel'],
        line_names=['Triton (parallel_based)', 'TileLang (ours)'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="Execution Time (ms)",
        plot_name="BASED_Parallel_Comparison",
        args={},
    ),
)
def benchmark_parallel(T, provider):
    """Benchmark parallel implementations at low-medium sequence lengths."""
    from fla.utils import device
    dtype = torch.bfloat16
    B, H, D = 8, 16, 128
    quantiles = [0.5, 0.2, 0.8]
    results = (0, 0, 0)
    
    if provider == 'naive_torch':
        # Naive torch OOMs at very long sequences, cap at 8k
        if T > 8192:
            return results
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)

        results = triton.testing.do_bench(lambda: naive_parallel_based(q, k, v), quantiles=quantiles)
        # Run correctness check after benchmarking this T value (only once per T)
        if T not in _correctness_checked:
            _correctness_checked.add(T)
            check_correctness(T=T)
        return results
    
    elif provider == 'parallel_triton':
        # Triton parallel_based uses (B, T, H, D) format with feature_dim=16
        q = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
        return triton.testing.do_bench(lambda: parallel_based(q, k, v), quantiles=quantiles)
    
    elif provider == 'tilelang_parallel':
        # TileLang uses (B, H, T, D) format
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        model = TileLangParallelNew()
        return triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)
    
    return results


# ============================================================================
# CHUNKED BENCHMARK (long sequences, 8k to 128k+)
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        # 8k, 16k, 32k, 64k, 128k
        x_vals=[16384, 32768, 65536],
        line_arg='provider',
        line_vals=['naive_chunk_torch', 'fused_chunk_triton', 'tilelang_chunked'],
        line_names=['Naive Chunk PyTorch', 'Triton (fused_chunk_based)', 'TileLang (ours)'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="Execution Time (ms)",
        plot_name="BASED_Chunked_Comparison",
        args={},
    ),
)
def benchmark_chunked(T, provider):
    """Benchmark chunked implementations at long sequence lengths."""
    from fla.utils import device
    dtype = torch.bfloat16
    B, H, D = 8, 16, 128
    quantiles = [0.5, 0.2, 0.8]
    results = (0, 0, 0)
    
    if provider == 'naive_chunk_torch':
        # naive_chunk_based uses (B, H, T, D) format with feature_dim=16
        q = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        return triton.testing.do_bench(lambda: naive_chunk_based(q, k, v), quantiles=quantiles)
    
    elif provider == 'fused_chunk_triton':
        # fused_chunk_based uses (B, T, H, D) format with feature_dim=16
        q = torch.randn(B, T, H, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, T, H, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
        return triton.testing.do_bench(lambda: fused_chunk_based(q, k, v), quantiles=quantiles)
    
    # elif provider == 'tilelang_chunked':
    #     # TileLang chunked uses (B, H, T, D) format
    #     q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
    #     k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
    #     v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
    #     model = TileLangChunkedNew()
    #     return triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)
    
    return results


if __name__ == '__main__':
    import os
    save_path = './plots/'
    os.makedirs(save_path, exist_ok=True)
    
    print("=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)
    check_correctness(T=512)
    
    print("\n" + "=" * 60)
    print("PARALLEL BENCHMARK (low-medium sequence lengths)")
    print("=" * 60)
    benchmark_parallel.run(print_data=True, show_plots=True, save_path=save_path)
    
    print("\n" + "=" * 60)
    print("CHUNKED BENCHMARK (long sequence lengths)")
    print("=" * 60)
    benchmark_chunked.run(print_data=True, show_plots=True, save_path=save_path)
    
    print(f"\nPlots saved to {save_path}")

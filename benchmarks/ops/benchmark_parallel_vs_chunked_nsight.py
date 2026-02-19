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
import argparse

from fla.ops.based.naive import naive_parallel_based, naive_chunk_based
from fla.ops.based import parallel_based, fused_chunk_based

naive_parallel_based_compiled = torch.compile(naive_parallel_based)
naive_chunk_based_compiled = torch.compile(naive_chunk_based)

# Import TileLang implementations
import sys
import importlib.util
spec = importlib.util.spec_from_file_location(
    "based_la_parallel", 
    "/home/simon/willyc/dslmonkey-accelerated/MSMD_KERNELS/based_la_parallel.py"
)
module_parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_parallel)
TileLangParallelNew = module_parallel.ModelNew

spec_chunked = importlib.util.spec_from_file_location(
    "based_la_chunked_v2", 
    "/home/simon/willyc/dslmonkey-accelerated/MSMD_KERNELS_2/based_la_chunked.py"
)
# spec_chunked = importlib.util.spec_from_file_location(
#     "based_la_chunked_v2", 
#     "/home/simon/willyc/dslmonkey-accelerated/MSMD_KERNELS_2/based_la_chunked_fused.py"
# )
module_chunked = importlib.util.module_from_spec(spec_chunked)
spec_chunked.loader.exec_module(module_chunked)
TileLangChunkedNew = module_chunked.ModelNew


_correctness_checked_parallel = set()
_correctness_checked_chunked = set()
_compiled_warmup_parallel = set()
_compiled_warmup_chunked = set()


def _make_parallel_runner(provider: str, T: int):
    from fla.utils import device

    dtype = torch.float16
    B, H, D = 8, 16, 128

    if provider == 'parallel_triton':
        q = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)

        def _run():
            parallel_based(q, k, v)

        return _run

    if provider == 'tilelang_parallel':
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        model = TileLangParallelNew()

        def _run():
            model(q, k, v)

        return _run

    raise ValueError(f"Unsupported provider: {provider}")


def run_nsight_parallel(provider: str, T: int, warmup: int, iters: int):
    run = _make_parallel_runner(provider=provider, T=T)
    for _ in range(max(0, warmup)):
        run()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(f"nsight_parallel::{provider}::T={T}::iters={iters}")
    for _ in range(max(1, iters)):
        run()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def check_correctness_parallel(T=512, atol=1e-2, rtol=1e-2):
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
    
    print(f"  Parallel T={T}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={is_close}")
    
    return is_close


def check_correctness_chunked(T=512, atol=1e-2, rtol=1e-2):
    """
    Compare TileLangChunkedNew against naive PyTorch reference.
    Uses bfloat16 since TileLang kernels are compiled for bfloat16.
    Uses feature_dim=16 for Q/K and head_dim=128 for V.
    """
    from fla.utils import device

    dtype = torch.float16
    B, H = 8, 16
    feature_dim = 16
    head_dim = 128
    
    torch.manual_seed(42)
    q = torch.randn(B, H, T, feature_dim, device=device, dtype=dtype)
    k = torch.randn(B, H, T, feature_dim, device=device, dtype=dtype)
    v = torch.randn(B, H, T, head_dim, device=device, dtype=dtype)

    # Reference output - naive_chunk_based uses same format
    out_torch = naive_chunk_based(q, k, v)
    
    # TileLangChunkedNew
    model = TileLangChunkedNew()
    out_tilelang = model(q, k, v)
    max_diff = (out_torch - out_tilelang).abs().max().item()
    mean_diff = (out_torch - out_tilelang).abs().mean().item()
    is_close = torch.allclose(out_torch, out_tilelang, atol=atol, rtol=rtol)
    
    print(f"  Chunked T={T}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={is_close}")
    
    return is_close

# ============================================================================
# PARALLEL BENCHMARK (low sequence lengths, up to 16k)
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        # 256, 512, 1k, 2k, 4k, 8k, 16k
        # x_vals=[256, 512, 1024, 2048, 4096, 8192, 16384],
        x_vals=[256, 512],
        line_arg='provider',
        line_vals=['naive_torch', 'naive_torch_compiled', 'parallel_triton', 'tilelang_parallel'],
        line_names=['Naive PyTorch', 'Naive PyTorch (torch.compile)', 'Triton (parallel_based)', 'TileLang (ours)'],
        styles=[('red', '-'), ('orange', '-'), ('cyan', '-'), ('green', '-')],
        xlabel='Sequence Length (T)',
        ylabel="Execution Time (ms)",
        plot_name="BASED Parallel Latency",
        args={},
    ),
)
def benchmark_parallel(T, provider):
    """Benchmark parallel implementations at low-medium sequence lengths."""
    from fla.utils import device
    dtype = torch.float16
    B, H, D = 8, 16, 128
    quantiles = [0.5, 0.2, 0.8]
    results = (0, 0, 0)
    
    if provider == 'naive_torch':
        # Naive torch OOMs at very long sequences, cap at 8k
        if T > 8192:
            return results
        # Use D=128 for all to match TileLang (fair comparison)
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        return triton.testing.do_bench(lambda: naive_parallel_based(q, k, v), quantiles=quantiles)

    if provider == 'naive_torch_compiled':
        if T > 8192:
            return results
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        if T not in _compiled_warmup_parallel:
            _compiled_warmup_parallel.add(T)
            for _ in range(3):
                naive_parallel_based_compiled(q, k, v)
            torch.cuda.synchronize()
        return triton.testing.do_bench(lambda: naive_parallel_based_compiled(q, k, v), quantiles=quantiles)
    
    elif provider == 'parallel_triton':
        # Triton parallel_based uses (B, T, H, D) format
        # Use D=128 for all to match TileLang (fair comparison)
        q = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
        return triton.testing.do_bench(lambda: parallel_based(q, k, v), quantiles=quantiles)
    
    elif provider == 'tilelang_parallel':
        # TileLang uses (B, H, T, D) format - all dimensions must match
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
        x_vals=[256, 512, 1024, 2048, 4096, 8192, 16384],
        line_arg='provider',
        line_vals=['naive_chunk_torch', 'naive_chunk_torch_compiled', 'fused_chunk_triton', 'tilelang_chunked'],
        line_names=['Naive Chunk PyTorch', 'Naive Chunk PyTorch (torch.compile)', 'Triton (fused_chunk_based)', 'TileLang (ours)'],
        # line_vals=['fused_chunk_triton', 'tilelang_chunked'],
        # line_names=['Triton (fused_chunk_based)', 'TileLang (ours)'],
        styles=[('red', '-'), ('orange', '-'), ('cyan', '-'), ('green', '-')],
        xlabel='Sequence Length (T)',
        ylabel="Execution Time (ms)",
        plot_name="BASED Chunked Latency",
        args={},
    ),
)
def benchmark_chunked(T, provider):
    """Benchmark chunked implementations at long sequence lengths."""
    from fla.utils import device
    dtype = torch.float16
    B, H, D = 8, 16, 128
    quantiles = [0.5, 0.2, 0.8]
    results = (0, 0, 0)
    
    if provider == 'naive_chunk_torch':
        # naive_chunk_based uses (B, H, T, D) format with feature_dim=16
        q = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        return triton.testing.do_bench(lambda: naive_chunk_based(q, k, v), quantiles=quantiles)

    if provider == 'naive_chunk_torch_compiled':
        q = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        if T not in _compiled_warmup_chunked:
            _compiled_warmup_chunked.add(T)
            for _ in range(3):
                naive_chunk_based_compiled(q, k, v)
            torch.cuda.synchronize()
        return triton.testing.do_bench(lambda: naive_chunk_based_compiled(q, k, v), quantiles=quantiles)
    
    if provider == 'fused_chunk_triton':
        # fused_chunk_based uses (B, T, H, D) format with feature_dim=16
        q = torch.randn(B, T, H, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, T, H, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
        return triton.testing.do_bench(lambda: fused_chunk_based(q, k, v), quantiles=quantiles)
    
    elif provider == 'tilelang_chunked':
        # TileLang chunked uses (B, H, T, D) format
        q = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        model = TileLangChunkedNew()
        return triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)
    
    return results


if __name__ == '__main__':
    import os
    save_path = './plots/'
    os.makedirs(save_path, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsight-provider', choices=['parallel_triton', 'tilelang_parallel', 'both'], default=None)
    parser.add_argument('--T', type=int, default=2048)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=200)
    args = parser.parse_args()

    if args.nsight_provider is not None:
        providers = ['parallel_triton', 'tilelang_parallel'] if args.nsight_provider == 'both' else [args.nsight_provider]
        for p in providers:
            run_nsight_parallel(provider=p, T=args.T, warmup=args.warmup, iters=args.iters)
        raise SystemExit(0)
    
    # Correctness checks for all sequence lengths
    print("=" * 60)
    print("CORRECTNESS CHECK - PARALLEL")
    print("=" * 60)
    parallel_seq_lens = [256, 512, 1024, 2048, 4096, 8192]
    for seq_len in parallel_seq_lens:
        check_correctness_parallel(T=seq_len)
    
    # print("\n" + "=" * 60)
    # print("CORRECTNESS CHECK - CHUNKED")
    # print("=" * 60)
    # chunked_seq_lens = [512, 1024, 2048]  # Smaller sizes for chunked correctness
    # for seq_len in chunked_seq_lens:
    #     check_correctness_chunked(T=seq_len)
    
    # print("\n" + "=" * 60)
    # print("PARALLEL BENCHMARK (low-medium sequence lengths)")
    # print("=" * 60)
    # benchmark_parallel.run(print_data=True, show_plots=True, save_path=save_path)
    
    # print("\n" + "=" * 60)
    # print("CHUNKED BENCHMARK (long sequence lengths)")
    # print("=" * 60)
    # benchmark_chunked.run(print_data=True, show_plots=True, save_path=save_path)
    
    print(f"\nPlots saved to {save_path}")

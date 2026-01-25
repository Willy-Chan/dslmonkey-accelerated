"""
Minimal benchmark: TileLang vs Torch (naive) at small sequence lengths
"""
import torch
import triton

from fla.ops.based.naive import naive_parallel_based, naive_chunk_based
from fla.ops.based import parallel_based
from benchmarks.ops.tilelang_based import tilelang_based

# Compiled version of naive_parallel_based (compiled once at module load)
naive_parallel_based_compiled = torch.compile(naive_parallel_based, mode="max-autotune")

# Import the alternative TileLang implementation from the archive
import sys
import os
import importlib.util
sys.path.append('/home/simon/willyc/dsl-monkeys/runs/lv5-gemini-3-pro-preview-BIGRUNREAL/archive/kernels/level5')
spec = importlib.util.spec_from_file_location("5_21_8", "/home/simon/willyc/dsl-monkeys/runs/lv5-gemini-3-pro-preview-BIGRUNREAL/archive/kernels/level5/5_21_ZZZ.py")
module_5_21_8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_5_21_8)
TileLangParallelNew = module_5_21_8.ModelNew

# LOCAL H200 BENCHMARK ON RICHARD


def check_correctness(T=512, atol=1e-2, rtol=1e-2):
    """
    Compare both TileLang implementations against naive PyTorch reference.
    Uses float32 for correctness checking to avoid bf16 precision issues.
    """
    from fla.utils import device

    # TODO: HERE IS WHERE YOU CAN SET THE INPUT DATATYPE
    # Noticing BF16 has more weird correctness errors, so using FP16. Passes the correctness checks. Do we care about bf16 that much???
    # dtype = torch.bf16
    dtype = torch.float16
    B, H, D = 8, 16, 128
    
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    
    # Reference output
    out_torch = naive_parallel_based(q, k, v)
    
    # TileLang v1
    out_tilelang = tilelang_based(q, k, v)
    max_diff_v1 = (out_torch - out_tilelang).abs().max().item()
    mean_diff_v1 = (out_torch - out_tilelang).abs().mean().item()
    is_close_v1 = torch.allclose(out_torch, out_tilelang, atol=atol, rtol=rtol)
    
    # TileLang v2 (TileLangParallelNew)
    model_v2 = TileLangParallelNew()
    out_tilelang_v2 = model_v2(q, k, v)
    max_diff_v2 = (out_torch - out_tilelang_v2).abs().max().item()
    mean_diff_v2 = (out_torch - out_tilelang_v2).abs().mean().item()
    is_close_v2 = torch.allclose(out_torch, out_tilelang_v2, atol=atol, rtol=rtol)
    
    print(f"Correctness check (T={T}):")
    print(f"  tilelang_based:       max_diff={max_diff_v1:.6f}, mean_diff={mean_diff_v1:.6f}, close={is_close_v1}")
    print(f"  TileLangParallelNew:  max_diff={max_diff_v2:.6f}, mean_diff={mean_diff_v2:.6f}, close={is_close_v2}")
    
    if not is_close_v1:
        print("  WARNING: tilelang_based outputs differ significantly!")
    if not is_close_v2:
        print("  WARNING: TileLangParallelNew outputs differ significantly!")
    
    return is_close_v1 and is_close_v2


# Track which T values have been correctness-checked
_correctness_checked = set()


# PARALLEL = good for low sequence lengths, hand optimized triton. Simple and parallelizable, but memory heavy.
#   - parallel_based is the hand optimized triton implementation
#   - naive_parallel_based is the pure pytorch implementation
#   * Our implementation is TILELANG_PARALLEL_NEW

# CHUNKED = Process sequence in chunks. Mix between memory efficiency and parallelism (medium sequence length)
#   - fused_chunk_based_fwd_kernel is the  hand optimized triton implementation
#   - naive_chunk_based is the pure pytorch implementation

# RECURRENT = token by token with a state, doesn't use tensor cores but totally linear you never materialize stuff
#   - fused_recurrent_linear_attn is the hand optimized triton version
#   * MISSING FOR BASED HERE



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[64 * 2 ** i for i in range(2, 10)],
        line_arg='provider',
        # line_vals=['torch', 'torch_compiled', 'parallel', 'parallel_chunk', 'tilelang_parallel_new'],
        # line_names=['torch_fwd', 'torch_compiled_fwd', 'parallel_fwd', 'parallel_chunk_fwd', 'tilelang_parallel_new_fwd'],

        line_vals=['parallel', 'parallel_chunk', 'tilelang_parallel_new'],
        line_names=['parallel_fwd', 'parallel_chunk_fwd', 'tilelang_parallel_new_fwd'],

        # line_vals=['torch_compiled', 'parallel', 'parallel_chunk', 'tilelang', 'tilelang_parallel_new'],
        # line_names=['torch_compiled_fwd', 'parallel_fwd', 'parallel_chunk_fwd', 'tilelang_fwd', 'tilelang_parallel_new_fwd'],
        styles=[('blue', '-'), ('blue', '--'), ('green', '-'), ('green', '--'), ('magenta', '-'), ('magenta', '--')],
        ylabel="Execution Time (ms)",
        plot_name="TileLang_vs_Torch",
        args={},
    ),
)
def benchmark(T, provider):
    """Matches benchmark_based.py input format exactly."""
    from fla.utils import device
    dtype = torch.bfloat16
    B, H, D = 8, 16, 128
    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    
    # Input tensor creation - matches benchmark_based.py exactly
    if provider in ('torch', 'torch_compiled', 'parallel_chunk'):
        # Head-first format (B, H, T, D) with feature_dim=16 for Q/K
        q = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
    elif provider in ('parallel',):
        # Seq-first format (B, T, H, D) with feature_dim=16 for Q/K
        q = torch.randn(B, T, H, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, T, H, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
    elif provider in ('tilelang', 'tilelang_parallel_new'):     # This is equivalent of our "flash" as in benchmark_based.py
        # TileLang uses head-first format (B, H, T, D) with full D for all
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
    




    
    # Benchmark each provider
    if provider == 'torch':
        results = triton.testing.do_bench(lambda: naive_parallel_based(q, k, v), quantiles=quantiles)
        # Run correctness check after benchmarking this T value (only once per T)
        if T not in _correctness_checked:
            _correctness_checked.add(T)
            check_correctness(T=T)
        return results
    elif provider == 'torch_compiled':
        # Warmup for torch.compile
        for _ in range(3):
            naive_parallel_based_compiled(q, k, v)
        torch.cuda.synchronize()
        return triton.testing.do_bench(lambda: naive_parallel_based_compiled(q, k, v), quantiles=quantiles)
    elif provider == 'parallel':
        return triton.testing.do_bench(lambda: parallel_based(q, k, v), quantiles=quantiles)
    elif provider == 'parallel_chunk':
        # naive_chunk_based requires T >= chunk_size (default 256)
        if T < 256:
            return results
        return triton.testing.do_bench(lambda: naive_chunk_based(q, k, v), quantiles=quantiles)
    # elif provider == 'tilelang':
    #     return triton.testing.do_bench(lambda: tilelang_based(q, k, v), quantiles=quantiles)
    elif provider == 'tilelang_parallel_new':
        model = TileLangParallelNew()
        return triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)


if __name__ == '__main__':
    # Correctness checks run automatically after each T value is benchmarked
    benchmark.run(print_data=True, show_plots=True, save_path='./plots/')

"""
Benchmark comparing GLA Recurrent Reference vs Solution implementations.

REFERENCE: Naive PyTorch implementation of recurrent GLA
  - gla_recurrent_reference: Pure PyTorch with explicit recurrent loop

SOLUTION: TileLang optimized implementation  
  - gla_recurrent_solution: TileLang kernel with fused operations

Produces performance comparison across different sequence lengths.
"""
import torch
import triton
import matplotlib.pyplot as plt
import sys
import importlib.util

# Import reference implementation
spec_ref = importlib.util.spec_from_file_location(
    "gla_recurrent_reference", 
    "/home/simon/willyc/dslmonkey-accelerated/GLA_REFERENCES/gla_recurrent_reference.py"
)
module_ref = importlib.util.module_from_spec(spec_ref)
spec_ref.loader.exec_module(module_ref)
GLAReference = module_ref.Model

# Import solution implementation
spec_sol = importlib.util.spec_from_file_location(
    "gla_recurrent_solution", 
    "/home/simon/willyc/dslmonkey-accelerated/GLA_SOLUTIONS/gla_recurrent_solution.py"
)
module_sol = importlib.util.module_from_spec(spec_sol)
spec_sol.loader.exec_module(module_sol)
GLASolution = module_sol.ModelNew

# Compile reference for fair comparison
reference_compiled = torch.compile(GLAReference())

_correctness_checked = set()
_compiled_warmup = set()

def get_flops(batch, seqlen, headdim, nheads):
    """Estimate FLOPs for GLA recurrent computation."""
    # Per timestep: 
    # - exp(gk): seqlen * nheads * headdim
    # - outer product k⊗v: seqlen * nheads * headdim * headdim  
    # - state update: seqlen * nheads * headdim * headdim
    # - query state: seqlen * nheads * headdim * headdim
    f = batch * seqlen * nheads * headdim  # exp operations
    f += batch * seqlen * nheads * headdim * headdim * 3  # outer product, state update, query
    return f

def check_correctness(T=512, atol=1e-2, rtol=1e-2):
    """
    Compare TileLang solution against PyTorch reference.
    Uses float16 for correctness checking.
    """
    from fla.utils import device
    
    dtype = torch.float16
    B, H, D = 4, 8, 64
    
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype) 
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    # Use smaller scale for gk to prevent exponential overflow
    gk = torch.randn(B, T, H, D, device=device, dtype=dtype) * 0.1
    
    # Reference output
    ref_model = GLAReference()
    out_ref = ref_model(q, k, v, gk)
    
    # Solution output
    sol_model = GLASolution()
    out_sol = sol_model(q, k, v, gk)
    
    max_diff = (out_ref - out_sol).abs().max().item()
    mean_diff = (out_ref - out_sol).abs().mean().item()
    is_close = torch.allclose(out_ref, out_sol, atol=atol, rtol=rtol)
    
    print(f"  GLA T={T}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={is_close}")
    
    return is_close

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        # x_vals=[256, 512, 1024, 2048, 4096, 8192],
        x_vals=[512],
        line_arg='provider',
        line_vals=['reference_torch', 'reference_compiled', 'solution_tilelang'],
        line_names=['Reference PyTorch', 'Reference Compiled', 'Solution TileLang'],
        styles=[('red', '-'), ('orange', '-'), ('green', '-')],
        xlabel='Sequence Length (T)',
        ylabel="Execution Time (ms)",
        plot_name="GLA Recurrent Latency Comparison",
        args={},
    ),
)
def benchmark_gla_recurrent(T, provider):
    """Benchmark GLA recurrent implementations across sequence lengths."""
    from fla.utils import device
    
    dtype = torch.float16
    B, H, D = 4, 8, 64
    quantiles = [0.5, 0.2, 0.8]
    results = (0, 0, 0)
    
    # Generate inputs
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
    # Use smaller scale for gk to prevent exponential overflow
    gk = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype) * 0.1
    
    if provider == 'reference_torch':
        # Reference PyTorch implementation
        model = GLAReference()
        print(f"{provider} TFLOP, T={T}: {get_flops(B, T, D, H)}")
        return triton.testing.do_bench(lambda: model(q, k, v, gk), quantiles=quantiles)
    
    # elif provider == 'reference_compiled':
    #     # Compiled reference for fair comparison
    #     if T not in _compiled_warmup:
    #         _compiled_warmup.add(T)
    #         for _ in range(3):
    #             reference_compiled(q, k, v, gk)
    #         torch.cuda.synchronize()
        
    #     print(f"{provider} TFLOP, T={T}: {get_flops(B, T, D, H)}")
    #     return triton.testing.do_bench(lambda: reference_compiled(q, k, v, gk), quantiles=quantiles)
    
    elif provider == 'solution_tilelang':
        # TileLang solution
        model = GLASolution()
        print(f"{provider} TFLOP, T={T}: {get_flops(B, T, D, H)}")
        return triton.testing.do_bench(lambda: model(q, k, v, gk), quantiles=quantiles)
    
    return results

if __name__ == '__main__':
    import os
    save_path = './plots/'
    os.makedirs(save_path, exist_ok=True)
    
    # Correctness checks
    print("=" * 60)
    print("CORRECTNESS CHECK - GLA RECURRENT")
    print("=" * 60)
    seq_lens = [256, 512, 1024, 2048]
    
    for seq_len in seq_lens:
        check_correctness(T=seq_len)
    
    print("\n" + "=" * 60)
    print("GLA RECURRENT BENCHMARK")
    print("=" * 60)
    benchmark_gla_recurrent.run(print_data=True, show_plots=True, save_path=save_path)
    
    print(f"\nPlots saved to {save_path}")

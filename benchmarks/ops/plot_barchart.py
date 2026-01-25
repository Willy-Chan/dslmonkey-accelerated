"""
Bar chart visualization for TileLang vs Torch benchmark results.
Displays grouped bars for each input token sequence length.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import triton

from fla.ops.based.naive import naive_parallel_based, naive_chunk_based
from fla.ops.based import parallel_based

# Import the alternative TileLang implementation from the archive
import sys
import importlib.util
sys.path.append('/home/simon/willyc/dsl-monkeys/runs/lv5-gemini-3-pro-preview-BIGRUNREAL/archive/kernels/level5')
spec = importlib.util.spec_from_file_location("5_21_8", "/home/simon/willyc/dsl-monkeys/runs/lv5-gemini-3-pro-preview-BIGRUNREAL/archive/kernels/level5/5_21_ZZZ.py")
module_5_21_8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_5_21_8)
TileLangParallelNew = module_5_21_8.ModelNew


def run_benchmark(T, provider, device, dtype=torch.bfloat16):
    """Run benchmark for a single T value and provider."""
    B, H, D = 8, 16, 128
    quantiles = [0.5, 0.2, 0.8]
    
    if provider in ('torch', 'torch_compiled', 'parallel_chunk'):
        q = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
    elif provider in ('parallel',):
        q = torch.randn(B, T, H, 16, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, T, H, 16, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=False, dtype=dtype)
    elif provider in ('tilelang_parallel_new',):
        q = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=False, dtype=dtype)
    else:
        return None
    
    if provider == 'torch':
        return triton.testing.do_bench(lambda: naive_parallel_based(q, k, v), quantiles=quantiles)
    elif provider == 'parallel':
        return triton.testing.do_bench(lambda: parallel_based(q, k, v), quantiles=quantiles)
    elif provider == 'parallel_chunk':
        if T < 256:
            return (0, 0, 0)
        return triton.testing.do_bench(lambda: naive_chunk_based(q, k, v), quantiles=quantiles)
    elif provider == 'tilelang_parallel_new':
        model = TileLangParallelNew()
        return triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)
    
    return None


def collect_benchmark_data():
    """Collect benchmark data for all T values and providers."""
    from fla.utils import device
    
    T_vals = [64 * 2 ** i for i in range(2, 10)]
    providers = ['parallel', 'parallel_chunk', 'tilelang_parallel_new']
    provider_names = ['Parallel (Triton)', 'Parallel Chunk (PyTorch)', 'TileLang']
    
    results = {p: [] for p in providers}
    
    for T in T_vals:
        print(f"Benchmarking T={T}...")
        for provider in providers:
            res = run_benchmark(T, provider, device)
            if res is not None:
                results[provider].append(res[0])  # median
            else:
                results[provider].append(0)
    
    return T_vals, providers, provider_names, results


def plot_barchart(T_vals, providers, provider_names, results, save_path='./plots/'):
    """Create grouped bar chart visualization."""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    x = np.arange(len(T_vals))
    width = 0.25
    n_providers = len(providers)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    for i, (provider, name) in enumerate(zip(providers, provider_names)):
        offset = (i - n_providers / 2 + 0.5) * width
        bars = ax.bar(x + offset, results[provider], width, label=name, color=colors[i % len(colors)])
        
    ax.set_xlabel('Sequence Length (T)', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title('TileLang vs Torch: Execution Time by Sequence Length', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in T_vals], rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_file = os.path.join(save_path, 'barchart_tilelang_vs_torch.png')
    plt.savefig(save_file, dpi=150)
    print(f"Saved bar chart to {save_file}")
    
    plt.show()


if __name__ == '__main__':
    T_vals, providers, provider_names, results = collect_benchmark_data()
    plot_barchart(T_vals, providers, provider_names, results)

import modal

app = modal.App("based-benchmark")

# Define the image - use CUDA base image for tilelang compatibility
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install([
        "torch==2.5.1",
        "triton==3.1.0",
        "einops",
        "transformers>=4.45.0",
        "flash-linear-attention",  # Install fla from PyPI
        "tilelang",
    ])
)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
)
def run_based_benchmark():
    import torch
    import torch.nn as nn
    import tilelang
    import tilelang.language as T
    
    print("=" * 60)
    print("TileLang vs PyTorch Reference - Based Linear Attention")
    print("=" * 60)
    
    # Import the reference kernel
    from fla.ops.based.naive import naive_parallel_based
    
    # Define TileLang kernel inline
    def _build_linear_attn_kernel(batch, heads, seq_len, dim, scale, dtype="bfloat16"):
        BLOCK_M = 64
        BLOCK_N = 64
        
        @T.prim_func
        def kernel(
            Q: T.Tensor((batch, heads, seq_len, dim), dtype),
            K: T.Tensor((batch, heads, seq_len, dim), dtype),
            V: T.Tensor((batch, heads, seq_len, dim), dtype),
            Output: T.Tensor((batch, heads, seq_len, dim), dtype)
        ):
            with T.Kernel(T.ceildiv(seq_len, BLOCK_M), heads, batch, threads=128) as (bx, by, bz):
                Q_shared = T.alloc_shared((BLOCK_M, dim), dtype)
                K_shared = T.alloc_shared((BLOCK_N, dim), dtype)
                V_shared = T.alloc_shared((BLOCK_N, dim), dtype)
                S_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
                Z_shared = T.alloc_shared((BLOCK_M,), "float32")
                
                Acc_O = T.alloc_fragment((BLOCK_M, dim), "float32")
                S_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
                
                T.clear(Acc_O)
                T.clear(Z_shared)
                
                T.annotate_layout({
                    Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                    K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                    V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                    S_shared: tilelang.layout.make_swizzled_layout(S_shared),
                })
                
                for i, j in T.Parallel(BLOCK_M, dim):
                    r = bx * BLOCK_M + i
                    if r < seq_len:
                        val = T.cast(Q[bz, by, r, j], "float32")
                        Q_shared[i, j] = T.cast(val * scale, dtype)
                    else:
                        Q_shared[i, j] = T.cast(0.0, dtype)
                
                T.copy(Q_shared, Q_shared)
                
                loop_limit = bx + 1
                
                for k in T.Pipelined(loop_limit, num_stages=1):
                    for i, j in T.Parallel(BLOCK_N, dim):
                        c = k * BLOCK_N + i
                        if c < seq_len:
                            K_shared[i, j] = K[bz, by, c, j]
                        else:
                            K_shared[i, j] = T.cast(0.0, dtype)
                    
                    for i, j in T.Parallel(BLOCK_N, dim):
                        c = k * BLOCK_N + i
                        if c < seq_len:
                            V_shared[i, j] = V[bz, by, c, j]
                        else:
                            V_shared[i, j] = T.cast(0.0, dtype)
                    
                    T.clear(S_frag)
                    T.gemm(Q_shared, K_shared, S_frag, transpose_B=True)
                    
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        row_gl = bx * BLOCK_M + i
                        col_gl = k * BLOCK_N + j
                        
                        if row_gl < seq_len and col_gl < seq_len and row_gl >= col_gl:
                            val = S_frag[i, j]
                            p_val = 1.0 + val + 0.5 * (val * val)
                            
                            S_frag[i, j] = p_val
                            S_shared[i, j] = T.cast(p_val, dtype)
                            
                            T.atomic_add(Z_shared[i], p_val)
                        else:
                            S_shared[i, j] = T.cast(0.0, dtype)
                    
                    T.copy(S_shared, S_shared)
                    T.gemm(S_shared, V_shared, Acc_O)
                
                for i, j in T.Parallel(BLOCK_M, dim):
                    r = bx * BLOCK_M + i
                    if r < seq_len:
                        z_val = Z_shared[i]
                        o_val = Acc_O[i, j]
                        norm = o_val / (z_val + 1e-6)
                        Output[bz, by, r, j] = T.cast(norm, dtype)

        return tilelang.compile(kernel, out_idx=[3], target="cuda")

    class TileLangBasedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self._kernel_cache = {}

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            b, h, l, d = q.shape
            
            dtype_str = "bfloat16" if q.dtype == torch.bfloat16 else "float16"
            key = (b, h, l, d, dtype_str)
            
            if key not in self._kernel_cache:
                scale = d ** -0.5
                self._kernel_cache[key] = _build_linear_attn_kernel(b, h, l, d, scale, dtype=dtype_str)
                
            kernel = self._kernel_cache[key]
            
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            
            return kernel(q, k, v)

    # Create tilelang instance
    tilelang_based = TileLangBasedAttention()
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # Test configurations
    B, H, D = 8, 16, 128  # batch, heads, head_dim
    seq_lengths = [1024, 2048, 4096, 8192, 16384]
    
    results = []
    
    print(f"\nConfig: B={B}, H={H}, D={D}, dtype={dtype}")
    print("-" * 60)
    print(f"{'Seq Len':>10} | {'PyTorch (ms)':>14} | {'TileLang (ms)':>14} | {'Speedup':>10}")
    print("-" * 60)
    
    for T in seq_lengths:
        # Create input tensors (head-first format for both)
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        
        quantiles = [0.5, 0.2, 0.8]
        
        # Benchmark PyTorch reference
        try:
            if T <= 2048:  # PyTorch naive can be slow for long sequences
                pytorch_time = triton.testing.do_bench(
                    lambda: naive_parallel_based(q, k, v), 
                    quantiles=quantiles
                )[0]
            else:
                pytorch_time = float('inf')
        except Exception as e:
            print(f"PyTorch failed at T={T}: {e}")
            pytorch_time = float('inf')
        
        # Benchmark TileLang kernel
        try:
            tilelang_time = triton.testing.do_bench(
                lambda: tilelang_based(q, k, v), 
                quantiles=quantiles
            )[0]
        except Exception as e:
            print(f"TileLang failed at T={T}: {e}")
            tilelang_time = float('inf')
        
        # Calculate speedup
        if pytorch_time != float('inf') and tilelang_time != float('inf'):
            speedup = pytorch_time / tilelang_time
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup = None
            speedup_str = "N/A"
        
        pytorch_str = f"{pytorch_time:.3f}" if pytorch_time != float('inf') else "OOM/Skip"
        tilelang_str = f"{tilelang_time:.3f}" if tilelang_time != float('inf') else "OOM/Skip"
        
        print(f"{T:>10} | {pytorch_str:>14} | {tilelang_str:>14} | {speedup_str:>10}")
        
        results.append({
            "seq_len": T,
            "pytorch_ms": pytorch_time,
            "tilelang_ms": tilelang_time,
            "speedup": speedup
        })
    
    print("-" * 60)
    
    # Correctness check at smaller size
    print("\n" + "=" * 60)
    print("Correctness Check (T=512)")
    print("=" * 60)
    
    T_check = 512
    q = torch.randn(B, H, T_check, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T_check, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T_check, D, device=device, dtype=dtype)
    
    try:
        out_pytorch = naive_parallel_based(q, k, v)
        out_tilelang = tilelang_based(q, k, v)
        
        # Compare outputs
        diff = (out_pytorch - out_tilelang).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"Max absolute difference:  {max_diff:.6e}")
        print(f"Mean absolute difference: {mean_diff:.6e}")
        
        if max_diff < 1e-2:
            print("✓ Results match within tolerance!")
        else:
            print("⚠ Results differ significantly - check implementation")
    except Exception as e:
        print(f"Correctness check failed: {e}")
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    with app.run():
        results = run_based_benchmark.remote()
        print("\nReturned results:", results)

"""
Entry ID: b0f7977d0b08
Problem: Level 5 Problem 23 - 23_chunk_linear_attention
Is Seed: False
Iteration Added: 21
Speedup (Eager): 19.66x
Speedup (Compile): 7.93x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-24T03:21:40.599545
Parents: ['seed_0_27', 'seed_0_132', 'seed_0_138', '5_7', '5_11', '5_7', '5_1', '5_11', '5_12', '5_17']
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

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
            # Allocations
            Q_shared = T.alloc_shared((BLOCK_M, dim), dtype)
            K_shared = T.alloc_shared((BLOCK_N, dim), dtype)
            V_shared = T.alloc_shared((BLOCK_N, dim), dtype)
            S_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            Z_shared = T.alloc_shared((BLOCK_M,), "float32")
            
            # Accumulators
            Acc_O = T.alloc_fragment((BLOCK_M, dim), "float32")
            S_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            
            # Initialize Accumulators
            T.clear(Acc_O)
            T.clear(Z_shared)
            
            # Layout optimization
            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                S_shared: tilelang.layout.make_swizzled_layout(S_shared),
            })
            
            # 1. Load Q tile
            for i, j in T.Parallel(BLOCK_M, dim):
                r = bx * BLOCK_M + i
                if r < seq_len:
                    val = T.cast(Q[bz, by, r, j], "float32")
                    Q_shared[i, j] = T.cast(val * scale, dtype)
                else:
                    Q_shared[i, j] = T.cast(0.0, dtype)
            
            T.copy(Q_shared, Q_shared) # Barrier
            
            # 2. Iterate over Key/Value blocks (Causal loop)
            loop_limit = bx + 1
            
            for k in T.Pipelined(loop_limit, num_stages=1):
                # Load K
                for i, j in T.Parallel(BLOCK_N, dim):
                    c = k * BLOCK_N + i
                    if c < seq_len:
                        K_shared[i, j] = K[bz, by, c, j]
                    else:
                        K_shared[i, j] = T.cast(0.0, dtype)
                
                # Load V
                for i, j in T.Parallel(BLOCK_N, dim):
                    c = k * BLOCK_N + i
                    if c < seq_len:
                        V_shared[i, j] = V[bz, by, c, j]
                    else:
                        V_shared[i, j] = T.cast(0.0, dtype)
                
                # Compute S = Q @ K.T
                T.clear(S_frag)
                T.gemm(Q_shared, K_shared, S_frag, transpose_B=True)
                
                # Apply Polynomial and Mask, then store to S_shared and accumulate Z
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row_gl = bx * BLOCK_M + i
                    col_gl = k * BLOCK_N + j
                    
                    if row_gl < seq_len and col_gl < seq_len and row_gl >= col_gl:
                        val = S_frag[i, j]
                        # P = 1 + S + 0.5 * S^2
                        p_val = 1.0 + val + 0.5 * (val * val)
                        
                        S_frag[i, j] = p_val
                        S_shared[i, j] = T.cast(p_val, dtype)
                        
                        T.atomic_add(Z_shared[i], p_val)
                    else:
                        S_shared[i, j] = T.cast(0.0, dtype)
                
                T.copy(S_shared, S_shared)
                
                # Compute O += S_shared @ V
                T.gemm(S_shared, V_shared, Acc_O)
            
            # 3. Finalize and Store
            for i, j in T.Parallel(BLOCK_M, dim):
                r = bx * BLOCK_M + i
                if r < seq_len:
                    z_val = Z_shared[i]
                    o_val = Acc_O[i, j]
                    norm = o_val / (z_val + 1e-6)
                    Output[bz, by, r, j] = T.cast(norm, dtype)

    return tilelang.compile(kernel, out_idx=[3], target="cuda")


class TileLangBasedAttention(nn.Module):
    """TileLang-based linear attention kernel for benchmarking."""
    
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


# Global instance for benchmark use
_tilelang_based = None

def tilelang_based(q, k, v):
    """Functional interface for tilelang-based attention."""
    global _tilelang_based
    if _tilelang_based is None:
        _tilelang_based = TileLangBasedAttention()
    return _tilelang_based(q, k, v)

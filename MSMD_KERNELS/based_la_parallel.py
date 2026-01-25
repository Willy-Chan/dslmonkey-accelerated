"""
Entry ID: decef183cdeb
Problem: Level 5 Problem 21 - 21_naive_chunk_based
Is Seed: False
Iteration Added: 33
Speedup (Eager): 95.34x
Speedup (Compile): 39.59x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-24T16:33:35.182293
Parents: ['seed_0_148', 'seed_0_109', 'seed_0_126', '5_16', '5_13', '5_16', '5_13', '5_11', '5_12', '5_17']
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def _build_based_linear_attn_kernel(batch, heads, seq_len, dim, dtype="bfloat16"):
    """
    Based Linear Attention Kernel.
    Computes O = (sum(P @ V)) / sum(P)
    where P = causal_mask(1 + S + 0.5*S^2)
    and S = (Q @ K^T) / sqrt(D)
    """
    BLOCK_M = 64
    BLOCK_N = 64
    scale = 1.0 / math.sqrt(dim)

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch, heads, seq_len, dim), dtype),
        K: T.Tensor((batch, heads, seq_len, dim), dtype),
        V: T.Tensor((batch, heads, seq_len, dim), dtype),
        Out: T.Tensor((batch, heads, seq_len, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, BLOCK_M), heads, batch, threads=128) as (bx, by, bz):
            # Shared Memory
            Q_shared = T.alloc_shared((BLOCK_M, dim), dtype)
            K_shared = T.alloc_shared((BLOCK_N, dim), dtype)
            V_shared = T.alloc_shared((BLOCK_N, dim), dtype)
            P_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)

            # Accumulators (Float32)
            acc_o = T.alloc_fragment((BLOCK_M, dim), "float32")
            acc_z = T.alloc_fragment((BLOCK_M,), "float32")
            
            # Fragments for computation
            S_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            row_sum = T.alloc_fragment((BLOCK_M,), "float32")

            # Initialization
            T.clear(acc_o)
            T.clear(acc_z)

            # Swizzle for performance
            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                P_shared: tilelang.layout.make_swizzled_layout(P_shared),
            })

            # Load Q Tile
            row_start = bx * BLOCK_M
            for i, j in T.Parallel(BLOCK_M, dim):
                r = row_start + i
                if r < seq_len:
                    Q_shared[i, j] = Q[bz, by, r, j]
                else:
                    Q_shared[i, j] = 0.0

            # Loop over K/V blocks (Causal: up to current block)
            loop_range = bx + 1
            
            for k_block in T.Pipelined(loop_range, num_stages=1):
                col_start = k_block * BLOCK_N
                
                # Load K, V Tiles
                for i, j in T.Parallel(BLOCK_N, dim):
                    c = col_start + i
                    if c < seq_len:
                        K_shared[i, j] = K[bz, by, c, j]
                        V_shared[i, j] = V[bz, by, c, j]
                    else:
                        K_shared[i, j] = 0.0
                        V_shared[i, j] = 0.0
                
                # Compute S = Q @ K^T
                T.clear(S_frag)
                T.gemm(Q_shared, K_shared, S_frag, transpose_B=True)
                
                # Compute Polynomial P = 1 + S + 0.5*S^2 and Apply Mask
                # Also accumulate Z
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    r_idx = row_start + i
                    c_idx = col_start + j
                    
                    if r_idx < seq_len and c_idx < seq_len:
                        if c_idx <= r_idx:
                            val_s = S_frag[i, j] * scale
                            val_p = 1.0 + val_s + 0.5 * (val_s * val_s)
                            S_frag[i, j] = val_p
                        else:
                            S_frag[i, j] = 0.0
                    else:
                        S_frag[i, j] = 0.0
                
                # Accumulate Z (Sum of P across columns)
                T.reduce_sum(S_frag, row_sum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    acc_z[i] += row_sum[i]

                # Accumulate O: P @ V
                # Copy P to shared for GEMM
                T.copy(S_frag, P_shared)
                T.gemm(P_shared, V_shared, acc_o)

            # Finalize: O = O / (Z + 1e-6)
            for i, j in T.Parallel(BLOCK_M, dim):
                r = row_start + i
                if r < seq_len:
                    denom = acc_z[i] + 1e-6
                    val = acc_o[i, j] / denom
                    Out[bz, by, r, j] = val

    return tilelang.compile(kernel, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, chunk_size: int = 256):
        super(ModelNew, self).__init__()
        self.chunk_size = chunk_size
        object.__setattr__(self, "_kernel_cache", {})

    def _get_kernel(self, batch, heads, seq_len, dim, dtype):
        key = (batch, heads, seq_len, dim, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_based_linear_attn_kernel(
                batch, heads, seq_len, dim, dtype
            )
        return self._kernel_cache[key]

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        b, h, seq_len, d = q.shape
        
        # Prepare inputs
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Determine dtype for TileLang
        tl_dtype = "bfloat16" if q.dtype == torch.bfloat16 else ("float16" if q.dtype == torch.float16 else "float32")
        
        kernel = self._get_kernel(b, h, seq_len, d, tl_dtype)
        output = kernel(q, k, v)
        
        return output
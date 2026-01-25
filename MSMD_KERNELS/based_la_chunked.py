"""
Entry ID: a4411190028d
Problem: Level 5 Problem 6 - 6_chunked_based
Is Seed: False
Iteration Added: 4
Speedup (Eager): 95.79x
Speedup (Compile): 39.58x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-25T05:33:17.156385
Parents: ['5_3', 'seed_0_148', 'seed_0_109', '5_5', 'seed_0_126', '5_3', '5_5', '5_13', '5_16', '5_11']
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def _build_based_kernel(batch, heads, seq_len, dim):
    scale = dim ** -0.5
    BLOCK_M = 64
    BLOCK_N = 64
    
    @T.prim_func
    def kernel(
        Q: T.Tensor((batch, heads, seq_len, dim), "bfloat16"),
        K: T.Tensor((batch, heads, seq_len, dim), "bfloat16"),
        V: T.Tensor((batch, heads, seq_len, dim), "bfloat16"),
        O: T.Tensor((batch, heads, seq_len, dim), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(seq_len, BLOCK_M), heads, batch, threads=128) as (bx, by, bz):
            # Shared Memory
            Q_s = T.alloc_shared((BLOCK_M, dim), "bfloat16")
            K_s = T.alloc_shared((BLOCK_N, dim), "bfloat16")
            V_s = T.alloc_shared((BLOCK_N, dim), "bfloat16")
            P_s = T.alloc_shared((BLOCK_M, BLOCK_N), "bfloat16")
            
            # Accumulators (Float32 for precision)
            Acc_O = T.alloc_fragment((BLOCK_M, dim), "float32")
            Acc_Z = T.alloc_fragment((BLOCK_M,), "float32")
            
            # Temporary Fragments
            S_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            row_sum = T.alloc_fragment((BLOCK_M,), "float32")
            
            # Layout Optimization
            T.annotate_layout({
                Q_s: tilelang.layout.make_swizzled_layout(Q_s),
                K_s: tilelang.layout.make_swizzled_layout(K_s),
                V_s: tilelang.layout.make_swizzled_layout(V_s),
                P_s: tilelang.layout.make_swizzled_layout(P_s),
            })
            
            # Initialize Accumulators
            T.clear(Acc_O)
            T.clear(Acc_Z)
            
            # Load Q Block (apply scale here)
            base_m = bx * BLOCK_M
            for i, j in T.Parallel(BLOCK_M, dim):
                if base_m + i < seq_len:
                    Q_s[i, j] = Q[bz, by, base_m + i, j] * scale
                else:
                    Q_s[i, j] = 0.0
            
            # Iterate over K/V blocks (Causal Loop)
            loop_range = bx + 1
            for k_idx in T.Pipelined(loop_range, num_stages=1):
                base_n = k_idx * BLOCK_N
                
                # Load K and V
                for i, j in T.Parallel(BLOCK_N, dim):
                    if base_n + i < seq_len:
                        K_s[i, j] = K[bz, by, base_n + i, j]
                        V_s[i, j] = V[bz, by, base_n + i, j]
                    else:
                        K_s[i, j] = 0.0
                        V_s[i, j] = 0.0
                
                # Compute Scores: S = Q @ K.T
                T.clear(S_frag)
                T.gemm(Q_s, K_s, S_frag, transpose_B=True)
                
                # Apply Taylor Expansion + Causal Mask
                # P = 1 + S + 0.5*S^2
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    r = base_m + i
                    c = base_n + j
                    if r < seq_len and c < seq_len:
                        if r >= c:
                            val = S_frag[i, j]
                            p = 1.0 + val + 0.5 * val * val
                            S_frag[i, j] = p
                        else:
                            S_frag[i, j] = 0.0
                    else:
                        S_frag[i, j] = 0.0
                
                # Accumulate Normalizer Z (Sum of P rows)
                T.reduce_sum(S_frag, row_sum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    Acc_Z[i] += row_sum[i]

                # Accumulate Output: O += P @ V
                # Move P back to shared memory for GEMM
                T.copy(S_frag, P_s)
                T.gemm(P_s, V_s, Acc_O)
            
            # Finalize and Store Output
            for i, j in T.Parallel(BLOCK_M, dim):
                if base_m + i < seq_len:
                    norm = Acc_Z[i] + 1e-6
                    val = Acc_O[i, j] / norm
                    O[bz, by, base_m + i, j] = val

    return tilelang.compile(kernel, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, chunk_size: int = 256):
        super(ModelNew, self).__init__()
        self.chunk_size = chunk_size
        object.__setattr__(self, "_kernel_cache", {})

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, H, L, D = q.shape
        
        # Prepare inputs
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Check cache
        key = (B, H, L, D)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_based_kernel(B, H, L, D)
        
        kernel = self._kernel_cache[key]
        
        # Execute kernel
        # Input q is NOT pre-scaled; the kernel applies the scaling.
        # The reference scaled q in forward(), so we pass raw q and let kernel handle it.
        return kernel(q, k, v)
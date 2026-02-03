"""
Entry ID: 3b2bbcb59b19
Problem: Level 5 Problem 48 - 48_rebased_parallel
Is Seed: False
Iteration Added: 4
Speedup (Eager): 6.95x
Speedup (Compile): 10.39x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-29T02:43:19.447865
Parents: []
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def _build_rebased_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    scale: float,
    block_M: int = 64,
    block_N: int = 64,
    dtype: str = "bfloat16",
    accum_dtype: str = "float32",
):
    @T.prim_func
    def rebased_attention(
        Q: T.Tensor((batch, heads, seq_len, dim), dtype),
        K: T.Tensor((batch, heads, seq_len, dim), dtype),
        V: T.Tensor((batch, heads, seq_len, dim), dtype),
        Output: T.Tensor((batch, heads, seq_len, dim), dtype),
    ):
        # Grid: (bx: query block, by: head, bz: batch)
        with T.Kernel(
            T.ceildiv(seq_len, block_M), heads, batch, threads=128
        ) as (bx, by, bz):
            # Shared memory allocations
            Q_shared = T.alloc_shared((block_M, dim), dtype)
            K_shared = T.alloc_shared((block_N, dim), dtype)
            V_shared = T.alloc_shared((block_N, dim), dtype)
            # Intermediate scores in shared memory for GEMM 2 operand
            S_shared = T.alloc_shared((block_M, block_N), dtype)

            # Register accumulators
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)
            acc_z = T.alloc_fragment((block_M,), accum_dtype)
            
            # Helper fragment for reductions
            row_sum = T.alloc_fragment((block_M,), accum_dtype)

            # Layout optimization
            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                S_shared: tilelang.layout.make_swizzled_layout(S_shared),
            })

            # Initialize accumulators
            T.fill(acc_o, 0)
            T.fill(acc_z, 0)

            # Load Q once per block
            for i, j in T.Parallel(block_M, dim):
                row = bx * block_M + i
                if row < seq_len:
                    Q_shared[i, j] = Q[bz, by, row, j]
                else:
                    Q_shared[i, j] = 0

            # Loop over K/V blocks. Since it's causal, we only go up to the current query block.
            loop_range = bx + 1
            for k in T.Pipelined(loop_range, num_stages=2):
                # Load K and V
                for i, j in T.Parallel(block_N, dim):
                    row = k * block_N + i
                    if row < seq_len:
                        K_shared[i, j] = K[bz, by, row, j]
                        V_shared[i, j] = V[bz, by, row, j]
                    else:
                        K_shared[i, j] = 0
                        V_shared[i, j] = 0
                
                # Clear score accumulator for this chunk
                T.clear(acc_s)
                
                # GEMM 1: Q @ K.T
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                # Element-wise operations: Scale, Square, Causal Mask
                # Fix: Use distinct variables to avoid 'already defined' error
                for i, j in T.Parallel(block_M, block_N):
                    global_q_idx = bx * block_M + i
                    global_k_idx = k * block_N + j
                    
                    # Apply scale
                    s_raw = acc_s[i, j]
                    s_scaled = s_raw * scale
                    
                    # Square
                    s_sq = s_scaled * s_scaled
                    
                    # Causal Masking
                    if global_k_idx > global_q_idx or global_k_idx >= seq_len:
                        acc_s[i, j] = 0.0
                    else:
                        acc_s[i, j] = s_sq

                # Accumulate row sums for normalization (Z)
                T.reduce_sum(acc_s, row_sum, dim=1)
                for i in T.Parallel(block_M):
                    acc_z[i] += row_sum[i]

                # Prepare S for GEMM 2: Copy from fp32 fragment to bf16 shared
                T.copy(acc_s, S_shared)
                
                # GEMM 2: S @ V
                T.gemm(S_shared, V_shared, acc_o)

            # Final Normalization and Store
            epsilon = 1e-6
            for i, j in T.Parallel(block_M, dim):
                global_q_idx = bx * block_M + i
                if global_q_idx < seq_len:
                    # Normalize
                    out_val = acc_o[i, j] / (acc_z[i] + epsilon)
                    Output[bz, by, global_q_idx, j] = out_val

    return tilelang.compile(rebased_attention, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # Kernel cache
        object.__setattr__(self, '_kernel_cache', {})

    def _get_kernel(self, batch, heads, seq_len, dim, scale, dtype):
        key = (batch, heads, seq_len, dim, scale, dtype)
        if key not in self._kernel_cache:
            tl_dtype = "bfloat16" if dtype == torch.bfloat16 else "float16"
            if dtype == torch.float32:
                tl_dtype = "float32"
            self._kernel_cache[key] = _build_rebased_kernel(
                batch=batch,
                heads=heads,
                seq_len=seq_len,
                dim=dim,
                scale=scale,
                dtype=tl_dtype
            )
        return self._kernel_cache[key]

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        scale: float = None,
        use_norm: bool = True
    ) -> torch.Tensor:
        if scale is None:
            scale = q.shape[-1] ** -0.5
            
        batch, heads, seq_len, dim = q.shape
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        kernel = self._get_kernel(batch, heads, seq_len, dim, scale, q.dtype)
        
        # Output allocated by runtime implicitly
        out = kernel(q, k, v)
        
        return out
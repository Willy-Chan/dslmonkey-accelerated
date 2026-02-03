"""
Entry ID: de5364ea785e
Problem: Level 22 Problem 6 - 6_retention_parallel
Is Seed: False
Iteration Added: 11
Speedup (Eager): 6.85x
Speedup (Compile): 3.85x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-02-03T04:07:59.013086
Parents: []
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def _build_retention_kernel(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    block_M: int = 64,
    block_N: int = 64,
    dtype: str = "bfloat16",
    accum_dtype: str = "float32",
):
    # Calculate scale constant
    scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def retention_kernel(
        Q: T.Tensor((batch_size, num_heads, seq_len, head_dim), dtype),
        K: T.Tensor((batch_size, num_heads, seq_len, head_dim), dtype),
        V: T.Tensor((batch_size, num_heads, seq_len, head_dim), dtype),
        Output: T.Tensor((batch_size, num_heads, seq_len, head_dim), dtype),
    ):
        # Grid: (query_blocks, heads, batch)
        with T.Kernel(
            T.ceildiv(seq_len, block_M), num_heads, batch_size, threads=128
        ) as (bx, by, bz):
            # Shared memory allocations
            Q_shared = T.alloc_shared((block_M, head_dim), dtype)
            K_shared = T.alloc_shared((block_N, head_dim), dtype)
            # Use accum_dtype for V to maintain higher precision in the second GEMM
            V_shared = T.alloc_shared((block_N, head_dim), accum_dtype)
            
            # Fragment allocations
            acc_o = T.alloc_fragment((block_M, head_dim), accum_dtype)
            scores = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            # Intermediate shared memory buffers
            # scores_shared must match fragment dtype (float32) for T.copy
            scores_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            # Use accum_dtype for p matrix to avoid truncation errors before V multiplication
            p_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            # Calculate head-specific decay slope s in float32
            # s = log2(1 - 2^(-5 - head_idx))
            head_idx_f32 = T.cast(by, "float32")
            exponent = -5.0 - head_idx_f32
            term = 1.0 - T.exp2(exponent)
            s_val = T.log2(term)

            # Initialize accumulator
            T.clear(acc_o)

            # Load Q block
            for i, j in T.Parallel(block_M, head_dim):
                row = bx * block_M + i
                if row < seq_len:
                    Q_shared[i, j] = Q[bz, by, row, j]
                else:
                    Q_shared[i, j] = 0.0
            T.sync_threads()

            # Loop limit for K/V blocks (Causal: only up to current block)
            loop_limit = bx + 1

            # Use standard range loop
            for k_block in range(loop_limit):
                # Load K and V blocks
                for i, j in T.Parallel(block_N, head_dim):
                    row = k_block * block_N + i
                    if row < seq_len:
                        K_shared[i, j] = K[bz, by, row, j]
                        # Cast V to accum_dtype
                        V_shared[i, j] = T.cast(V[bz, by, row, j], accum_dtype)
                    else:
                        K_shared[i, j] = 0.0
                        V_shared[i, j] = 0.0
                
                T.sync_threads()

                # Compute Scores = Q @ K.T
                T.clear(scores)
                # Q (bf16) @ K (bf16) -> scores (fp32)
                T.gemm(Q_shared, K_shared, scores, transpose_B=True)

                # Move scores to shared for element-wise ops
                T.copy(scores, scores_shared)
                
                T.sync_threads()

                # Apply causal mask and decay pattern element-wise on shared memory
                base_i = bx * block_M
                base_j = k_block * block_N

                for i, j in T.Parallel(block_M, block_N):
                    row_global = base_i + i
                    col_global = base_j + j
                    
                    # Check causal mask
                    if row_global >= col_global:
                        # Computation in float32
                        diff = T.cast(row_global - col_global, "float32")
                        decay_exponent = diff * s_val
                        decay = T.exp2(decay_exponent)
                        
                        val_f32 = scores_shared[i, j]
                        scaled_val = val_f32 * scale * decay
                        
                        # Store as float32
                        p_shared[i, j] = scaled_val
                    else:
                        p_shared[i, j] = 0.0

                # Ensure modifications are visible before V multiplication
                T.sync_threads()

                # Accumulate into acc_o using modified scores
                # p (fp32) @ V (fp32) -> acc_o (fp32)
                T.gemm(p_shared, V_shared, acc_o)
                
                T.sync_threads()

            # Store Output
            for i, j in T.Parallel(block_M, head_dim):
                row = bx * block_M + i
                if row < seq_len:
                    Output[bz, by, row, j] = T.cast(acc_o[i, j], dtype)

    return tilelang.compile(retention_kernel, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # Kernel cache
        object.__setattr__(self, '_kernel_cache', {})

    def _get_kernel(self, batch_size, num_heads, seq_len, head_dim):
        key = (batch_size, num_heads, seq_len, head_dim)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_retention_kernel(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
            )
        return self._kernel_cache[key]

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        orig_type = q.dtype
        target_dtype = torch.bfloat16
        
        # Move to target dtype if necessary
        if q.dtype != target_dtype:
            q = q.to(target_dtype)
        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if v.dtype != target_dtype:
            v = v.to(target_dtype)
            
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        batch_size, num_heads, seq_len, head_dim = q.shape
        
        kernel = self._get_kernel(batch_size, num_heads, seq_len, head_dim)
        
        # The kernel writes to a new tensor allocated by tilelang (out_idx=[3])
        out = kernel(q, k, v)
        
        return out.to(orig_type)
"""
Entry ID: a821c13ed7ff
Problem: Level 22 Problem 12 - 12_naive_chunk_based
Is Seed: False
Iteration Added: 1
Speedup (Eager): 98.92x
Speedup (Compile): 41.41x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-02-03T01:37:52.661527
Parents: ['seed_0_2', 'seed_0_1', 'seed_0_3', 'seed_0_1', 'seed_0_3', 'seed_0_2']
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def _build_based_flash_attn_kernel(
    batch: int, 
    heads: int, 
    seq_len: int, 
    dim: int, 
    block_m: int = 64, 
    block_n: int = 64
):
    # Precompute scaling factor d**-0.5
    scale = 1.0 / math.sqrt(dim)

    @T.prim_func
    def based_kernel(
        Q: T.Tensor((batch, heads, seq_len, dim), "bfloat16"),
        K: T.Tensor((batch, heads, seq_len, dim), "bfloat16"),
        V: T.Tensor((batch, heads, seq_len, dim), "bfloat16"),
        Out: T.Tensor((batch, heads, seq_len, dim), "bfloat16")
    ):
        # Grid: (seq_len // block_m, heads, batch)
        with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=128) as (bx, by, bz):
            # Shared Memory Allocations
            Q_s = T.alloc_shared((block_m, dim), "bfloat16")
            K_s = T.alloc_shared((block_n, dim), "bfloat16")
            V_s = T.alloc_shared((block_n, dim), "bfloat16")
            
            # Shared memory for scores to be used as input for the second GEMM
            Scores_s = T.alloc_shared((block_m, block_n), "bfloat16")

            # Fragment Accumulators
            # acc_o: accumulates the weighted sum of values (numerator)
            acc_o = T.alloc_fragment((block_m, dim), "float32")
            # acc_z: accumulates the sum of weights (denominator)
            acc_z = T.alloc_fragment((block_m,), "float32")
            # scores: intermediate storage for Q @ K.T
            scores = T.alloc_fragment((block_m, block_n), "float32")
            # temp_z: temporary storage for row-wise reduction of current block
            temp_z = T.alloc_fragment((block_m,), "float32")

            # Layout optimization
            T.annotate_layout({
                Q_s: tilelang.layout.make_swizzled_layout(Q_s),
                K_s: tilelang.layout.make_swizzled_layout(K_s),
                V_s: tilelang.layout.make_swizzled_layout(V_s),
                Scores_s: tilelang.layout.make_swizzled_layout(Scores_s),
            })

            # Initialize accumulators
            T.clear(acc_o)
            T.clear(acc_z)

            # Load Q block for this iteration
            # We scale Q immediately upon loading or during GEMM? 
            # For simplicity, we load raw Q, then multiply scale during the polynomial computation
            # or just scale Q_s in place. Scaling Q_s in place is cleaner.
            T.copy(Q[bz, by, bx * block_m : (bx + 1) * block_m, :], Q_s)
            
            # Apply scaling to Q_s
            for i, d in T.Parallel(block_m, dim):
                Q_s[i, d] = Q_s[i, d] * scale

            # Pipelined loop over Key/Value blocks
            # We only iterate up to bx (inclusive) for causal attention
            for k_idx in T.Pipelined(bx + 1, num_stages=2):
                # Load K and V blocks
                T.copy(K[bz, by, k_idx * block_n : (k_idx + 1) * block_n, :], K_s)
                T.copy(V[bz, by, k_idx * block_n : (k_idx + 1) * block_n, :], V_s)

                # 1. Compute Scores = Q_s @ K_s.T
                T.clear(scores)
                T.gemm(Q_s, K_s, scores, transpose_B=True)

                # 2. Apply Polynomial Activation and Causal Masking
                # phi(x) = 1 + x + 0.5 * x^2
                for i, j in T.Parallel(block_m, block_n):
                    # Causal logic: valid if (k_idx < bx) OR (k_idx == bx AND j <= i)
                    # Note: Since block_m == block_n == 64, direct comparison j <= i works for diagonal blocks
                    is_causal = (k_idx < bx) or (j <= i)
                    
                    val = scores[i, j]
                    poly = 1.0 + val + 0.5 * val * val
                    
                    # Mask out non-causal entries
                    scores[i, j] = T.if_then_else(is_causal, poly, 0.0)

                # 3. Accumulate Normalizer (Z)
                # Reduce row-wise sum of the activated scores
                T.reduce_sum(scores, temp_z, dim=1)
                for i in T.Parallel(block_m):
                    acc_z[i] += temp_z[i]

                # 4. Accumulate Output (O)
                # We need scores in shared memory to multiply with V_s (scores @ V_s)
                # Cast fp32 scores to bf16 for tensor core GEMM input
                T.copy(scores, Scores_s)
                
                # acc_o += Scores_s @ V_s
                T.gemm(Scores_s, V_s, acc_o)

            # Epilogue: Normalization and Write-back
            for i, d in T.Parallel(block_m, dim):
                # Add epsilon for numerical stability
                acc_o[i, d] = acc_o[i, d] / (acc_z[i] + 1e-6)

            # Store result to global memory
            T.copy(acc_o, Out[bz, by, bx * block_m : (bx + 1) * block_m, :])

    return tilelang.compile(based_kernel, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, chunk_size: int = 256):
        super(ModelNew, self).__init__()
        # chunk_size is kept for compatibility but the kernel determines its own blocking
        self.chunk_size = chunk_size
        # Initialize kernel cache
        object.__setattr__(self, '_kernel_cache', {})

    def _get_kernel(self, b, h, s, d):
        key = (b, h, s, d)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_based_flash_attn_kernel(b, h, s, d)
        return self._kernel_cache[key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous and in correct dtype
        # Shape: [Batch, Heads, SeqLen, Dim]
        if q.dtype != torch.bfloat16:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        b, h, s, d = q.shape
        
        kernel = self._get_kernel(b, h, s, d)
        out = kernel(q, k, v)
        
        return out
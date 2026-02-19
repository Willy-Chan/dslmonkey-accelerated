"""
Entry ID: 16668c832a34
Problem: Level 5 Problem 21 - 21_naive_parallel_based
Is Seed: False
Iteration Added: 12
Speedup (Eager): 70.39x
Speedup (Compile): 33.81x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-26T12:00:08.826844
Parents: ['seed_0_148', 'seed_0_109', '5_25', 'seed_0_126', '5_1', '5_11', '5_12', '5_8', '5_1', '5_25', '5_11', '5_12', '5_8', '5_17', '5_20', 'seed_0_147']
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def _build_based_kernel(batch, heads, seq_len, dim, block_m=64, block_n=64, dtype="bfloat16"):
    """
    Fused Based Linear Attention Kernel.
    Computes: O = (P @ V) / Z
    Where P = CausalMask(1 + S + 0.5 * S^2), S = (Q @ K.T) / sqrt(d)
    Z = rowsum(P)
    """
    
    # Scaling factor for Q (applied mathematically as S = Q * scale @ K.T)
    scale = 1.0 / math.sqrt(dim)

    @T.prim_func
    def based_kernel(
        Q: T.Tensor((batch, heads, seq_len, dim), dtype),
        K: T.Tensor((batch, heads, seq_len, dim), dtype),
        V: T.Tensor((batch, heads, seq_len, dim), dtype),
        Output: T.Tensor((batch, heads, seq_len, dim), dtype),
    ):
        # Grid: (seq_len // block_m, heads, batch)
        with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=128) as (bx, by, bz):
            # Shared memory allocations
            Q_shared = T.alloc_shared((block_m, dim), dtype)
            K_shared = T.alloc_shared((block_n, dim), dtype)
            V_shared = T.alloc_shared((block_n, dim), dtype)
            
            # Accumulators in registers (float32)
            acc_o = T.alloc_fragment((block_m, dim), "float32")
            acc_z = T.alloc_fragment((block_m,), "float32")
            
            # Intermediate scores (registers)
            scores = T.alloc_fragment((block_m, block_n), "float32")
            
            # Initialize accumulators
            T.clear(acc_o)
            T.clear(acc_z)
            
            # Load Q tile
            # Loop over block_m rows, dim cols
            # Parallel copy using 2D slice syntax
            T.copy(Q[bz, by, bx * block_m : (bx + 1) * block_m, :], Q_shared)
            
            # Loop over KV blocks up to the current Q block (causal)
            # We iterate k_idx from 0 to bx (inclusive)
            # Pipelining is possible but we'll stick to a simple loop for clarity given the custom math
            for k_idx in T.Pipelined(bx + 1, num_stages=1):
                # Load K and V tiles
                T.copy(K[bz, by, k_idx * block_n : (k_idx + 1) * block_n, :], K_shared)
                T.copy(V[bz, by, k_idx * block_n : (k_idx + 1) * block_n, :], V_shared)
                
                # 1. Compute S = (Q @ K.T) * scale
                # Initialize scores to 0
                T.clear(scores)
                T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                # 2. Apply Taylor Expansion and Causal Masking
                # P = 1 + S + 0.5 * S^2
                # We do this in a parallel loop over the score tile
                for i, j in T.Parallel(block_m, block_n):
                    # Global indices to check causality
                    row = bx * block_m + i
                    col = k_idx * block_n + j
                    
                    if row >= col:
                        s_val = scores[i, j] * scale
                        # Taylor expansion: 1 + s + 0.5 * s^2
                        p_val = 1.0 + s_val + 0.5 * (s_val * s_val)
                        
                        # Update Score for GEMM 2 (P @ V)
                        scores[i, j] = p_val
                        
                        # Accumulate Z (normalizer)
                        # Note: Z is sum over columns (j)
                        # Since we are inside Parallel(i, j), we can't directly accumulate to acc_z[i] race-free
                        # unless we use atomics or reduce. 
                        # However, with T.GemmWarpPolicy.FullRow, threads in a warp usually handle the same row or we can use atomic_add.
                        # Using atomic_add on register fragment is not standard, usually we accumulate locally then reduce.
                        # BUT, let's look at the structure. T.Parallel maps to threads.
                        # We can use T.atomic_add on acc_z if it's in shared or global, but it's a fragment.
                        # Better strategy: accumulate Z in a separate pass or structure.
                        # Or, simply use atomic_add on a shared memory buffer for Z, then move to register?
                        # Actually, we can use a reduction logic.
                        # Let's use a temporary register for the reduction if possible, or just T.atomic_add if TileLang supports fragment atomics (it usually maps to direct updates).
                        # Given the constraints, let's use a standard pattern: 
                        # We can accumulate Z later? No, we need it now.
                        # Let's perform the math on scores, then separate the reduction.
                    else:
                        scores[i, j] = 0.0
                
                # 3. Accumulate O += P @ V
                # We reuse 'scores' fragment which now holds 'P'.
                # Note: scores is in float32 fragment. We need to cast to appropriate type for GEMM if needed.
                # T.gemm handles fragment inputs.
                
                # Convert scores to shared for GEMM 2 input? 
                # Standard FlashAttn moves Accum -> Shared(Cast) -> GEMM.
                # Here we modified 'scores' in place. 
                # To feed 'scores' into GEMM as matrix A, we usually need it in Shared or Fragment.
                # T.gemm(A, B, C) where A is fragment is supported in some configs, but safe path is Shared.
                scores_shared = T.alloc_shared((block_m, block_n), dtype)
                T.copy(scores, scores_shared)
                
                # Now Accumulate Output
                T.gemm(scores_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                
                # 4. Accumulate Z
                # We can do this by reducing scores_shared.
                # T.reduce_sum(scores_shared, acc_z_tmp, dim=1)
                # Then accumulate to acc_z.
                # We need a temporary buffer for the reduction result of this chunk
                z_chunk = T.alloc_fragment((block_m,), "float32")
                T.reduce_sum(scores, z_chunk, dim=1)
                for r in T.Parallel(block_m):
                    acc_z[r] += z_chunk[r]

            # Finalize: O = O / (Z + epsilon)
            for i, j in T.Parallel(block_m, dim):
                val_z = acc_z[i] + 1e-6
                acc_o[i, j] = acc_o[i, j] / val_z
            
            # Store Output
            T.copy(acc_o, Output[bz, by, bx * block_m : (bx + 1) * block_m, :])

    return tilelang.compile(based_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, chunk_size: int = 256):
        super(ModelNew, self).__init__()
        # chunk_size is kept for compatibility but the kernel determines its own blocking
        self.chunk_size = chunk_size
        self._kernel_cache = {}

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        
        # q, k, v shapes: (b, h, seq_len, d)
        b, h, seq_len, d = q.shape
        
        # Enforce contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Check dtype
        dtype_str = "bfloat16" if q.dtype == torch.bfloat16 else "float16"
        
        key = (b, h, seq_len, d, dtype_str)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_based_kernel(b, h, seq_len, d, dtype=dtype_str)
            
        kernel = self._kernel_cache[key]
        
        return kernel(q, k, v)
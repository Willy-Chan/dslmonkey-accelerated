"""
Entry ID: df47751b140b
Problem: Level 5 Problem 21 - 21_naive_parallel_based
Is Seed: False
Iteration Added: 1
Speedup (Eager): 97.86x
Speedup (Compile): 41.23x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-26T09:53:04.374495
Parents: ['seed_0_2', 'seed_0_3', 'seed_0_1', 'seed_0_2', 'seed_0_1', 'seed_0_3']
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def _build_based_linear_attention_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
):
    dtype = "bfloat16"
    accum_dtype = "float32"
    scale = 1.0 / math.sqrt(dim)

    @T.prim_func
    def based_attention(
        Q: T.Tensor((batch, heads, seq_len, dim), dtype),
        K: T.Tensor((batch, heads, seq_len, dim), dtype),
        V: T.Tensor((batch, heads, seq_len, dim), dtype),
        Output: T.Tensor((batch, heads, seq_len, dim), dtype),
    ):
        # Grid: (batch, heads, num_q_blocks)
        # bx -> num_q_blocks (sequence length chunk)
        # by -> heads
        # bz -> batch
        with T.Kernel(
            T.ceildiv(seq_len, block_M), heads, batch, threads=threads
        ) as (bx, by, bz):
            # Shared memory allocations
            Q_shared = T.alloc_shared((block_M, dim), dtype)
            K_shared = T.alloc_shared((block_N, dim), dtype)
            V_shared = T.alloc_shared((block_N, dim), dtype)
            
            # Buffer for S = Q @ K.T
            # We need to compute S, then P = 1 + S + 0.5 * S^2
            # Then O += P @ V
            # Since we can't easily chain GEMMs in registers without staging, 
            # we might need to stage P to shared if we want to use tensor cores for P @ V,
            # or we perform element-wise ops.
            # Given the polynomial nature, we compute S in fragments, apply poly, store to shared (maybe) or directly use.
            # For P @ V, P is (M, N), V is (N, D). 
            # Since we need to construct P elementwise from S, let's keep S in fragments, transform to P in fragments.
            # However, T.gemm takes shared memory inputs usually for the A matrix in the second GEMM.
            # So we will compute S (frag), transform to P (frag), store P to shared, then P (shared) @ V (shared).
            
            P_shared = T.alloc_shared((block_M, block_N), dtype)
            
            # Accumulators
            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)
            acc_z = T.alloc_fragment((block_M,), accum_dtype)
            
            # Fragments for intermediate calc
            scores = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            # Layout optimizations
            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                P_shared: tilelang.layout.make_swizzled_layout(P_shared),
            })

            # Initialize accumulators
            T.clear(acc_o)
            T.clear(acc_z)
            
            # Load Q
            # Handle boundary for Q loading if seq_len not multiple of block_M
            # But prompt examples usually assume divisible or handled via padding. 
            # We use min/boundary checks implicitly or explicitly.
            # For simplicity in this fused kernel, we assume padding or handle via predicates in copy if needed.
            # Using standard copy with slicing handles bounds if carefully written, 
            # but simple slicing assumes valid memory. Let's use standard copy.
            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)

            # Loop over K, V blocks. Causal: only up to current block index bx.
            # (bx + 1) because the current block bx is included (diagonal).
            loop_range = bx + 1
            
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # Load K, V
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                
                # 1. Compute S = Q @ K.T
                T.clear(scores)
                T.gemm(Q_shared, K_shared, scores, transpose_B=True)
                
                # 2. Apply Polynomial and Causal Mask
                # P = 1 + S + 0.5 * S^2
                # Also need to accumulate Z += row_sum(P)
                for i, j in T.Parallel(block_M, block_N):
                    # Global indices
                    row_idx = bx * block_M + i
                    col_idx = k * block_N + j
                    
                    # Causal masking
                    if col_idx > row_idx:
                        scores[i, j] = 0.0
                    else:
                        val = scores[i, j] * scale
                        # P = 1 + val + 0.5 * val^2
                        p_val = 1.0 + val + 0.5 * val * val
                        scores[i, j] = p_val
                        # Accumulate Z
                        # Note: This is a reduction within the parallel loop, 
                        # which is tricky without atomic or warp reduce.
                        # However, we can update acc_z later or here if we trust the compiler
                        # to handle reduction on registers? No, better to do separate reduction step or atomic.
                        # Since we need P for the next GEMM, let's store P to shared first.
                        
                # Store P to shared for next GEMM
                T.copy(scores, P_shared)
                
                # Update Z accumulator
                # We can do a reduction over P_shared or scores
                # T.reduce_sum is not fully available on fragments in this way inside Pipelined easily without shared.
                # We can manually accumulate.
                # However, to avoid race conditions in T.Parallel if multiple threads update same Z[i],
                # we rely on the fact that T.Parallel(block_M, block_N) maps (i, j) to threads.
                # If we iterate i in outer loop and j in inner, we can sum over j.
                # But T.Parallel flattens.
                # Let's use a separate reduction step.
                # scores is fragment (registers). We can use T.reduce_sum if supported on fragments? 
                # The context says: "Reductions: T.reduce_sum ...". 
                # Let's try T.reduce_sum(scores, acc_z_tmp, dim=1) then acc_z += acc_z_tmp.
                # But `scores` is modified in place to be P.
                
                # Reduce sum of P for Z
                # Since T.gemm uses tensor cores, we prefer to compute O += P @ V.
                # Z accumulation needs row sum.
                # Let's do explicit accumulation for Z using a register array and loop over N.
                # But we are in a pipelined loop. 
                # Let's use a warp reduction or simple serial loop over N per thread? No, slow.
                # T.reduce_sum(scores, tmp_z, dim=1) works on register fragments usually.
                # Let's define a temp fragment for the reduction.
                
                tmp_z = T.alloc_fragment((block_M,), accum_dtype)
                T.reduce_sum(scores, tmp_z, dim=1)
                for i in T.Parallel(block_M):
                    acc_z[i] += tmp_z[i]

                # 3. Compute O += P @ V
                # P is in P_shared, V is in V_shared
                T.gemm(P_shared, V_shared, acc_o)

            # Normalization
            # Output = O / (Z + epsilon)
            epsilon = 1e-6
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] = acc_o[i, j] / (acc_z[i] + epsilon)
            
            # Store Output
            T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return tilelang.compile(based_attention, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, chunk_size: int = 256):
        super(ModelNew, self).__init__()
        self.chunk_size = chunk_size
        object.__setattr__(self, '_kernel_cache', {})

    def _get_kernel(self, batch, heads, seq_len, dim):
        key = (batch, heads, seq_len, dim)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_based_linear_attention_kernel(
                batch, heads, seq_len, dim
            )
        return self._kernel_cache[key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous and BF16
        q = q.contiguous().to(torch.bfloat16)
        k = k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.bfloat16)
        
        batch, heads, seq_len, dim = q.shape
        
        # Kernel expects 4D inputs
        kernel = self._get_kernel(batch, heads, seq_len, dim)
        output = kernel(q, k, v)
        
        return output
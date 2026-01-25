"""
Entry ID: 08865ec42413
Problem: Level 5 Problem 6 - 6_quad_state_expansion
Is Seed: False
Iteration Added: 11
Speedup (Eager): 3.07x
Speedup (Compile): 1.57x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-25T09:38:10.095450
Parents: ['5_8', '5_4', '5_7', '5_5', 'seed_0_109', '5_8', '5_4', '5_5', '5_7', 'seed_0_7']
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def _build_quad_expand_kernel(batch, heads, seq_len, head_dim, chunk_size, dtype="bfloat16"):
    """
    Computes the rank-3 quadratic state expansion K @ K @ V per chunk.
    Output shape: (B, H, N_chunks, D, D, D)
    Grid parallelizes over (B, H, N_chunks, D).
    Each block computes one (D, D) slice for a fixed d.
    """
    n_chunks = seq_len // chunk_size
    total_chunks = batch * heads * n_chunks
    
    # Each block computes a (D, D) matrix for a specific 'd' index of the output
    # Total tasks = total_chunks * head_dim
    
    BLOCK_C = 64 # Tiling along the chunk_size dimension (reduction dimension for this kernel)
    BLOCK_D = 64 # head_dim
    
    @T.prim_func
    def kernel_func(
        K: T.Tensor((batch, heads, n_chunks, chunk_size, head_dim), dtype),
        V: T.Tensor((batch, heads, n_chunks, chunk_size, head_dim), dtype),
        Out: T.Tensor((batch, heads, n_chunks, head_dim, head_dim, head_dim), dtype)
    ):
        # Grid layout: 1D grid covering all (chunk, d) pairs
        # total_blocks = total_chunks * head_dim
        # We can use a 2D grid to avoid overflow if needed, but 4*8*2*64 = 4096 is small.
        
        with T.Kernel(head_dim, total_chunks, threads=128) as (bx, by):
            # bx: d index (0..63)
            # by: chunk index flattened (b, h, n)
            
            d_idx = bx
            
            # Decode 'by' into (b, h, n)
            n_idx = by % n_chunks
            temp = by // n_chunks
            h_idx = temp % heads
            b_idx = temp // heads
            
            # Allocations
            # K_shared: Stores a tile of K (BLOCK_C rows, D cols)
            # V_shared: Stores a tile of V (BLOCK_C rows, D cols)
            # K_scaled_shared: Stores K scaled by K[:, d] (BLOCK_C rows, D cols)
            # Acc: Accumulator for the result (D, D)
            
            K_shared = T.alloc_shared((BLOCK_C, BLOCK_D), dtype)
            V_shared = T.alloc_shared((BLOCK_C, BLOCK_D), dtype)
            K_scaled_shared = T.alloc_shared((BLOCK_C, BLOCK_D), dtype)
            
            Acc = T.alloc_fragment((BLOCK_D, BLOCK_D), "float32")
            
            T.annotate_layout({
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                K_scaled_shared: tilelang.layout.make_swizzled_layout(K_scaled_shared),
            })
            
            T.clear(Acc)
            
            # Loop over chunk_size in tiles
            num_stages = chunk_size // BLOCK_C
            
            for k_iter in T.Pipelined(num_stages, num_stages=2):
                c_offset = k_iter * BLOCK_C
                
                # Load K and V tiles
                # K[b, h, n, c_offset:..., :]
                # T.Parallel for efficient copy
                for i, j in T.Parallel(BLOCK_C, BLOCK_D):
                    K_shared[i, j] = K[b_idx, h_idx, n_idx, c_offset + i, j]
                    V_shared[i, j] = V[b_idx, h_idx, n_idx, c_offset + i, j]
                
                # Compute Scaled K: K_scaled[i, j] = K[i, j] * K[i, d_idx]
                # We need to synchronize before reading K_shared because it's populated in the same stage
                # But T.Pipelined implies stage separation for data dependency if we structure carefully.
                # However, here we load then compute in the same stage. Explicit sync needed within the stage logic?
                # T.Pipelined mainly handles the asynchronous copy. The compute part happens after wait.
                # In TileLang, operations after copy are compute. TileLang handles barriers for shared mem.
                
                # Scale K
                # Note: We need K[i, d_idx] which is column 'd_idx' of the current tile.
                # Broadcasting: for each row i, we take the scalar at column d_idx.
                for i, j in T.Parallel(BLOCK_C, BLOCK_D):
                    # Cast to float32 for multiplication to avoid overflow if needed, but bf16 is fine
                    # K_scaled_shared is bf16
                    scalar = K_shared[i, d_idx]
                    val = K_shared[i, j]
                    K_scaled_shared[i, j] = val * scalar
                
                # GEMM
                # We want sum_c (K_scaled[c, e] * V[c, f])
                # K_scaled is (BLOCK_C, D), V is (BLOCK_C, D)
                # We need (D, D) output. transpose_A=True makes it (D, BLOCK_C) @ (BLOCK_C, D) -> (D, D)
                T.gemm(K_scaled_shared, V_shared, Acc, transpose_A=True)
            
            # Store result
            # Out[b, h, n, d, :, :] = Acc
            for i, j in T.Parallel(BLOCK_D, BLOCK_D):
                Out[b_idx, h_idx, n_idx, d_idx, i, j] = Acc[i, j]

    return tilelang.compile(kernel_func, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, chunk_size: int):
        super(ModelNew, self).__init__()
        self.chunk_size = chunk_size
        object.__setattr__(self, '_kernel_cache', {})

    def _get_kernel(self, b, h, s, d, dtype):
        key = (b, h, s, d, self.chunk_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_quad_expand_kernel(
                b, h, s, d, self.chunk_size, dtype=dtype
            )
        return self._kernel_cache[key]

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, h, seq_len, d = k.shape
        n_chunks = seq_len // self.chunk_size
        
        # Prepare inputs
        target_dtype = torch.bfloat16
        if k.dtype != target_dtype: k = k.to(target_dtype)
        if v.dtype != target_dtype: v = v.to(target_dtype)
        k = k.contiguous()
        v = v.contiguous()
        
        # View as chunks for the kernel
        k_chunk = k.view(b, h, n_chunks, self.chunk_size, d)
        v_chunk = v.view(b, h, n_chunks, self.chunk_size, d)
        
        kernel = self._get_kernel(b, h, seq_len, d, "bfloat16")
        
        # Kernel returns (B, H, N, D, D, D)
        # We pass inputs as chunks
        output = kernel(k_chunk, v_chunk)
        
        return output
"""
Entry ID: 9f3f1fe04222
Problem: Level 5 Problem 4 - 4_linear_state_prep
Is Seed: False
Iteration Added: 20
Speedup (Eager): 5.04x
Speedup (Compile): 5.89x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-25T10:16:38.412306
Parents: ['5_6', '5_7', '5_5', '5_8', 'seed_0_109', '5_7', '5_6', '5_5', '5_8', 'seed_0_7']
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def _build_linear_state_kernel(batch, heads, seq_len, head_dim, chunk_size, dtype="bfloat16"):
    """
    Computes the shifted cumulative sum of K^T @ V for each chunk.
    Input: K, V (B, H, Seq, D)
    Output: State (B, H, N_chunks, D, D)
    
    Algorithm:
    Acc = 0
    For n in 0..N_chunks-1:
        Output[b, h, n] = Acc
        Acc += K[n]^T @ V[n]
    """
    n_chunks = seq_len // chunk_size
    total_seqs = batch * heads
    
    # Dimensions
    # D = head_dim = 64
    # C = chunk_size = 256
    
    BLOCK_C = 32 # Tiling along the chunk dimension (inner loop)
    
    @T.prim_func
    def kernel_func(
        K: T.Tensor((batch, heads, seq_len, head_dim), dtype),
        V: T.Tensor((batch, heads, seq_len, head_dim), dtype),
        Out: T.Tensor((batch, heads, n_chunks, head_dim, head_dim), dtype),
    ):
        # Grid: One block per sequence (Batch * Head)
        # This ensures we can maintain the state 'acc' in registers across chunks
        with T.Kernel(total_seqs, threads=128) as bx:
            # Decode indices
            h_idx = bx % heads
            b_idx = bx // heads
            
            # Accumulator for the running state (D, D)
            # D=64, so 64x64 accumulator
            acc = T.alloc_fragment((head_dim, head_dim), "float32")
            T.clear(acc)
            
            # Shared memory for K and V tiles
            # Shape: (BLOCK_C, D)
            shm_K = T.alloc_shared((BLOCK_C, head_dim), dtype)
            shm_V = T.alloc_shared((BLOCK_C, head_dim), dtype)
            
            # Apply swizzling to avoid bank conflicts
            T.annotate_layout({
                shm_K: tilelang.layout.make_swizzled_layout(shm_K),
                shm_V: tilelang.layout.make_swizzled_layout(shm_V),
            })
            
            # Iterate over chunks sequentially to perform cumsum
            for n in T.serial(n_chunks):
                # 1. Write current accumulator to global memory (Shifted output)
                # This corresponds to the state BEFORE adding the current chunk
                for i, j in T.Parallel(head_dim, head_dim):
                    Out[b_idx, h_idx, n, i, j] = acc[i, j]
                
                # 2. Accumulate current chunk's K^T @ V into acc
                # We break the chunk_size into smaller tiles to fit in shared memory
                num_steps = T.ceildiv(chunk_size, BLOCK_C)
                
                for k in T.Pipelined(num_steps, num_stages=2):
                    # Global base offset for the current sub-block within the chunk
                    # global_row = n * chunk_size + k * BLOCK_C
                    base_row = n * chunk_size + k * BLOCK_C
                    
                    # Load K tile
                    for ii, jj in T.Parallel(BLOCK_C, head_dim):
                        row = base_row + ii
                        if row < seq_len:
                            shm_K[ii, jj] = K[b_idx, h_idx, row, jj]
                        else:
                            shm_K[ii, jj] = 0.0
                            
                    # Load V tile
                    for ii, jj in T.Parallel(BLOCK_C, head_dim):
                        row = base_row + ii
                        if row < seq_len:
                            shm_V[ii, jj] = V[b_idx, h_idx, row, jj]
                        else:
                            shm_V[ii, jj] = 0.0
                            
                    # GEMM: acc += K.T @ V
                    # K_tile: (BLOCK_C, D)
                    # V_tile: (BLOCK_C, D)
                    # Result: (D, D)
                    # Transpose A (K), do not transpose B (V)
                    T.gemm(shm_K, shm_V, acc, transpose_A=True, transpose_B=False)
                    
    return tilelang.compile(kernel_func, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, chunk_size: int):
        super(ModelNew, self).__init__()
        self.chunk_size = chunk_size
        object.__setattr__(self, '_kernel_cache', {})

    def _get_kernel(self, b, h, s, d, dtype):
        key = (b, h, s, d, self.chunk_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_state_kernel(
                b, h, s, d, self.chunk_size, dtype=dtype
            )
        return self._kernel_cache[key]

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, h, seq_len, d = k.shape
        
        # Ensure correct dtype and contiguous memory
        target_dtype = torch.bfloat16
        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if v.dtype != target_dtype:
            v = v.to(target_dtype)
            
        k = k.contiguous()
        v = v.contiguous()
        
        kernel = self._get_kernel(b, h, seq_len, d, "bfloat16")
        
        # The kernel allocates the output tensor via out_idx=[2]
        # We pass only inputs k and v
        output = kernel(k, v)
        
        return output
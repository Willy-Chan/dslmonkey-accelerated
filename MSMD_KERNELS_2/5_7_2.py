"""
Entry ID: 529a1988f318
Problem: Level 5 Problem 7 - 7_quad_state_accum
Is Seed: False
Iteration Added: 4
Speedup (Eager): 4.12x
Speedup (Compile): 5.41x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-25T09:00:43.553088
Parents: ['5_4', '5_5', 'seed_0_109', 'seed_0_126', 'seed_0_148', '5_4', '5_5', 'seed_0_7', 'seed_0_121', 'seed_0_147']
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def _build_quad_cumsum_kernel(batch, heads, n_chunks, dim, dtype="bfloat16"):
    """
    Computes shifted cumsum for (B, H, N, D, D, D).
    Flatten (D, D, D) -> M.
    Grid: (M_tiles, B*H)
    Loop over N sequentially.
    """
    M = dim * dim * dim
    # Tiling configuration
    BLOCK_M = 256  # Elements per block
    accum_dtype = "float32"
    
    @T.prim_func
    def kernel_func(
        In: T.Tensor((batch, heads, n_chunks, M), dtype),
        Out: T.Tensor((batch, heads, n_chunks, M), dtype)
    ):
        # Grid layout:
        # bx: Tile index for flattened feature dimension M
        # by: Combined Batch * Heads index
        grid_m = T.ceildiv(M, BLOCK_M)
        
        with T.Kernel(grid_m, batch * heads, threads=128) as (bx, by):
            # Decode batch/head
            h_idx = by % heads
            b_idx = by // heads
            
            # Base offset in M
            m_base = bx * BLOCK_M
            
            # Accumulator in registers (float32)
            # Each thread handles a portion of BLOCK_M. 
            # With 128 threads and BLOCK_M=256, each thread handles 2 elements.
            # We use T.Parallel logic implicitly or explicitly via vectorized ops.
            acc = T.alloc_fragment((BLOCK_M,), accum_dtype)
            input_frag = T.alloc_fragment((BLOCK_M,), dtype)
            
            # Initialize accumulator to 0
            T.clear(acc)
            
            # Iterate sequentially over chunks
            for n in T.serial(n_chunks):
                # 1. Store current accumulator (sum of 0..n-1) to Output at n
                # Use T.Parallel to distribute the vector store among threads
                for i in T.Parallel(BLOCK_M):
                    m_idx = m_base + i
                    if m_idx < M:
                        Out[b_idx, h_idx, n, m_idx] = acc[i]
                        
                # 2. Load Input at n and accumulate
                for i in T.Parallel(BLOCK_M):
                    m_idx = m_base + i
                    if m_idx < M:
                        val = In[b_idx, h_idx, n, m_idx]
                        acc[i] += val
                        
    return tilelang.compile(kernel_func, out_idx=[1], target="cuda")

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        object.__setattr__(self, '_kernel_cache', {})

    def _get_kernel(self, b, h, n, d, dtype):
        key = (b, h, n, d, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_quad_cumsum_kernel(
                b, h, n, d, dtype=dtype
            )
        return self._kernel_cache[key]

    def forward(self, kv_quad_raw: torch.Tensor) -> torch.Tensor:
        b, h, n, d1, d2, d3 = kv_quad_raw.shape
        
        # Check/Convert dtype
        target_dtype = torch.bfloat16
        if kv_quad_raw.dtype != target_dtype:
            kv_quad_raw = kv_quad_raw.to(target_dtype)
            
        kv_quad_raw = kv_quad_raw.contiguous()
        
        # Reinterpret as (B, H, N, M) where M = D*D*D for the kernel
        # This view doesn't change data, just shape metadata for the kernel factory
        # But the kernel expects 4D tensor. We can pass the 6D tensor if the kernel signature matches,
        # or view it here. The simplest is to view it as flattened for the kernel call.
        M_flat = d1 * d2 * d3
        in_view = kv_quad_raw.view(b, h, n, M_flat)
        
        # We need to allocate output manually if we want to reshape it back easily, 
        # or rely on TileLang out_idx. 
        # TileLang out_idx allocates based on signature. 
        # Let's let TileLang allocate (B, H, N, M) and we reshape return.
        
        kernel = self._get_kernel(b, h, n, d1, "bfloat16")
        out_view = kernel(in_view)
        
        # Reshape back to (B, H, N, D, D, D)
        return out_view.view(b, h, n, d1, d2, d3)
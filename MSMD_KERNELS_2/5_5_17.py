"""
Entry ID: fc42c87f0d77
Problem: Level 5 Problem 5 - 5_linear_state_proj
Is Seed: False
Iteration Added: 18
Speedup (Eager): 9.75x
Speedup (Compile): 14.23x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-25T10:09:05.492134
Parents: ['5_8', '5_4', 'seed_0_109', '5_6', 'seed_0_148', '5_8', '5_6', '5_4', '5_7', 'seed_0_147']
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def _build_linear_proj_kernel(batch, heads, seq_len, head_dim, chunk_size, dtype="bfloat16"):
    """
    Computes O = Q * S where Q is scaled by head_dim**-0.5.
    Input Q: (B, H, Seq, D)
    Input S: (B, H, N_chunks, D, D)
    Output: (B, H, Seq, D)
    
    Maps (batch, heads, n_chunk) to grid_y.
    Maps (chunk_row_tile) to grid_x.
    """
    n_chunks = seq_len // chunk_size
    total_chunks = batch * heads * n_chunks
    
    BLOCK_M = 64
    BLOCK_N = 64  # head_dim
    BLOCK_K = 64  # head_dim
    
    scale = head_dim ** -0.5
    
    @T.prim_func
    def kernel_func(
        Q: T.Tensor((batch, heads, seq_len, head_dim), dtype),
        S: T.Tensor((batch, heads, n_chunks, head_dim, head_dim), dtype),
        Out: T.Tensor((batch, heads, seq_len, head_dim), dtype)
    ):
        # grid_x: tiles along the chunk_size dimension (M)
        # grid_y: flattening of (batch, head, chunk_idx)
        grid_x = T.ceildiv(chunk_size, BLOCK_M)
        
        with T.Kernel(grid_x, total_chunks, threads=128) as (bx, by):
            # Decode indices
            n_idx = by % n_chunks
            temp = by // n_chunks
            h_idx = temp % heads
            b_idx = temp // heads
            
            # Shared Memory Allocations
            # Q tile: (BLOCK_M, D)
            Q_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            # S matrix: (D, D) - acting as the 'B' matrix in GEMM
            S_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            # Accumulator
            Acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            
            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                S_shared: tilelang.layout.make_swizzled_layout(S_shared),
            })
            
            # Load S matrix (whole DxD matrix fits in shared)
            # S shape is (B, H, N, D, D)
            for i, j in T.Parallel(BLOCK_K, BLOCK_N):
                S_shared[i, j] = S[b_idx, h_idx, n_idx, i, j]
            
            # Load Q tile
            # Q shape is (B, H, Seq, D)
            # Base row in sequence for this chunk and block
            base_row = n_idx * chunk_size + bx * BLOCK_M
            
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                row = base_row + i
                if row < seq_len:
                    Q_shared[i, j] = Q[b_idx, h_idx, row, j] * scale
                else:
                    Q_shared[i, j] = 0.0
            
            # Compute GEMM
            # O = Q @ S
            # Q: (BLOCK_M, D), S: (D, D) -> (BLOCK_M, D)
            T.clear(Acc)
            T.gemm(Q_shared, S_shared, Acc)
            
            # Store Output
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                row = base_row + i
                if row < seq_len:
                    Out[b_idx, h_idx, row, j] = Acc[i, j]

    return tilelang.compile(kernel_func, out_idx=[2], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, chunk_size: int, head_dim: int):
        super(ModelNew, self).__init__()
        self.chunk_size = chunk_size
        self.head_dim = head_dim
        object.__setattr__(self, '_kernel_cache', {})

    def _get_kernel(self, b, h, s, d, dtype):
        key = (b, h, s, d, self.chunk_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_proj_kernel(
                b, h, s, d, self.chunk_size, dtype=dtype
            )
        return self._kernel_cache[key]

    def forward(self, q: torch.Tensor, kv_lin_state: torch.Tensor) -> torch.Tensor:
        b, h, seq_len, d = q.shape
        
        # Prepare inputs
        target_dtype = torch.bfloat16
        if q.dtype != target_dtype:
            q = q.to(target_dtype)
        if kv_lin_state.dtype != target_dtype:
            kv_lin_state = kv_lin_state.to(target_dtype)
            
        q = q.contiguous()
        kv_lin_state = kv_lin_state.contiguous()
        
        kernel = self._get_kernel(b, h, seq_len, d, "bfloat16")
        
        # Invoke kernel
        output = kernel(q, kv_lin_state)
        
        return output
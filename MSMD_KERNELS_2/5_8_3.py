"""
Entry ID: cf1a4d60b97e
Problem: Level 5 Problem 8 - 8_quad_state_proj
Is Seed: False
Iteration Added: 8
Speedup (Eager): 5.00x
Speedup (Compile): 5.58x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-01-25T09:24:30.010605
Parents: ['5_6', '5_5', '5_7', '5_4', 'seed_0_109', '5_6', '5_5', '5_7', '5_4', 'seed_0_7']
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def _build_quad_project_kernel(batch, heads, seq_len, head_dim, chunk_size, dtype="bfloat16"):
    """
    Computes O = 0.5 * einsum('bhndef,bhncd,bhnce->bhncf', S, Q, Q)
    Decomposition:
      O_cf = 0.5 * sum_e ( Q_ce * (sum_d Q_cd * S_def) )
    
    Algorithm (per block of C rows):
      Load Q_tile (BLOCK_C, D)
      Accumulator Acc = 0
      For e in 0..D-1:
         Load S_slice_e (D, D) corresponding to S[:, e, :]
         Temp = Q_tile @ S_slice_e  (BLOCK_C, D)
         Acc += Temp * Q_tile[:, e].broadcast
      Store Acc
    """
    n_chunks = seq_len // chunk_size
    total_chunks = batch * heads * n_chunks
    
    # Tiling constants
    BLOCK_C = 64
    D = head_dim  # 64
    
    # Scaling factors
    q_scale = head_dim ** -0.5
    final_scale = 0.5

    @T.prim_func
    def kernel_func(
        Q: T.Tensor((batch, heads, seq_len, head_dim), dtype),
        S: T.Tensor((batch, heads, n_chunks, head_dim, head_dim, head_dim), dtype),
        Out: T.Tensor((batch, heads, seq_len, head_dim), dtype)
    ):
        # Grid: (batch * heads * n_chunks, chunk_size // BLOCK_C)
        grid_c = T.ceildiv(chunk_size, BLOCK_C)
        
        with T.Kernel(total_chunks, grid_c, threads=128) as (by, bx):
            # by: global chunk index
            # bx: block index within chunk (for C dimension)
            
            # Decode indices
            n_idx = by % n_chunks
            tmp = by // n_chunks
            h_idx = tmp % heads
            b_idx = tmp // heads
            
            # Base offset for this chunk's query
            chunk_start_row = n_idx * chunk_size
            # Offset for this block
            row_offset = chunk_start_row + bx * BLOCK_C
            
            # Allocations
            # Shared memory for Q tile: (BLOCK_C, D)
            shm_Q = T.alloc_shared((BLOCK_C, D), dtype)
            # Shared memory for S slice: (D, D)
            # We use double buffering for S loads inside the loop
            shm_S = T.alloc_shared((D, D), dtype)
            
            # Accumulator in registers
            acc = T.alloc_fragment((BLOCK_C, D), "float32")
            # Fragment for GEMM result
            gemm_res = T.alloc_fragment((BLOCK_C, D), "float32")
            
            T.annotate_layout({
                shm_Q: tilelang.layout.make_swizzled_layout(shm_Q),
                shm_S: tilelang.layout.make_swizzled_layout(shm_S),
            })

            # Initialize accumulator
            T.clear(acc)
            
            # 1. Load Q tile for this block
            # Q shape: (B, H, Seq, D)
            for i, j in T.Parallel(BLOCK_C, D):
                r = row_offset + i
                if r < seq_len:
                    val = Q[b_idx, h_idx, r, j]
                    shm_Q[i, j] = val * q_scale
                else:
                    shm_Q[i, j] = 0.0
            
            # 2. Loop over e (0 to D-1)
            # We pipeline the loading of S slices
            for e in T.Pipelined(D, num_stages=2):
                # Load S slice S[:, e, :] -> (D, D)
                # S indices: [b, h, n, d, e, f]
                # We want to load for fixed 'e', iterating d (rows) and f (cols)
                # d is dim 3, e is dim 4, f is dim 5
                for d_local, f_local in T.Parallel(D, D):
                    # S is [b, h, n, d_local, e, f_local]
                    shm_S[d_local, f_local] = S[b_idx, h_idx, n_idx, d_local, e, f_local]
                
                # Compute GEMM: Temp = Q_tile @ S_slice
                # shm_Q: (BLOCK_C, D), shm_S: (D, D) -> gemm_res: (BLOCK_C, D)
                T.clear(gemm_res)
                T.gemm(shm_Q, shm_S, gemm_res)
                
                # Scale and Accumulate: Acc += gemm_res * Q_tile[:, e]
                # Q_tile[:, e] is a column vector from shm_Q
                for i, j in T.Parallel(BLOCK_C, D):
                    # Get the scalar q_val = Q_tile[i, e]
                    # We access shared memory randomly here, might benefit from broadcast or reg shuffle
                    # but shm access is fast enough.
                    q_val = shm_Q[i, e]
                    acc[i, j] += gemm_res[i, j] * T.cast(q_val, "float32")

            # 3. Store result
            for i, j in T.Parallel(BLOCK_C, D):
                r = row_offset + i
                if r < seq_len:
                    val = acc[i, j] * final_scale
                    Out[b_idx, h_idx, r, j] = val

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
            self._kernel_cache[key] = _build_quad_project_kernel(
                b, h, s, d, self.chunk_size, dtype=dtype
            )
        return self._kernel_cache[key]

    def forward(self, q: torch.Tensor, kv_quad_state: torch.Tensor) -> torch.Tensor:
        b, h, seq_len, d = q.shape
        
        # Type conversion
        target_dtype = torch.bfloat16
        if q.dtype != target_dtype:
            q = q.to(target_dtype)
        if kv_quad_state.dtype != target_dtype:
            kv_quad_state = kv_quad_state.to(target_dtype)
            
        q = q.contiguous()
        kv_quad_state = kv_quad_state.contiguous()
        
        kernel = self._get_kernel(b, h, seq_len, d, "bfloat16")
        
        # Invoke kernel
        # Output allocated by runtime due to out_idx=[2]
        o_quad = kernel(q, kv_quad_state)
        
        return o_quad
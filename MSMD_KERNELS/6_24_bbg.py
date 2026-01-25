"""
Entry ID: manual_combined
Problem: Level 6 Problem 24 - 24_naive_retention
Is Seed: False
Iteration Added: manual
Speedup (Eager): TBD
Speedup (Compile): TBD
Model: manual_combination
Timestamp: 2026-01-24
Parents: ['6_20_1', '6_21_2', '6_22_4', '6_23_1']
"""

import torch
import torch.nn as nn
import math
import tilelang
import tilelang.language as T


def _compute_decay_mask_torch(num_heads, seq_len, device, dtype):
    """
    Compute decay mask using PyTorch.
    D[h, i, j] = gamma_h^(i-j) if i >= j else 0
    where gamma_h = 1 - 2^(-5-h)
    """
    # Compute log2_gamma for each head
    head_indices = torch.arange(num_heads, device=device, dtype=torch.float32)
    exponents = -5.0 - head_indices
    powers = torch.pow(2.0, exponents)
    gamma = 1.0 - powers
    log2_gamma = torch.log2(gamma)  # Shape: (num_heads,)
    
    # Create position indices
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    row_indices = positions.unsqueeze(1)  # (seq_len, 1)
    col_indices = positions.unsqueeze(0)  # (1, seq_len)
    pos_diff = row_indices - col_indices  # (seq_len, seq_len)
    
    # Causal mask
    causal_mask = (row_indices >= col_indices).float()  # (seq_len, seq_len)
    
    # Compute decay: 2^((i-j) * log2_gamma)
    # log2_gamma: (num_heads,) -> (num_heads, 1, 1)
    log2_gamma_expanded = log2_gamma.view(num_heads, 1, 1)
    decay_exponent = pos_diff.unsqueeze(0) * log2_gamma_expanded  # (num_heads, seq_len, seq_len)
    decay_values = torch.pow(2.0, decay_exponent)
    
    # Apply causal mask
    decay_mask = decay_values * causal_mask.unsqueeze(0)  # (num_heads, seq_len, seq_len)
    
    return decay_mask.to(dtype)


def _build_attention_scores_kernel(batch_size, num_heads, seq_len, head_dim, dtype):
    """
    Kernel 2: Compute S = (Q @ K^T / sqrt(d)) * D
    """
    block_M = 64
    block_N = 64
    block_K = 32
    
    scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def attn_scores_kernel(
        Q: T.Tensor((batch_size, num_heads, seq_len, head_dim), dtype),
        K_in: T.Tensor((batch_size, num_heads, seq_len, head_dim), dtype),
        D: T.Tensor((num_heads, seq_len, seq_len), dtype),
        S_out: T.Tensor((batch_size, num_heads, seq_len, seq_len), dtype),
    ):
        grid_x = T.ceildiv(seq_len, block_N)
        grid_y = T.ceildiv(seq_len, block_M)
        grid_z = batch_size * num_heads
        
        with T.Kernel(grid_x, grid_y, grid_z, threads=128) as (bx, by, bz):
            batch_idx = bz // num_heads
            head_idx = bz % num_heads
            
            acc = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(acc)
            
            Q_shared = T.alloc_shared((block_M, block_K), dtype)
            K_shared = T.alloc_shared((block_N, block_K), dtype)
            D_shared = T.alloc_shared((block_M, block_N), dtype)
            
            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
            })
            
            m_start = by * block_M
            n_start = bx * block_N
            
            for k_iter in T.Pipelined(T.ceildiv(head_dim, block_K), num_stages=2):
                k_start = k_iter * block_K
                T.copy(Q[batch_idx, head_idx, m_start : m_start + block_M, k_start : k_start + block_K], Q_shared)
                T.copy(K_in[batch_idx, head_idx, n_start : n_start + block_N, k_start : k_start + block_K], K_shared)
                T.gemm(Q_shared, K_shared, acc, transpose_B=True)
            
            # Scale
            for i, j in T.Parallel(block_M, block_N):
                acc[i, j] = acc[i, j] * T.float32(scale)
            
            # Load and apply decay mask
            T.copy(D[head_idx, m_start : m_start + block_M, n_start : n_start + block_N], D_shared)
            
            for i, j in T.Parallel(block_M, block_N):
                mask_val = T.cast(D_shared[i, j], dtype)
                acc[i, j] = acc[i, j] * mask_val
            
            T.copy(acc, S_out[batch_idx, head_idx, m_start : m_start + block_M, n_start : n_start + block_N])
    
    return tilelang.compile(attn_scores_kernel, out_idx=[3], target="cuda")


def _build_output_kernel(batch_size, num_heads, seq_len, head_dim, dtype):
    """
    Kernel 2: Compute O = S @ V
    """
    block_M = 64
    block_N = 64
    block_K = 32
    

    @T.prim_func
    def output_kernel(
        S: T.Tensor((batch_size, num_heads, seq_len, seq_len), dtype),
        V: T.Tensor((batch_size, num_heads, seq_len, head_dim), dtype),
        O: T.Tensor((batch_size, num_heads, seq_len, head_dim), dtype),
    ):
        grid_x = T.ceildiv(head_dim, block_N)
        grid_y = T.ceildiv(seq_len, block_M)
        grid_z = batch_size * num_heads
        
        with T.Kernel(grid_x, grid_y, grid_z, threads=128) as (bx, by, bz):
            batch_idx = bz // num_heads
            head_idx = bz % num_heads
            
            acc = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(acc)
            
            S_shared = T.alloc_shared((block_M, block_K), dtype)
            V_shared = T.alloc_shared((block_K, block_N), dtype)
            
            T.annotate_layout({
                S_shared: tilelang.layout.make_swizzled_layout(S_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
            })
            
            m_start = by * block_M
            n_start = bx * block_N
            
            for k_iter in T.Pipelined(T.ceildiv(seq_len, block_K), num_stages=2):
                k_start = k_iter * block_K
                T.copy(S[batch_idx, head_idx, m_start : m_start + block_M, k_start : k_start + block_K], S_shared)
                T.copy(V[batch_idx, head_idx, k_start : k_start + block_K, n_start : n_start + block_N], V_shared)
                T.gemm(S_shared, V_shared, acc)
            
            T.copy(acc, O[batch_idx, head_idx, m_start : m_start + block_M, n_start : n_start + block_N])
    
    return tilelang.compile(output_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape
        in_dtype = q.dtype
        
        # Map torch dtype to tilelang dtype string
        dtype_map = {
            torch.float32: "float32",
            torch.float16: "float16", 
            torch.bfloat16: "bfloat16",
        }
        tl_dtype = dtype_map.get(in_dtype, "float32")
        
        # Ensure contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Cache key includes dtype
        key = (batch_size, num_heads, seq_len, head_dim, tl_dtype)
        
        if key not in self._kernel_cache:
            self._kernel_cache[key] = {
                'attn_scores': _build_attention_scores_kernel(
                    batch_size, num_heads, seq_len, head_dim, tl_dtype
                ),
                'output': _build_output_kernel(
                    batch_size, num_heads, seq_len, head_dim, tl_dtype
                ),
            }
        
        kernels = self._kernel_cache[key]
        
        # Compute decay mask using PyTorch (fast enough, avoids TVM issues)
        D = _compute_decay_mask_torch(num_heads, seq_len, q.device, q.dtype)
        
        # Kernel 1: S = (Q @ K^T / sqrt(d)) * D
        S = kernels['attn_scores'](q, k, D)
        
        # Kernel 2: O = S @ V
        output = kernels['output'](S, v)
        
        return output

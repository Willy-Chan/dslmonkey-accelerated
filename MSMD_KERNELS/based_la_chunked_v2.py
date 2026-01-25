"""
Hybrid Chunked Based Linear Attention with separate feature_dim and head_dim.

Supports Q/K with feature_dim (e.g., 16) and V with head_dim (e.g., 128).

Combines PyTorch for some operations with optimized TileLang kernels for:
1. Normalizer computation (from 5_0_4.py)
2. Inter-chunk linear term (from 5_3_5.py)  
3. Final combine + normalize (from 5_5_2.py)

THIS IS A VERY UNFUSED THING

^^^ WILLY: BASICALLY THIS WAS MADE BY JUST COPYING AND PASTING THE 3 KERNELS ABOVE
AND THEN BRAINDED COPYING THEM VERBATIM

MAKE A GPT SCRIPT THAT AUTO DOES THIS INSTEAD OF ME ASKING CURSOR

"TAKE THESE 3 THINGS AND BRAINDED PUT THEM INTO THIS SO ALL THE PARTS ARE FASTER"

"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


# =============================================================================
# TileLang Kernel 1: Normalizer (from 5_0_4.py)
# Computes Z = sum over causal positions of (1 + S + 0.5*S^2) where S = Q @ K.T
# Q and K have feature_dim
# =============================================================================
def _build_normalizer_kernel(batch, heads, seq_len, feature_dim, dtype="bfloat16"):
    scale = feature_dim ** -0.5
    BLOCK_M = 64
    BLOCK_N = 64
    
    @T.prim_func
    def kernel(
        Q: T.Tensor((batch, heads, seq_len, feature_dim), dtype),
        K: T.Tensor((batch, heads, seq_len, feature_dim), dtype),
        Z: T.Tensor((batch, heads, seq_len), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, BLOCK_M), heads, batch, threads=128) as (bx, by, bz):
            # Shared Memory
            Q_s = T.alloc_shared((BLOCK_M, feature_dim), dtype)
            K_s = T.alloc_shared((BLOCK_N, feature_dim), dtype)
            
            # Accumulator for Z (row sums), FP32 for precision
            Acc_Z = T.alloc_fragment((BLOCK_M,), "float32")
            
            # Fragment for intermediate scores
            S_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            row_sum_frag = T.alloc_fragment((BLOCK_M,), "float32")
            
            # Layout Hints - only swizzle for larger dimensions
            if feature_dim >= 64:
                T.annotate_layout({
                    Q_s: tilelang.layout.make_swizzled_layout(Q_s),
                    K_s: tilelang.layout.make_swizzled_layout(K_s),
                })
            
            # Initialize Accumulator
            T.clear(Acc_Z)
            
            # Load Q Block (apply scale)
            base_m = bx * BLOCK_M
            for i, j in T.Parallel(BLOCK_M, feature_dim):
                if base_m + i < seq_len:
                    Q_s[i, j] = Q[bz, by, base_m + i, j] * scale
                else:
                    Q_s[i, j] = 0.0
            
            # Causal Loop over K blocks
            loop_range = bx + 1
            for k_idx in T.Pipelined(loop_range, num_stages=1):
                base_n = k_idx * BLOCK_N
                
                # Load K Block
                for i, j in T.Parallel(BLOCK_N, feature_dim):
                    if base_n + i < seq_len:
                        K_s[i, j] = K[bz, by, base_n + i, j]
                    else:
                        K_s[i, j] = 0.0
                
                # Compute S = Q @ K.T
                T.clear(S_frag)
                T.gemm(Q_s, K_s, S_frag, transpose_B=True)
                
                # Apply Polynomial & Causal Mask & Row Reduction
                # P = 1 + S + 0.5 * S^2
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    r = base_m + i
                    c = base_n + j
                    
                    if r < seq_len and c < seq_len:
                        if r >= c:
                            val = S_frag[i, j]
                            p_val = 1.0 + val + 0.5 * val * val
                            S_frag[i, j] = p_val
                        else:
                            S_frag[i, j] = 0.0
                    else:
                        S_frag[i, j] = 0.0
                
                # Reduce rows
                T.reduce_sum(S_frag, row_sum_frag, dim=1)
                
                # Accumulate
                for i in T.Parallel(BLOCK_M):
                    Acc_Z[i] += row_sum_frag[i]
            
            # Store Output
            for i in T.Parallel(BLOCK_M):
                if base_m + i < seq_len:
                    Z[bz, by, base_m + i] = T.cast(Acc_Z[i], dtype)

    return tilelang.compile(kernel, out_idx=[2], target="cuda")


# =============================================================================
# TileLang Kernel 2: Inter-chunk Linear (from 5_3_5.py)
# Computes O = Q @ cumulative_state where state accumulates K.T @ V
# Q/K have feature_dim, V/O have head_dim
# State S has shape [feature_dim, head_dim]
# =============================================================================
def _build_inter_chunk_kernel(batch, heads, n_chunks, chunk_size, feature_dim, head_dim, dtype="bfloat16"):
    scale = feature_dim ** -0.5
    
    @T.prim_func
    def kernel(
        Q: T.Tensor((batch, heads, n_chunks, chunk_size, feature_dim), dtype),
        K: T.Tensor((batch, heads, n_chunks, chunk_size, feature_dim), dtype),
        V: T.Tensor((batch, heads, n_chunks, chunk_size, head_dim), dtype),
        O: T.Tensor((batch, heads, n_chunks, chunk_size, head_dim), dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            # State in Shared Memory: [feature_dim, head_dim]
            S_s = T.alloc_shared((feature_dim, head_dim), dtype)
            
            # Input tiles for current chunk
            Q_s = T.alloc_shared((chunk_size, feature_dim), dtype)
            K_s = T.alloc_shared((chunk_size, feature_dim), dtype)
            V_s = T.alloc_shared((chunk_size, head_dim), dtype)
            
            # Accumulators
            O_frag = T.alloc_fragment((chunk_size, head_dim), "float32")
            Update_frag = T.alloc_fragment((feature_dim, head_dim), "float32")
            
            # Initialize State to 0
            T.clear(S_s)
            
            # Layout Hints - only swizzle for larger dimensions
            if feature_dim >= 64 and head_dim >= 64:
                T.annotate_layout({
                    S_s: tilelang.layout.make_swizzled_layout(S_s),
                    Q_s: tilelang.layout.make_swizzled_layout(Q_s),
                    K_s: tilelang.layout.make_swizzled_layout(K_s),
                    V_s: tilelang.layout.make_swizzled_layout(V_s),
                })

            # Loop over chunks sequentially
            for c in T.serial(n_chunks):
                # 1. Load Inputs
                for i, j in T.Parallel(chunk_size, feature_dim):
                    Q_s[i, j] = Q[bz, by, c, i, j] * scale
                    K_s[i, j] = K[bz, by, c, i, j]
                
                for i, j in T.Parallel(chunk_size, head_dim):
                    V_s[i, j] = V[bz, by, c, i, j]
                
                T.copy(Q_s, Q_s)
                
                # 2. Compute Output: O = Q @ S_s  [chunk_size, feature_dim] @ [feature_dim, head_dim]
                T.clear(O_frag)
                T.gemm(Q_s, S_s, O_frag)
                
                # Store Output
                for i, j in T.Parallel(chunk_size, head_dim):
                    O[bz, by, c, i, j] = O_frag[i, j]
                
                # 3. Update State: Update = K.T @ V  [feature_dim, chunk_size] @ [chunk_size, head_dim]
                T.clear(Update_frag)
                T.gemm(K_s, V_s, Update_frag, transpose_A=True)
                
                # Add Update to S_s
                for i, j in T.Parallel(feature_dim, head_dim):
                    val = S_s[i, j] + Update_frag[i, j]
                    S_s[i, j] = T.cast(val, dtype)
                
                T.copy(S_s, S_s)

    return tilelang.compile(kernel, out_idx=[3], target="cuda")


# =============================================================================
# TileLang Kernel 3: Combine + Normalize (from 5_5_2.py)
# Computes: out = (intra + inter_q + inter_l + const) / (norm + eps)
# All outputs have head_dim
# =============================================================================
def _build_combine_norm_kernel(total_rows, head_dim, eps, dtype="bfloat16"):
    BLOCK_M = 128

    @T.prim_func
    def kernel(
        intra: T.Tensor((total_rows, head_dim), dtype),
        inter_q: T.Tensor((total_rows, head_dim), dtype),
        inter_l: T.Tensor((total_rows, head_dim), dtype),
        const_out: T.Tensor((total_rows, head_dim), dtype),
        norm: T.Tensor((total_rows,), dtype),
        out: T.Tensor((total_rows, head_dim), dtype),
    ):
        with T.Kernel(T.ceildiv(total_rows, BLOCK_M), threads=128) as bx:
            base_row = bx * BLOCK_M
            
            for m in T.Parallel(BLOCK_M):
                row_idx = base_row + m
                if row_idx < total_rows:
                    norm_val = T.cast(norm[row_idx], "float32")
                    denom = norm_val + eps
                    recip = 1.0 / denom
                    
                    for n in T.vectorized(head_dim):
                        v1 = T.cast(intra[row_idx, n], "float32")
                        v2 = T.cast(inter_q[row_idx, n], "float32")
                        v3 = T.cast(inter_l[row_idx, n], "float32")
                        v4 = T.cast(const_out[row_idx, n], "float32")
                        
                        sum_val = v1 + v2 + v3 + v4
                        res = sum_val * recip
                        
                        out[row_idx, n] = T.cast(res, dtype)

    return tilelang.compile(kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    """
    Hybrid Chunked Based Linear Attention with separate feature_dim and head_dim.
    
    Supports Q/K with feature_dim (e.g., 16) and V with head_dim (e.g., 128).
    """
    def __init__(self, chunk_size: int = 256, eps: float = 1e-6):
        super(ModelNew, self).__init__()
        self.chunk_size = chunk_size
        self.eps = eps
        object.__setattr__(self, "_normalizer_cache", {})
        object.__setattr__(self, "_inter_chunk_cache", {})
        object.__setattr__(self, "_combine_cache", {})

    def _get_dtype_str(self, tensor):
        if tensor.dtype == torch.bfloat16:
            return "bfloat16"
        elif tensor.dtype == torch.float16:
            return "float16"
        return "float32"

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, h, seq_len, feature_dim = q.shape
        head_dim = v.shape[-1]
        chunk_size = self.chunk_size
        dtype_str = self._get_dtype_str(q)
        
        assert seq_len % chunk_size == 0, f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
        n_chunks = seq_len // chunk_size
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # 1. Compute Normalizer using TileLang kernel (uses feature_dim)
        norm_key = (b, h, seq_len, feature_dim, dtype_str)
        if norm_key not in self._normalizer_cache:
            self._normalizer_cache[norm_key] = _build_normalizer_kernel(b, h, seq_len, feature_dim, dtype_str)
        normalizer_kernel = self._normalizer_cache[norm_key]
        z = normalizer_kernel(q, k)  # [B, H, L]
        
        # 2. Compute Constant term (cumsum of V) - PyTorch
        constant_output = v.cumsum(dim=-2)  # [B, H, L, head_dim]
        
        # 3. Reshape into chunks
        q_chunks = q.view(b, h, n_chunks, chunk_size, feature_dim).contiguous()
        k_chunks = k.view(b, h, n_chunks, chunk_size, feature_dim).contiguous()
        v_chunks = v.view(b, h, n_chunks, chunk_size, head_dim).contiguous()
        
        # 4. Intra-chunk attention - PyTorch
        scale = feature_dim ** -0.5
        q_scaled = q_chunks * scale
        
        # S = Q @ K.T  [B, H, N, C, feature_dim] @ [B, H, N, feature_dim, C] -> [B, H, N, C, C]
        intra_chunk_attn = q_scaled @ k_chunks.transpose(-2, -1)
        intra_chunk_attn = intra_chunk_attn + 0.5 * (intra_chunk_attn ** 2)
        
        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device))
        intra_chunk_attn = intra_chunk_attn.masked_fill(~causal_mask, 0)
        
        # O = attn @ V  [B, H, N, C, C] @ [B, H, N, C, head_dim] -> [B, H, N, C, head_dim]
        intra_output = intra_chunk_attn @ v_chunks
        intra_output = intra_output.view(b, h, seq_len, head_dim)
        
        # 5. Inter-chunk quadratic term - PyTorch
        # kv_quad = einsum('bhncd,bhnce,bhncf->bhndef', k, k, v)
        # k: [B, H, N, C, feature_dim], v: [B, H, N, C, head_dim]
        # kv_quad: [B, H, N, feature_dim, feature_dim, head_dim]
        kv_quad = torch.einsum('bhncd,bhnce,bhncf->bhndef', k_chunks, k_chunks, v_chunks)
        kv_quad = kv_quad.cumsum(2)
        kv_quad = torch.cat([torch.zeros_like(kv_quad[:, :, :1]), kv_quad[:, :, :-1]], dim=2)
        # Contract with q⊗q: [B, H, N, feature_dim, feature_dim, head_dim] x [B, H, N, C, feature_dim] x [B, H, N, C, feature_dim]
        inter_quad_output = 0.5 * torch.einsum('bhndef,bhncd,bhnce->bhncf', kv_quad, q_scaled, q_scaled)
        inter_quad_output = inter_quad_output.view(b, h, seq_len, head_dim)
        
        # 6. Inter-chunk linear term - TileLang kernel
        inter_key = (b, h, n_chunks, chunk_size, feature_dim, head_dim, q.dtype)
        if inter_key not in self._inter_chunk_cache:
            self._inter_chunk_cache[inter_key] = _build_inter_chunk_kernel(
                b, h, n_chunks, chunk_size, feature_dim, head_dim, dtype=dtype_str
            )
        inter_kernel = self._inter_chunk_cache[inter_key]
        inter_linear_output = inter_kernel(q_chunks, k_chunks, v_chunks)
        inter_linear_output = inter_linear_output.view(b, h, seq_len, head_dim)
        
        # 7. Combine and Normalize - TileLang kernel
        total_rows = b * h * seq_len
        combine_key = (total_rows, head_dim, self.eps, dtype_str)
        if combine_key not in self._combine_cache:
            self._combine_cache[combine_key] = _build_combine_norm_kernel(total_rows, head_dim, self.eps, dtype_str)
        combine_kernel = self._combine_cache[combine_key]
        
        intra_flat = intra_output.view(total_rows, head_dim).contiguous()
        inter_q_flat = inter_quad_output.view(total_rows, head_dim).contiguous()
        inter_l_flat = inter_linear_output.view(total_rows, head_dim).contiguous()
        const_flat = constant_output.view(total_rows, head_dim).contiguous()
        norm_flat = z.view(total_rows).contiguous()
        
        out_flat = combine_kernel(intra_flat, inter_q_flat, inter_l_flat, const_flat, norm_flat)
        
        return out_flat.view(b, h, seq_len, head_dim)


# Kernelbench Parameters
batch_size = 4
num_heads = 8
seq_len = 512
feature_dim = 16
head_dim = 128
chunk_size = 256

def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_len, feature_dim)
    k = torch.randn(batch_size, num_heads, seq_len, feature_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return [q, k, v]

def get_init_inputs():
    return [chunk_size]

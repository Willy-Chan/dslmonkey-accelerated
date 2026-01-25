"""
Hybrid Chunked Based Linear Attention.

Combines PyTorch for some operations with optimized TileLang kernels for:
1. Normalizer computation (5_0_4.py)
2. Inter-chunk linear term (5_3_5.py)  
3. Final combine + normalize (5_5_2.py)
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


# =============================================================================
# TileLang Kernel 1: Normalizer (from 5_0_4.py)
# Computes Z = sum over causal positions of (1 + S + 0.5*S^2) where S = Q @ K.T
# =============================================================================
def _build_normalizer_kernel(batch, heads, seq_len, dim, dtype="bfloat16"):
    scale = dim ** -0.5
    BLOCK_M = 64
    BLOCK_N = 64
    
    @T.prim_func
    def kernel(
        Q: T.Tensor((batch, heads, seq_len, dim), dtype),
        K: T.Tensor((batch, heads, seq_len, dim), dtype),
        Z: T.Tensor((batch, heads, seq_len), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, BLOCK_M), heads, batch, threads=128) as (bx, by, bz):
            # Shared Memory
            Q_s = T.alloc_shared((BLOCK_M, dim), dtype)
            K_s = T.alloc_shared((BLOCK_N, dim), dtype)
            
            # Accumulator for Z (row sums), FP32 for precision
            Acc_Z = T.alloc_fragment((BLOCK_M,), "float32")
            
            # Fragment for intermediate scores
            S_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            row_sum_frag = T.alloc_fragment((BLOCK_M,), "float32")
            
            # Layout Hints
            T.annotate_layout({
                Q_s: tilelang.layout.make_swizzled_layout(Q_s),
                K_s: tilelang.layout.make_swizzled_layout(K_s),
            })
            
            # Initialize Accumulator
            T.clear(Acc_Z)
            
            # Load Q Block (apply scale)
            base_m = bx * BLOCK_M
            for i, j in T.Parallel(BLOCK_M, dim):
                if base_m + i < seq_len:
                    Q_s[i, j] = Q[bz, by, base_m + i, j] * scale
                else:
                    Q_s[i, j] = 0.0
            
            # Causal Loop over K blocks
            loop_range = bx + 1
            for k_idx in T.Pipelined(loop_range, num_stages=1):
                base_n = k_idx * BLOCK_N
                
                # Load K Block
                for i, j in T.Parallel(BLOCK_N, dim):
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
                    
                    # Check global boundaries and causality
                    if r < seq_len and c < seq_len:
                        if r >= c:
                            val = S_frag[i, j]
                            # Polynomial expansion
                            p_val = 1.0 + val + 0.5 * val * val
                            S_frag[i, j] = p_val
                        else:
                            S_frag[i, j] = 0.0
                    else:
                        S_frag[i, j] = 0.0
                
                # Reduce rows of the current block score matrix
                T.reduce_sum(S_frag, row_sum_frag, dim=1)
                
                # Accumulate into global Z accumulator
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
# =============================================================================
def _build_inter_chunk_kernel(batch, heads, n_chunks, chunk_size, dim, dtype="bfloat16"):
    scale = dim ** -0.5
    
    @T.prim_func
    def kernel(
        Q: T.Tensor((batch, heads, n_chunks, chunk_size, dim), dtype),
        K: T.Tensor((batch, heads, n_chunks, chunk_size, dim), dtype),
        V: T.Tensor((batch, heads, n_chunks, chunk_size, dim), dtype),
        O: T.Tensor((batch, heads, n_chunks, chunk_size, dim), dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            # State in Shared Memory: [D, D]
            S_s = T.alloc_shared((dim, dim), dtype)
            
            # Input tiles for current chunk
            Q_s = T.alloc_shared((chunk_size, dim), dtype)
            K_s = T.alloc_shared((chunk_size, dim), dtype)
            V_s = T.alloc_shared((chunk_size, dim), dtype)
            
            # Accumulators
            O_frag = T.alloc_fragment((chunk_size, dim), "float32")
            Update_frag = T.alloc_fragment((dim, dim), "float32")
            
            # Initialize State to 0
            T.clear(S_s)
            
            T.annotate_layout({
                S_s: tilelang.layout.make_swizzled_layout(S_s),
                Q_s: tilelang.layout.make_swizzled_layout(Q_s),
                K_s: tilelang.layout.make_swizzled_layout(K_s),
                V_s: tilelang.layout.make_swizzled_layout(V_s),
            })

            # Loop over chunks sequentially to maintain state dependency
            for c in T.serial(n_chunks):
                # 1. Load Inputs (Q scaled, K, V)
                for i, j in T.Parallel(chunk_size, dim):
                    Q_s[i, j] = Q[bz, by, c, i, j] * scale
                    K_s[i, j] = K[bz, by, c, i, j]
                    V_s[i, j] = V[bz, by, c, i, j]
                
                # Ensure loads are visible
                T.copy(Q_s, Q_s)
                
                # 2. Compute Output using PREVIOUS state: O = Q @ S_s
                T.clear(O_frag)
                T.gemm(Q_s, S_s, O_frag)
                
                # Store Output
                for i, j in T.Parallel(chunk_size, dim):
                    O[bz, by, c, i, j] = O_frag[i, j]
                
                # 3. Update State for NEXT chunk: Update = K.T @ V
                T.clear(Update_frag)
                T.gemm(K_s, V_s, Update_frag, transpose_A=True)
                
                # Add Update_frag to S_s
                for i, j in T.Parallel(dim, dim):
                    val = S_s[i, j] + Update_frag[i, j]
                    S_s[i, j] = T.cast(val, dtype)
                
                # Synchronization barrier
                T.copy(S_s, S_s)

    return tilelang.compile(kernel, out_idx=[3], target="cuda")


# =============================================================================
# TileLang Kernel 3: Combine + Normalize (from 5_5_2.py)
# Computes: out = (intra + inter_q + inter_l + const) / (norm + eps)
# =============================================================================
def _build_combine_norm_kernel(total_rows, D, eps, dtype="bfloat16"):
    BLOCK_M = 128
    BLOCK_N = 64

    @T.prim_func
    def kernel(
        intra: T.Tensor((total_rows, D), dtype),
        inter_q: T.Tensor((total_rows, D), dtype),
        inter_l: T.Tensor((total_rows, D), dtype),
        const_out: T.Tensor((total_rows, D), dtype),
        norm: T.Tensor((total_rows,), dtype),
        out: T.Tensor((total_rows, D), dtype),
    ):
        with T.Kernel(T.ceildiv(total_rows, BLOCK_M), threads=128) as bx:
            base_row = bx * BLOCK_M
            
            for m in T.Parallel(BLOCK_M):
                row_idx = base_row + m
                if row_idx < total_rows:
                    norm_val = T.cast(norm[row_idx], "float32")
                    denom = norm_val + eps
                    recip = 1.0 / denom
                    
                    for n in T.vectorized(D):
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
    Hybrid Chunked Based Linear Attention.
    
    Uses TileLang kernels for:
    - Normalizer computation
    - Inter-chunk linear term
    - Final combine + normalize
    
    Uses PyTorch for:
    - Intra-chunk attention
    - Inter-chunk quadratic term
    - Constant term (cumsum)
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
        b, h, seq_len, d = q.shape
        chunk_size = self.chunk_size
        dtype_str = self._get_dtype_str(q)
        
        # Ensure divisibility
        assert seq_len % chunk_size == 0, f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
        n_chunks = seq_len // chunk_size
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # =====================================================================
        # 1. Compute Normalizer using TileLang kernel
        # =====================================================================
        norm_key = (b, h, seq_len, d, dtype_str)
        if norm_key not in self._normalizer_cache:
            self._normalizer_cache[norm_key] = _build_normalizer_kernel(b, h, seq_len, d, dtype_str)
        normalizer_kernel = self._normalizer_cache[norm_key]
        
        # Normalizer kernel expects unscaled Q (it scales internally)
        z = normalizer_kernel(q, k)  # [B, H, L]
        
        # =====================================================================
        # 2. Compute Constant term (cumsum of V) - PyTorch
        # =====================================================================
        constant_output = v.cumsum(dim=-2)  # [B, H, L, D]
        
        # =====================================================================
        # 3. Reshape into chunks for chunk-wise operations
        # =====================================================================
        q_chunks = q.view(b, h, n_chunks, chunk_size, d).contiguous()
        k_chunks = k.view(b, h, n_chunks, chunk_size, d).contiguous()
        v_chunks = v.view(b, h, n_chunks, chunk_size, d).contiguous()
        
        # =====================================================================
        # 4. Intra-chunk attention - PyTorch
        # =====================================================================
        scale = d ** -0.5
        q_scaled = q_chunks * scale
        
        # S = Q @ K.T within each chunk
        intra_chunk_attn = q_scaled @ k_chunks.transpose(-2, -1)
        # Polynomial: 1 + S + 0.5*S^2
        intra_chunk_attn = intra_chunk_attn + 0.5 * (intra_chunk_attn ** 2)
        
        # Apply causal mask within chunk
        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device))
        intra_chunk_attn = intra_chunk_attn.masked_fill(~causal_mask, 0)
        
        # Intra output
        intra_output = intra_chunk_attn @ v_chunks  # [B, H, N, C, D]
        intra_output = intra_output.view(b, h, seq_len, d)
        
        # =====================================================================
        # 5. Inter-chunk quadratic term - PyTorch
        # =====================================================================
        # kv_quad = einsum('bhncd,bhnce,bhncf->bhndef', k, k, v)
        kv_quad = torch.einsum('bhncd,bhnce,bhncf->bhndef', k_chunks, k_chunks, v_chunks)
        kv_quad = kv_quad.cumsum(2)
        # Shift by one chunk (previous chunks only)
        kv_quad = torch.cat([torch.zeros_like(kv_quad[:, :, :1]), kv_quad[:, :, :-1]], dim=2)
        # Contract with q⊗q
        inter_quad_output = 0.5 * torch.einsum('bhndef,bhncd,bhnce->bhncf', kv_quad, q_scaled, q_scaled)
        inter_quad_output = inter_quad_output.view(b, h, seq_len, d)
        
        # =====================================================================
        # 6. Inter-chunk linear term - TileLang kernel
        # =====================================================================
        inter_key = (b, h, n_chunks, chunk_size, d, q.dtype)
        if inter_key not in self._inter_chunk_cache:
            self._inter_chunk_cache[inter_key] = _build_inter_chunk_kernel(
                b, h, n_chunks, chunk_size, d, dtype=dtype_str
            )
        inter_kernel = self._inter_chunk_cache[inter_key]
        
        # Inter-chunk kernel expects unscaled Q (it scales internally)
        inter_linear_output = inter_kernel(q_chunks, k_chunks, v_chunks)  # [B, H, N, C, D]
        inter_linear_output = inter_linear_output.view(b, h, seq_len, d)
        
        # =====================================================================
        # 7. Combine and Normalize - TileLang kernel
        # =====================================================================
        total_rows = b * h * seq_len
        combine_key = (total_rows, d, self.eps, dtype_str)
        if combine_key not in self._combine_cache:
            self._combine_cache[combine_key] = _build_combine_norm_kernel(total_rows, d, self.eps, dtype_str)
        combine_kernel = self._combine_cache[combine_key]
        
        # Flatten inputs for combine kernel
        intra_flat = intra_output.view(total_rows, d).contiguous()
        inter_q_flat = inter_quad_output.view(total_rows, d).contiguous()
        inter_l_flat = inter_linear_output.view(total_rows, d).contiguous()
        const_flat = constant_output.view(total_rows, d).contiguous()
        norm_flat = z.view(total_rows).contiguous()
        
        out_flat = combine_kernel(intra_flat, inter_q_flat, inter_l_flat, const_flat, norm_flat)
        
        return out_flat.view(b, h, seq_len, d)


# Kernelbench Parameters
batch_size = 4
num_heads = 8
seq_len = 512  # Must be divisible by chunk_size
head_dim = 64
chunk_size = 256

def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return [q, k, v]

def get_init_inputs():
    return [chunk_size]

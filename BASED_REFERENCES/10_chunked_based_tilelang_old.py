import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

# ============================================================================
# TileLang Kernels
# ============================================================================

def _build_linear_state_kernel(batch, heads, seq_len, head_dim, chunk_size, dtype="float16"):
    """
    Computes the shifted cumulative sum of K^T @ V for each chunk.
    Input: K, V (B, H, Seq, D)
    Output: State (B, H, N_chunks, D, D)
    """
    n_chunks = seq_len // chunk_size
    total_seqs = batch * heads
    BLOCK_C = 32 
    
    @T.prim_func
    def kernel_func(
        K: T.Tensor((batch, heads, seq_len, head_dim), dtype),
        V: T.Tensor((batch, heads, seq_len, head_dim), dtype),
        Out: T.Tensor((batch, heads, n_chunks, head_dim, head_dim), dtype),
    ):
        with T.Kernel(total_seqs, threads=128) as bx:
            h_idx = bx % heads
            b_idx = bx // heads
            
            acc = T.alloc_fragment((head_dim, head_dim), "float32")
            T.clear(acc)
            
            shm_K = T.alloc_shared((BLOCK_C, head_dim), dtype)
            shm_V = T.alloc_shared((BLOCK_C, head_dim), dtype)
            
            T.annotate_layout({
                shm_K: tilelang.layout.make_swizzled_layout(shm_K),
                shm_V: tilelang.layout.make_swizzled_layout(shm_V),
            })
            
            for n in T.serial(n_chunks):
                # 1. Write current accumulator (shifted state)
                for i, j in T.Parallel(head_dim, head_dim):
                    Out[b_idx, h_idx, n, i, j] = acc[i, j]
                
                # 2. Accumulate current chunk
                num_steps = T.ceildiv(chunk_size, BLOCK_C)
                for k in T.Pipelined(num_steps, num_stages=2):
                    base_row = n * chunk_size + k * BLOCK_C
                    
                    for ii, jj in T.Parallel(BLOCK_C, head_dim):
                        row = base_row + ii
                        if row < seq_len:
                            shm_K[ii, jj] = K[b_idx, h_idx, row, jj]
                        else:
                            shm_K[ii, jj] = 0.0
                            
                    for ii, jj in T.Parallel(BLOCK_C, head_dim):
                        row = base_row + ii
                        if row < seq_len:
                            shm_V[ii, jj] = V[b_idx, h_idx, row, jj]
                        else:
                            shm_V[ii, jj] = 0.0
                            
                    T.gemm(shm_K, shm_V, acc, transpose_A=True, transpose_B=False)
                    
    return tilelang.compile(kernel_func, out_idx=[2], target="cuda")


def _build_linear_proj_kernel(batch, heads, seq_len, head_dim, chunk_size, dtype="float16"):
    """
    Computes O = Q * S where Q is scaled by head_dim**-0.5.
    Input Q: (B, H, Seq, D)
    Input S: (B, H, N_chunks, D, D)
    Output: (B, H, Seq, D)
    """
    n_chunks = seq_len // chunk_size
    total_chunks = batch * heads * n_chunks
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    scale = head_dim ** -0.5
    
    @T.prim_func
    def kernel_func(
        Q: T.Tensor((batch, heads, seq_len, head_dim), dtype),
        S: T.Tensor((batch, heads, n_chunks, head_dim, head_dim), dtype),
        Out: T.Tensor((batch, heads, seq_len, head_dim), dtype)
    ):
        grid_x = T.ceildiv(chunk_size, BLOCK_M)
        
        with T.Kernel(grid_x, total_chunks, threads=128) as (bx, by):
            n_idx = by % n_chunks
            temp = by // n_chunks
            h_idx = temp % heads
            b_idx = temp // heads
            
            Q_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            S_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            Acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            
            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                S_shared: tilelang.layout.make_swizzled_layout(S_shared),
            })
            
            for i, j in T.Parallel(BLOCK_K, BLOCK_N):
                S_shared[i, j] = S[b_idx, h_idx, n_idx, i, j]
            
            base_row = n_idx * chunk_size + bx * BLOCK_M
            
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                row = base_row + i
                if row < seq_len:
                    Q_shared[i, j] = Q[b_idx, h_idx, row, j] * scale
                else:
                    Q_shared[i, j] = 0.0
            
            T.clear(Acc)
            T.gemm(Q_shared, S_shared, Acc)
            
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                row = base_row + i
                if row < seq_len:
                    Out[b_idx, h_idx, row, j] = Acc[i, j]

    return tilelang.compile(kernel_func, out_idx=[2], target="cuda")


def _build_quad_expand_kernel(batch, heads, seq_len, head_dim, chunk_size, dtype="float16"):
    """
    Computes the rank-3 quadratic state expansion K @ K @ V per chunk.
    Output shape: (B, H, N_chunks, D, D, D)
    """
    n_chunks = seq_len // chunk_size
    total_chunks = batch * heads * n_chunks
    
    BLOCK_C = 64
    BLOCK_D = 64
    
    @T.prim_func
    def kernel_func(
        K: T.Tensor((batch, heads, n_chunks, chunk_size, head_dim), dtype),
        V: T.Tensor((batch, heads, n_chunks, chunk_size, head_dim), dtype),
        Out: T.Tensor((batch, heads, n_chunks, head_dim, head_dim, head_dim), dtype)
    ):
        with T.Kernel(head_dim, total_chunks, threads=128) as (bx, by):
            d_idx = bx
            n_idx = by % n_chunks
            temp = by // n_chunks
            h_idx = temp % heads
            b_idx = temp // heads
            
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
            
            num_stages = chunk_size // BLOCK_C
            
            for k_iter in T.Pipelined(num_stages, num_stages=2):
                c_offset = k_iter * BLOCK_C
                
                for i, j in T.Parallel(BLOCK_C, BLOCK_D):
                    K_shared[i, j] = K[b_idx, h_idx, n_idx, c_offset + i, j]
                    V_shared[i, j] = V[b_idx, h_idx, n_idx, c_offset + i, j]
                
                for i, j in T.Parallel(BLOCK_C, BLOCK_D):
                    scalar = K_shared[i, d_idx]
                    val = K_shared[i, j]
                    K_scaled_shared[i, j] = val * scalar
                
                T.gemm(K_scaled_shared, V_shared, Acc, transpose_A=True)
            
            for i, j in T.Parallel(BLOCK_D, BLOCK_D):
                Out[b_idx, h_idx, n_idx, d_idx, i, j] = Acc[i, j]

    return tilelang.compile(kernel_func, out_idx=[2], target="cuda")


def _build_quad_cumsum_kernel(batch, heads, n_chunks, dim, dtype="float16"):
    """
    Computes shifted cumsum for (B, H, N, D, D, D).
    Flatten (D, D, D) -> M.
    """
    M = dim * dim * dim
    BLOCK_M = 256
    accum_dtype = "float32"
    
    @T.prim_func
    def kernel_func(
        In: T.Tensor((batch, heads, n_chunks, M), dtype),
        Out: T.Tensor((batch, heads, n_chunks, M), dtype)
    ):
        grid_m = T.ceildiv(M, BLOCK_M)
        
        with T.Kernel(grid_m, batch * heads, threads=128) as (bx, by):
            h_idx = by % heads
            b_idx = by // heads
            
            m_base = bx * BLOCK_M
            
            acc = T.alloc_fragment((BLOCK_M,), accum_dtype)
            T.clear(acc)
            
            for n in T.serial(n_chunks):
                for i in T.Parallel(BLOCK_M):
                    m_idx = m_base + i
                    if m_idx < M:
                        Out[b_idx, h_idx, n, m_idx] = acc[i]
                        
                for i in T.Parallel(BLOCK_M):
                    m_idx = m_base + i
                    if m_idx < M:
                        val = In[b_idx, h_idx, n, m_idx]
                        acc[i] += val
                        
    return tilelang.compile(kernel_func, out_idx=[1], target="cuda")


def _build_quad_project_kernel(batch, heads, seq_len, head_dim, chunk_size, dtype="float16"):
    """
    Computes O = 0.5 * einsum('bhndef,bhncd,bhnce->bhncf', S, Q, Q)
    """
    n_chunks = seq_len // chunk_size
    total_chunks = batch * heads * n_chunks
    
    BLOCK_C = 64
    D = head_dim
    
    q_scale = head_dim ** -0.5
    final_scale = 0.5

    @T.prim_func
    def kernel_func(
        Q: T.Tensor((batch, heads, seq_len, head_dim), dtype),
        S: T.Tensor((batch, heads, n_chunks, head_dim, head_dim, head_dim), dtype),
        Out: T.Tensor((batch, heads, seq_len, head_dim), dtype)
    ):
        grid_c = T.ceildiv(chunk_size, BLOCK_C)
        
        with T.Kernel(total_chunks, grid_c, threads=128) as (by, bx):
            n_idx = by % n_chunks
            tmp = by // n_chunks
            h_idx = tmp % heads
            b_idx = tmp // heads
            
            chunk_start_row = n_idx * chunk_size
            row_offset = chunk_start_row + bx * BLOCK_C
            
            shm_Q = T.alloc_shared((BLOCK_C, D), dtype)
            shm_S = T.alloc_shared((D, D), dtype)
            
            acc = T.alloc_fragment((BLOCK_C, D), "float32")
            gemm_res = T.alloc_fragment((BLOCK_C, D), "float32")
            
            T.annotate_layout({
                shm_Q: tilelang.layout.make_swizzled_layout(shm_Q),
                shm_S: tilelang.layout.make_swizzled_layout(shm_S),
            })

            T.clear(acc)
            
            for i, j in T.Parallel(BLOCK_C, D):
                r = row_offset + i
                if r < seq_len:
                    val = Q[b_idx, h_idx, r, j]
                    shm_Q[i, j] = val * q_scale
                else:
                    shm_Q[i, j] = 0.0
            
            for e in T.Pipelined(D, num_stages=2):
                for d_local, f_local in T.Parallel(D, D):
                    shm_S[d_local, f_local] = S[b_idx, h_idx, n_idx, d_local, e, f_local]
                
                T.clear(gemm_res)
                T.gemm(shm_Q, shm_S, gemm_res)
                
                for i, j in T.Parallel(BLOCK_C, D):
                    q_val = shm_Q[i, e]
                    acc[i, j] += gemm_res[i, j] * T.cast(q_val, "float32")

            for i, j in T.Parallel(BLOCK_C, D):
                r = row_offset + i
                if r < seq_len:
                    val = acc[i, j] * final_scale
                    Out[b_idx, h_idx, r, j] = val

    return tilelang.compile(kernel_func, out_idx=[2], target="cuda")


# ============================================================================
# Model Definition
# ============================================================================

class ModelNew(nn.Module):
    """
    Optimized Chunk-based Linear Attention (Based) using TileLang kernels.
    """
    def __init__(self, chunk_size: int = 256):
        super(ModelNew, self).__init__()
        self.chunk_size = chunk_size
        self._kernel_cache = {}

    def _get_kernel(self, kernel_type, *args):
        key = (kernel_type, *args)
        if key not in self._kernel_cache:
            if kernel_type == 'linear_state':
                self._kernel_cache[key] = _build_linear_state_kernel(*args)
            elif kernel_type == 'linear_proj':
                self._kernel_cache[key] = _build_linear_proj_kernel(*args)
            elif kernel_type == 'quad_expand':
                self._kernel_cache[key] = _build_quad_expand_kernel(*args)
            elif kernel_type == 'quad_cumsum':
                self._kernel_cache[key] = _build_quad_cumsum_kernel(*args)
            elif kernel_type == 'quad_proj':
                self._kernel_cache[key] = _build_quad_project_kernel(*args)
        return self._kernel_cache[key]

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q, k, v: (batch_size, num_heads, seq_len, head_dim)
        """
        b, h, seq_len, d = q.shape
        chunk_size = self.chunk_size
        
        # Ensure fp16 and contiguous memory
        target_dtype = torch.float16
        if q.dtype != target_dtype: q = q.to(target_dtype)
        if k.dtype != target_dtype: k = k.to(target_dtype)
        if v.dtype != target_dtype: v = v.to(target_dtype)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # --- PyTorch: Preprocessing and Intra-chunk Attention ---
        
        # Scale queries for PyTorch parts (kernels handle scaling internally)
        q_scaled = q * (d ** -0.5)
        
        # Compute normalizer (PyTorch)
        # Note: We use PyTorch for normalizer as no kernel was provided for it
        k_cumsum = torch.cumsum(k, dim=-2)
        kk_cumsum = torch.cumsum(k.unsqueeze(-1) * k.unsqueeze(-2), dim=-3)
        
        z = (q_scaled * k_cumsum).sum(-1)
        z = z + (q_scaled.unsqueeze(-1) * q_scaled.unsqueeze(-2) * kk_cumsum).sum((-1, -2)) * 0.5
        z = z + (torch.arange(0, seq_len, device=z.device, dtype=z.dtype) + 1.0)[None, None, :]
        
        # Constant term: cumulative sum of values (PyTorch)
        _o = v.cumsum(-2)
        
        # Intra-chunk attention (PyTorch)
        n_chunks = seq_len // chunk_size
        q_chunk = q_scaled.view(b, h, n_chunks, chunk_size, d)
        k_chunk = k.view(b, h, n_chunks, chunk_size, d)
        v_chunk = v.view(b, h, n_chunks, chunk_size, d)
        
        intra_chunk_attn = q_chunk @ k_chunk.transpose(-2, -1)
        intra_chunk_attn = intra_chunk_attn + 0.5 * (intra_chunk_attn ** 2)
        
        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device))
        intra_chunk_attn = intra_chunk_attn.masked_fill(~causal_mask, 0)
        
        o_intra = intra_chunk_attn @ v_chunk
        o_intra = o_intra.view(b, h, seq_len, d)
        
        # --- TileLang: Inter-chunk Linear Term ---
        
        # 1. Compute shifted cumulative state (K^T @ V)
        # Replaces: kv_lin = einsum(k, v).cumsum.shift
        linear_state_kernel = self._get_kernel('linear_state', b, h, seq_len, d, chunk_size, "float16")
        S_lin = linear_state_kernel(k, v)
        
        # 2. Project state to output (Q @ S)
        # Replaces: o += einsum(kv_lin, q)
        # Note: Kernel scales Q internally
        linear_proj_kernel = self._get_kernel('linear_proj', b, h, seq_len, d, chunk_size, "float16")
        o_lin = linear_proj_kernel(q, S_lin)
        
        # --- TileLang: Inter-chunk Quadratic Term ---
        
        # 3. Compute quadratic state expansion (K @ K @ V) per chunk
        # Replaces: kv_quad = einsum(k, k, v)
        quad_expand_kernel = self._get_kernel('quad_expand', b, h, seq_len, d, chunk_size, "float16")
        S_quad_raw = quad_expand_kernel(k_chunk, v_chunk)
        
        # 4. Compute shifted cumulative sum of quadratic state
        # Replaces: kv_quad.cumsum.shift
        quad_cumsum_kernel = self._get_kernel('quad_cumsum', b, h, n_chunks, d, "float16")
        # Flatten D*D*D for the kernel
        S_quad_flat = S_quad_raw.view(b, h, n_chunks, -1)
        S_quad_shifted_flat = quad_cumsum_kernel(S_quad_flat)
        S_quad_shifted = S_quad_shifted_flat.view(b, h, n_chunks, d, d, d)
        
        # 5. Project quadratic state to output (0.5 * S * Q * Q)
        # Replaces: o += 0.5 * einsum(kv_quad, q, q)
        # Note: Kernel scales Q internally
        quad_proj_kernel = self._get_kernel('quad_proj', b, h, seq_len, d, chunk_size, "float16")
        o_quad = quad_proj_kernel(q, S_quad_shifted)
        
        # --- Combine and Normalize ---
        
        o = o_intra + o_lin + o_quad
        
        o = o + _o
        o = o / (z[..., None] + 1e-6)
        
        return o

# ============================================================================
# KernelBench Parameters & Helpers
# ============================================================================

batch_size = 4
num_heads = 8
seq_len = 512
head_dim = 64
chunk_size = 256

def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    return [q, k, v]

def get_init_inputs():
    return [chunk_size]
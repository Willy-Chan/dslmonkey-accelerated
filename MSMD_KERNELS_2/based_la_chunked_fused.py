import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


### THIS IS FUSED BUT FAILS CORRECTNESS CHECK

# ============================================================================
# Fused TileLang Kernels
# ============================================================================

def _build_fused_linear_kernel(batch, heads, seq_len, feat_dim, head_dim, chunk_size, dtype="float16"):
    """
    Fused Linear Attention Kernel.
    Computes O = Q @ S_prev, then updates S += K^T @ V.
    Iterates over chunks, maintaining state in shared memory.
    """
    n_chunks = seq_len // chunk_size
    total_seqs = batch * heads
    BLOCK_C = 64
    
    @T.prim_func
    def kernel_func(
        Q: T.Tensor((batch, heads, seq_len, feat_dim), dtype),
        K: T.Tensor((batch, heads, seq_len, feat_dim), dtype),
        V: T.Tensor((batch, heads, seq_len, head_dim), dtype),
        Out: T.Tensor((batch, heads, seq_len, head_dim), dtype),
    ):
        with T.Kernel(total_seqs, threads=128) as bx:
            h_idx = bx % heads
            b_idx = bx // heads
            
            # State in shared memory: (Feat, Head)
            shm_S = T.alloc_shared((feat_dim, head_dim), dtype)
            
            # Buffers for sub-chunks
            shm_Q = T.alloc_shared((BLOCK_C, feat_dim), dtype)
            shm_K = T.alloc_shared((BLOCK_C, feat_dim), dtype)
            shm_V = T.alloc_shared((BLOCK_C, head_dim), dtype)
            
            # Fragments
            frag_O = T.alloc_fragment((BLOCK_C, head_dim), "float32")
            frag_S_upd = T.alloc_fragment((feat_dim, head_dim), "float32")
            
            # Annotate layouts for performance
            if feat_dim >= 64 and head_dim >= 64:
                T.annotate_layout({
                    shm_Q: tilelang.layout.make_swizzled_layout(shm_Q),
                    shm_K: tilelang.layout.make_swizzled_layout(shm_K),
                    shm_V: tilelang.layout.make_swizzled_layout(shm_V),
                    shm_S: tilelang.layout.make_swizzled_layout(shm_S),
                })
            
            T.clear(shm_S)
            
            for n in T.serial(n_chunks):
                num_steps = chunk_size // BLOCK_C
                for k in T.serial(num_steps):
                    base_row = n * chunk_size + k * BLOCK_C
                    
                    # 1. Load Q
                    for i, j in T.Parallel(BLOCK_C, feat_dim):
                        row = base_row + i
                        if row < seq_len:
                            shm_Q[i, j] = Q[b_idx, h_idx, row, j]
                        else:
                            shm_Q[i, j] = 0.0
                            
                    # 2. Compute O = Q @ S (using current state from previous chunks)
                    T.clear(frag_O)
                    T.gemm(shm_Q, shm_S, frag_O)
                    
                    # 3. Store O
                    for i, j in T.Parallel(BLOCK_C, head_dim):
                        row = base_row + i
                        if row < seq_len:
                            Out[b_idx, h_idx, row, j] = frag_O[i, j]
                            
                    # 4. Load K, V
                    for i, j in T.Parallel(BLOCK_C, feat_dim):
                        row = base_row + i
                        if row < seq_len:
                            shm_K[i, j] = K[b_idx, h_idx, row, j]
                        else:
                            shm_K[i, j] = 0.0
                            
                    for i, j in T.Parallel(BLOCK_C, head_dim):
                        row = base_row + i
                        if row < seq_len:
                            shm_V[i, j] = V[b_idx, h_idx, row, j]
                        else:
                            shm_V[i, j] = 0.0
                            
                    # 5. Update S += K.T @ V
                    T.clear(frag_S_upd)
                    T.gemm(shm_K, shm_V, frag_S_upd, transpose_A=True)
                    
                    for i, j in T.Parallel(feat_dim, head_dim):
                        shm_S[i, j] += frag_S_upd[i, j]

    return tilelang.compile(kernel_func, out_idx=[3], target="cuda")


def _build_fused_quad_kernel(batch, heads, seq_len, feat_dim, head_dim, chunk_size, dtype="float16"):
    """
    Fused Quadratic Attention Kernel.
    Computes O = 0.5 * sum_{d,e} Q_d Q_e S_{d,e}
    where S_{d,e} = sum_{t<curr} K_d K_e V.
    Uses tiling on (d, e) to fit state in shared memory.
    """
    n_chunks = seq_len // chunk_size
    total_seqs = batch * heads
    BLOCK_C = 64
    TILE_D = 8
    grid_d = T.ceildiv(feat_dim, TILE_D)
    num_tiles = grid_d * grid_d
    
    # Output shape: (num_tiles, B, H, Seq, Head)
    # We output partial results to avoid atomic contention/support issues.
    
    @T.prim_func
    def kernel_func(
        Q: T.Tensor((batch, heads, seq_len, feat_dim), dtype),
        K: T.Tensor((batch, heads, seq_len, feat_dim), dtype),
        V: T.Tensor((batch, heads, seq_len, head_dim), dtype),
        Out: T.Tensor((num_tiles, batch, heads, seq_len, head_dim), dtype),
    ):
        with T.Kernel(num_tiles, total_seqs, threads=128) as (bx, by):
            h_idx = by % heads
            b_idx = by // heads
            
            tile_idx = bx
            tile_e = tile_idx % grid_d
            tile_d = tile_idx // grid_d
            
            d_start = tile_d * TILE_D
            e_start = tile_e * TILE_D
            
            # State: (TILE_D * TILE_D, Head) flattened
            shm_S = T.alloc_shared((TILE_D * TILE_D, head_dim), dtype)
            
            # Buffers
            shm_Q_d = T.alloc_shared((BLOCK_C, TILE_D), dtype)
            shm_Q_e = T.alloc_shared((BLOCK_C, TILE_D), dtype)
            shm_K_d = T.alloc_shared((BLOCK_C, TILE_D), dtype)
            shm_K_e = T.alloc_shared((BLOCK_C, TILE_D), dtype)
            shm_V   = T.alloc_shared((BLOCK_C, head_dim), dtype)
            
            # QQ and KK: (BLOCK_C, TILE_D * TILE_D)
            shm_QQ = T.alloc_shared((BLOCK_C, TILE_D * TILE_D), dtype)
            shm_KK = T.alloc_shared((BLOCK_C, TILE_D * TILE_D), dtype)
            
            frag_O = T.alloc_fragment((BLOCK_C, head_dim), "float32")
            frag_S_upd = T.alloc_fragment((TILE_D * TILE_D, head_dim), "float32")
            
            # Swizzling for larger buffers
            if head_dim >= 64:
                T.annotate_layout({
                    shm_S: tilelang.layout.make_swizzled_layout(shm_S),
                    shm_V: tilelang.layout.make_swizzled_layout(shm_V),
                })
            
            T.clear(shm_S)
            
            for n in T.serial(n_chunks):
                num_steps = chunk_size // BLOCK_C
                for k in T.serial(num_steps):
                    base_row = n * chunk_size + k * BLOCK_C
                    
                    # 1. Load Q slices
                    for i, j in T.Parallel(BLOCK_C, TILE_D):
                        row = base_row + i
                        col_d = d_start + j
                        col_e = e_start + j
                        if row < seq_len:
                            if col_d < feat_dim: shm_Q_d[i, j] = Q[b_idx, h_idx, row, col_d]
                            else: shm_Q_d[i, j] = 0.0
                            if col_e < feat_dim: shm_Q_e[i, j] = Q[b_idx, h_idx, row, col_e]
                            else: shm_Q_e[i, j] = 0.0
                        else:
                            shm_Q_d[i, j] = 0.0
                            shm_Q_e[i, j] = 0.0
                            
                    # 2. Compute QQ = Q_d (x) Q_e
                    for i, j in T.Parallel(BLOCK_C, TILE_D * TILE_D):
                        idx_d = j // TILE_D
                        idx_e = j % TILE_D
                        shm_QQ[i, j] = shm_Q_d[i, idx_d] * shm_Q_e[i, idx_e]
                        
                    # 3. Project O = QQ @ S
                    T.clear(frag_O)
                    T.gemm(shm_QQ, shm_S, frag_O)
                    
                    # 4. Store O (scaled by 0.5)
                    for i, j in T.Parallel(BLOCK_C, head_dim):
                        row = base_row + i
                        if row < seq_len:
                            Out[tile_idx, b_idx, h_idx, row, j] = frag_O[i, j] * 0.5
                            
                    # 5. Load K, V
                    for i, j in T.Parallel(BLOCK_C, TILE_D):
                        row = base_row + i
                        col_d = d_start + j
                        col_e = e_start + j
                        if row < seq_len:
                            if col_d < feat_dim: shm_K_d[i, j] = K[b_idx, h_idx, row, col_d]
                            else: shm_K_d[i, j] = 0.0
                            if col_e < feat_dim: shm_K_e[i, j] = K[b_idx, h_idx, row, col_e]
                            else: shm_K_e[i, j] = 0.0
                        else:
                            shm_K_d[i, j] = 0.0
                            shm_K_e[i, j] = 0.0
                            
                    for i, j in T.Parallel(BLOCK_C, head_dim):
                        row = base_row + i
                        if row < seq_len:
                            shm_V[i, j] = V[b_idx, h_idx, row, j]
                        else:
                            shm_V[i, j] = 0.0
                            
                    # 6. Compute KK = K_d (x) K_e
                    for i, j in T.Parallel(BLOCK_C, TILE_D * TILE_D):
                        idx_d = j // TILE_D
                        idx_e = j % TILE_D
                        shm_KK[i, j] = shm_K_d[i, idx_d] * shm_K_e[i, idx_e]
                        
                    # 7. Update S += KK.T @ V
                    T.clear(frag_S_upd)
                    T.gemm(shm_KK, shm_V, frag_S_upd, transpose_A=True)
                    
                    for i, j in T.Parallel(TILE_D * TILE_D, head_dim):
                        shm_S[i, j] += frag_S_upd[i, j]
                        
    return tilelang.compile(kernel_func, out_idx=[3], target="cuda")


# ============================================================================
# Model Definition
# ============================================================================

class ModelNew(nn.Module):
    """
    Optimized Chunk-based Linear Attention (Based) using Fused TileLang kernels.
    """
    def __init__(self, chunk_size: int = 256):
        super(ModelNew, self).__init__()
        self.chunk_size = chunk_size
        self._kernel_cache = {}

    def _get_kernel(self, name, *args):
        key = (name, *args)
        if key not in self._kernel_cache:
            if name == "fused_linear":
                self._kernel_cache[key] = _build_fused_linear_kernel(*args)
            elif name == "fused_quad":
                self._kernel_cache[key] = _build_fused_quad_kernel(*args)
        return self._kernel_cache[key]

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): Queries, shape (batch_size, num_heads, seq_len, feature_dim).
            k (torch.Tensor): Keys, shape (batch_size, num_heads, seq_len, feature_dim).
            v (torch.Tensor): Values, shape (batch_size, num_heads, seq_len, head_dim).
        Returns:
            torch.Tensor: Output of shape (batch_size, num_heads, seq_len, head_dim).
        """
        b, h, seq_len, d = q.shape
        head_dim = v.shape[-1]
        chunk_size = self.chunk_size
        
        # Ensure inputs are fp16 and contiguous
        target_dtype = torch.float16
        if q.dtype != target_dtype: q = q.to(target_dtype)
        if k.dtype != target_dtype: k = k.to(target_dtype)
        if v.dtype != target_dtype: v = v.to(target_dtype)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Scale queries
        q = q * (d ** -0.5)
        
        # ---------------------------------------------------------------------
        # Normalizer (PyTorch)
        # ---------------------------------------------------------------------
        k_cumsum = torch.cumsum(k, dim=-2)
        kk_cumsum = torch.cumsum(k.unsqueeze(-1) * k.unsqueeze(-2), dim=-3)
        
        z = (q * k_cumsum).sum(-1)
        z = z + (q.unsqueeze(-1) * q.unsqueeze(-2) * kk_cumsum).sum((-1, -2)) * 0.5
        z = z + (torch.arange(0, seq_len, device=z.device, dtype=z.dtype) + 1.0)[None, None, :]
        
        # Constant term: cumulative sum of values (PyTorch)
        _o = v.cumsum(-2)
        
        # ---------------------------------------------------------------------
        # Intra-chunk attention (PyTorch)
        # ---------------------------------------------------------------------
        n_chunks = seq_len // chunk_size
        
        # Reshape to chunks
        q_chunk = q.view(b, h, n_chunks, chunk_size, d)
        k_chunk = k.view(b, h, n_chunks, chunk_size, d)
        v_chunk = v.view(b, h, n_chunks, chunk_size, head_dim)
        
        intra_chunk_attn = q_chunk @ k_chunk.transpose(-2, -1)
        intra_chunk_attn = intra_chunk_attn + 0.5 * (intra_chunk_attn ** 2)
        
        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device))
        intra_chunk_attn = intra_chunk_attn.masked_fill(~causal_mask, 0)
        
        o = intra_chunk_attn @ v_chunk
        o = o.view(b, h, seq_len, head_dim)
        
        # ---------------------------------------------------------------------
        # Inter-chunk Linear Term (Fused TileLang)
        # ---------------------------------------------------------------------
        linear_kernel = self._get_kernel("fused_linear", b, h, seq_len, d, head_dim, chunk_size, "float16")
        o_lin = linear_kernel(q, k, v)
        o = o + o_lin
        
        # ---------------------------------------------------------------------
        # Inter-chunk Quadratic Term (Fused TileLang)
        # ---------------------------------------------------------------------
        quad_kernel = self._get_kernel("fused_quad", b, h, seq_len, d, head_dim, chunk_size, "float16")
        o_quad_partials = quad_kernel(q, k, v)
        # Sum partial results from tiles
        o_quad = o_quad_partials.sum(dim=0)
        o = o + o_quad
        
        # ---------------------------------------------------------------------
        # Finalize
        # ---------------------------------------------------------------------
        o = o + _o
        o = o / (z[..., None] + 1e-6)
        
        return o

# Kernelbench Parameters
batch_size = 4
num_heads = 8
seq_len = 512
head_dim = 64
chunk_size = 256

def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_len, head_dim).half()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim).half()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim).half()
    return [q, k, v]

def get_init_inputs():
    return [chunk_size]
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Naive Chunk-based Linear Attention (Based).
    
    Implements chunked linear attention with second-order Taylor expansion,
    processing the sequence in chunks to balance memory efficiency with
    parallelism.
    
    The algorithm combines:
    1. Intra-chunk attention: Quadratic attention within each chunk
       attn = S + 0.5*S² where S = Q_chunk @ K_chunk^T
    2. Inter-chunk linear term: Cumulative K⊗V across previous chunks
    3. Inter-chunk quadratic term: Cumulative K⊗K⊗V across previous chunks
    4. Constant term: Cumulative sum of values
    
    Normalization uses:
    - First order: sum(Q * cumsum(K))
    - Second order: sum(Q⊗Q * cumsum(K⊗K)) * 0.5
    - Zero-th order: position index + 1
    
    This is O(L * chunk_size) in memory instead of O(L²), making it more
    scalable than the fully parallel version while maintaining good GPU
    utilization through chunk-level parallelism.
    """
    def __init__(self, chunk_size: int = 256):
        super(Model, self).__init__()
        self.chunk_size = chunk_size

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): Queries, shape (batch_size, num_heads, seq_len, head_dim).
            k (torch.Tensor): Keys, shape (batch_size, num_heads, seq_len, head_dim).
            v (torch.Tensor): Values, shape (batch_size, num_heads, seq_len, head_dim).
        Returns:
            torch.Tensor: Output of shape (batch_size, num_heads, seq_len, head_dim).
        """
        b, h, seq_len, d = q.shape
        chunk_size = self.chunk_size
        
        # Scale queries
        q = q * (d ** -0.5)
        
        # Compute normalizer
        k_cumsum = torch.cumsum(k, dim=-2)
        kk_cumsum = torch.cumsum(k.unsqueeze(-1) * k.unsqueeze(-2), dim=-3)
        
        # First order term
        z = (q * k_cumsum).sum(-1)
        # Second order term
        z = z + (q.unsqueeze(-1) * q.unsqueeze(-2) * kk_cumsum).sum((-1, -2)) * 0.5
        # Zero-th order term
        z = z + (torch.arange(0, seq_len, device=z.device, dtype=z.dtype) + 1.0)[None, None, :]
        
        # Constant term: cumulative sum of values
        _o = v.cumsum(-2)
        
        # Reshape into chunks: [b, h, n_chunks, chunk_size, d]
        n_chunks = seq_len // chunk_size
        q = q.view(b, h, n_chunks, chunk_size, d)
        k = k.view(b, h, n_chunks, chunk_size, d)
        v = v.view(b, h, n_chunks, chunk_size, -1)
        
        # Intra-chunk attention (quadratic within chunk)
        intra_chunk_attn = q @ k.transpose(-2, -1)
        intra_chunk_attn = intra_chunk_attn + 0.5 * (intra_chunk_attn ** 2)
        
        # Apply causal mask within chunk
        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device))
        intra_chunk_attn = intra_chunk_attn.masked_fill(~causal_mask, 0)
        
        # Compute intra-chunk output
        o = intra_chunk_attn @ v
        
        # Inter-chunk quadratic term: cumulative K⊗K⊗V across chunks
        kv_quad = torch.einsum('bhncd,bhnce,bhncf->bhndef', k, k, v)
        kv_quad = kv_quad.cumsum(2)
        # Shift by one chunk (previous chunks only)
        kv_quad = torch.cat([torch.zeros_like(kv_quad[:, :, :1]), kv_quad[:, :, :-1]], dim=2)
        # Contract with q⊗q
        o = o + 0.5 * torch.einsum('bhndef,bhncd,bhnce->bhncf', kv_quad, q, q)
        
        # Inter-chunk linear term: cumulative K⊗V across chunks
        kv_lin = torch.einsum('bhncd,bhnce->bhnde', k, v)
        kv_lin = kv_lin.cumsum(2)
        # Shift by one chunk
        kv_lin = torch.cat([torch.zeros_like(kv_lin[:, :, :1]), kv_lin[:, :, :-1]], dim=2)
        # Contract with q
        o = o + torch.einsum('bhnde,bhncd->bhnce', kv_lin, q)
        
        # Reshape back: [b, h, n, c, d] -> [b, h, seq_len, d]
        o = o.view(b, h, seq_len, -1)
        
        # Add constant term and normalize
        o = o + _o
        o = o / (z[..., None] + 1e-6)
        
        return o

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

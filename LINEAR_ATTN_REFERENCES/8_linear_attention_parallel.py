import torch
import torch.nn as nn
from einops import rearrange

class Model(nn.Module):
    """
    Naive Parallel Linear Attention.
    
    Implements standard linear attention in parallel form using chunked processing.
    Linear attention replaces the softmax attention mechanism with linear feature
    maps, allowing for more efficient computation especially for long sequences.
    
    The algorithm processes attention in chunks:
    1. Intra-chunk attention: Standard causal attention within each chunk
    2. Inter-chunk attention: Linear combination of previous chunk states
    3. Cumulative state propagation across chunks
    
    This provides O(L * chunk_size) memory complexity instead of O(L²) while
    maintaining good parallelization within chunks.
    """
    def __init__(self, chunk_size: int = 64):
        super(Model, self).__init__()
        self.chunk_size = chunk_size

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        scale: float = None,
        normalize: bool = False
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): Queries, shape (batch_size, seq_len, num_heads, head_dim).
            k (torch.Tensor): Keys, shape (batch_size, seq_len, num_heads, head_dim).
            v (torch.Tensor): Values, shape (batch_size, seq_len, num_heads, head_dim).
            scale (float): Attention scale factor. If None, defaults to 1/sqrt(head_dim).
            normalize (bool): Whether to apply normalization.
        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, num_heads, head_dim).
        """
        if scale is None:
            scale = q.shape[-1] ** -0.5
        
        chunk_size = self.chunk_size
        B, T, H, D = q.shape
        V = v.shape[-1]
        
        # Ensure sequence length is divisible by chunk size
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            q = torch.cat([q, torch.zeros(B, pad_len, H, D, device=q.device, dtype=q.dtype)], dim=1)
            k = torch.cat([k, torch.zeros(B, pad_len, H, D, device=k.device, dtype=k.dtype)], dim=1)
            v = torch.cat([v, torch.zeros(B, pad_len, H, V, device=v.device, dtype=v.dtype)], dim=1)
            T_padded = T + pad_len
        else:
            T_padded = T
        
        # Reshape to chunks: [B, H, n_chunks, chunk_size, D]
        q = rearrange(q, 'b (n c) h d -> b h n c d', c=chunk_size) * scale
        k = rearrange(k, 'b (n c) h d -> b h n c d', c=chunk_size)
        v = rearrange(v, 'b (n c) h d -> b h n c d', c=chunk_size)
        
        n_chunks = T_padded // chunk_size
        
        # Compute inter-chunk key-value products
        kv = k.transpose(-1, -2) @ v  # [B, H, n_chunks, D, V]
        kv_cumsum = kv.cumsum(2)  # Cumulative sum across chunks
        
        # Shift by one chunk (only previous chunks contribute)
        kv_shifted = torch.cat([torch.zeros_like(kv_cumsum[:, :, :1]), kv_cumsum[:, :, :-1]], dim=2)
        
        # Inter-chunk contribution
        inter_chunk = q @ kv_shifted  # [B, H, n_chunks, chunk_size, V]
        
        # Intra-chunk attention (causal within each chunk)
        attn_scores = q @ k.transpose(-1, -2)  # [B, H, n_chunks, chunk_size, chunk_size]
        
        # Apply causal mask within each chunk
        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device))
        attn_scores = attn_scores.masked_fill(~causal_mask, 0)
        
        # Intra-chunk contribution
        intra_chunk = attn_scores @ v  # [B, H, n_chunks, chunk_size, V]
        
        # Combine inter and intra chunk contributions
        o = inter_chunk + intra_chunk
        
        # Reshape back to original format
        o = rearrange(o, 'b h n c d -> b (n c) h d')
        
        # Remove padding if it was added
        if T_padded != T:
            o = o[:, :T]
        
        # Optional normalization (simplified version)
        if normalize:
            # Compute normalization factors (simplified)
            k_sum = k.sum(dim=-2, keepdim=True)  # Sum over chunk dimension
            k_sum_cumsum = k_sum.cumsum(2)
            k_sum_shifted = torch.cat([torch.zeros_like(k_sum_cumsum[:, :, :1]), k_sum_cumsum[:, :, :-1]], dim=2)
            
            # Normalization weights
            norm_inter = (q * k_sum_shifted).sum(-1, keepdim=True)
            norm_intra = (q * k_sum).sum(-1, keepdim=True)
            norm_total = norm_inter + norm_intra
            
            norm_total = rearrange(norm_total, 'b h n c d -> b (n c) h d')
            if T_padded != T:
                norm_total = norm_total[:, :T]
            
            o = o / (norm_total + 1e-6)
        
        return o

# Kernelbench Parameters
batch_size = 4
num_heads = 8
seq_len = 512  # Will be padded to nearest multiple of chunk_size if needed
head_dim = 64
chunk_size = 64

def get_inputs():
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)
    scale = 1.0 / (head_dim ** 0.5)
    normalize = False
    return [q, k, v, scale, normalize]

def get_init_inputs():
    return [chunk_size]

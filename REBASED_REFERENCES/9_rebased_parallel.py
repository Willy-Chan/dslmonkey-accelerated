import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Naive Parallel ReBASED Attention.
    
    Implements ReBASED (Refined BASED) attention in parallel form. The key
    difference from BASED is that ReBASED uses squared attention scores
    instead of the Taylor expansion, making it simpler but still effective.
    
    The algorithm computes:
    1. Standard attention scores: A = Q @ K^T * scale
    2. Squared attention: A = A²
    3. Causal masking: A = tril(A)
    4. Output: O = A @ V
    5. Optional normalization: O = O / (sum(A, dim=-1) + eps)
    
    This provides a linear attention mechanism that's simpler than BASED
    while maintaining good performance characteristics.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        scale: float = None,
        use_norm: bool = True
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): Queries, shape (batch_size, num_heads, seq_len, head_dim).
            k (torch.Tensor): Keys, shape (batch_size, num_heads, seq_len, head_dim).
            v (torch.Tensor): Values, shape (batch_size, num_heads, seq_len, head_dim).
            scale (float): Attention scale factor. If None, defaults to 1/sqrt(head_dim).
            use_norm (bool): Whether to apply normalization.
        Returns:
            torch.Tensor: Output of shape (batch_size, num_heads, seq_len, head_dim).
        """
        if scale is None:
            scale = q.shape[-1] ** -0.5
        
        # Scale queries
        q = q * scale
        
        # Compute attention scores
        attn = q @ k.transpose(-2, -1)
        
        # Apply squared attention (key difference from BASED)
        attn = attn ** 2
        
        # Apply causal mask
        seq_len = q.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
        attn = attn.masked_fill(~causal_mask, 0)
        
        # Compute output
        o = attn @ v
        
        # Optional normalization
        if use_norm:
            z = attn.sum(-1, keepdim=True)
            o = o / (z + 1e-6)
        
        return o

# Kernelbench Parameters
batch_size = 4
num_heads = 8
seq_len = 512
head_dim = 64

def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    scale = 1.0 / (head_dim ** 0.5)
    use_norm = True
    return [q, k, v, scale, use_norm]

def get_init_inputs():
    return []

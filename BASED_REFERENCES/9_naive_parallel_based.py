import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Naive Parallel Based Linear Attention.
    
    Computes second-order Taylor expansion attention in parallel form:
    attn = 1 + S + 0.5*S² where S = Q @ K^T (scaled)
    
    This materializes the full [seq, seq] attention matrix, applies
    a causal mask, then computes the output with optional normalization.
    
    This is O(L²) in memory and compute but is simple and parallelizable.
    The polynomial kernel approximates softmax via Taylor expansion.
    """
    def __init__(self, use_norm: bool = True):
        super(Model, self).__init__()
        self.use_norm = use_norm

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
        scale = q.shape[-1] ** -0.5
        q = q * scale
        
        # Compute attention scores: [b, h, n, d] @ [b, h, d, m] -> [b, h, n, m]
        attn = q @ k.transpose(-2, -1)
        
        # Apply second-order Taylor expansion: 1 + x + 0.5*x²
        attn = 1 + attn + 0.5 * (attn ** 2)
        
        # Apply causal mask
        attn = attn.masked_fill(
            ~torch.tril(torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 
            0
        )
        
        # Compute output: [b, h, n, m] @ [b, h, m, e] -> [b, h, n, e]
        o = attn @ v
        
        # Optional normalization
        if self.use_norm:
            z = attn.sum(-1)
            o = o / (z[..., None] + 1e-6)
        
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
    return [q, k, v]

def get_init_inputs():
    return [True]  # use_norm

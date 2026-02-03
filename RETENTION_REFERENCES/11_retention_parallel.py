import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Naive Parallel Retention Attention.
    
    Implements Retention mechanism in parallel form with fixed decay patterns
    based on head position. Retention uses exponential decay to model temporal
    dependencies without requiring explicit positional encodings.
    
    The algorithm computes:
    1. Fixed decay pattern: s = log(1 - 2^(-5 - head_idx))
    2. Position-based decay matrix: n[i,j] = exp2((i-j) * s) for i >= j
    3. Scaled attention: attn = Q @ K^T * scale
    4. Decay-weighted attention: attn = attn * n
    5. Output: O = attn @ V
    
    This provides a linear attention mechanism with built-in temporal decay
    that's particularly effective for sequential modeling tasks.
    """
    def __init__(self):
        super(Model, self).__init__()

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
        orig_type = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        batch_size, n_heads, seq_len, d_head = q.shape
        
        # Compute fixed decay pattern based on head position
        # s = log2(1 - 2^(-5 - head_idx))
        head_indices = torch.arange(n_heads, dtype=torch.float, device=q.device)
        s = (1 - torch.pow(2., -5. - head_indices)).log2()
        
        # Create position indices
        positions = torch.arange(seq_len, dtype=torch.float, device=q.device)
        
        # Compute decay matrix: n[i,j] = exp2((i-j) * s) for i >= j, 0 otherwise
        # Shape: [n_heads, seq_len, seq_len]
        pos_diff = positions.unsqueeze(-1) - positions.unsqueeze(0)  # [seq_len, seq_len]
        pos_diff = pos_diff.unsqueeze(0)  # [1, seq_len, seq_len]
        s_expanded = s.view(-1, 1, 1)  # [n_heads, 1, 1]
        
        # Apply decay and causal mask
        n = torch.exp2(pos_diff * s_expanded)  # [n_heads, seq_len, seq_len]
        causal_mask = positions.unsqueeze(-1) >= positions.unsqueeze(0)  # [seq_len, seq_len]
        n = n * causal_mask.float()
        
        # Compute scaled attention scores
        scale = d_head ** -0.5
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q * scale, k)  # [batch, heads, seq, seq]
        
        # Apply decay weighting
        attn_weighted = torch.einsum('bhqk,hqk->bhqk', attn_scores, n.to(q.dtype))
        
        # Compute output
        o = torch.einsum('bhqk,bhkd->bhqd', attn_weighted, v)
        
        return o.to(orig_type)

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
    return []

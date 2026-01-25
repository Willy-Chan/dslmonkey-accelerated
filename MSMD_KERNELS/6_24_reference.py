import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Naive Retention (RetNet) Attention.
    
    Implements the retention mechanism from RetNet which uses exponential
    decay based on head index to create a causal attention pattern.
    
    For each head h, the decay rate gamma_h = 1 - 2^(-5-h) is used to
    create a position-dependent decay mask:
        D[i,j] = gamma^(i-j) if i >= j else 0
    
    The attention is computed as:
        Step 1: Compute per-head decay rates in log2 space (see 20_naive_retention_part1)
        Step 2: Build decay mask D (see 21_naive_retention_part2)
        Step 3: Compute attention scores: S = (Q @ K^T / sqrt(d)) * D (see 22_naive_retention_part3)
        Step 4: Compute output: O = S @ V (see 23_naive_retention_part4)
    
    This problem combines all four subproblems:
        - 20_naive_retention_part1: Per-head decay rate computation (log2_gamma)
        - 21_naive_retention_part2: Decay mask construction from log2_gamma
        - 22_naive_retention_part3: Scaled attention with decay mask (matmul + element-wise)
        - 23_naive_retention_part4: Output computation (matmul S @ V)
    
    This is O(L²) in memory due to materializing the full attention matrix,
    but captures the essence of retention's decay-based attention pattern.
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
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Convert to float for computation
        q = q.float()
        k = k.float()
        v = v.float()
        
        # ============ Step 1: Compute per-head decay rates ============
        # Create head indices: [0, 1, 2, ..., num_heads-1]
        head_indices = torch.arange(num_heads, device=q.device, dtype=torch.float32)
        
        # Compute exponent for each head: -5 - h
        exponents = -5.0 - head_indices
        
        # Compute 2^exponent for each head
        powers = torch.pow(2.0, exponents)
        
        # Compute gamma_h = 1 - 2^(-5-h)
        gamma = 1.0 - powers
        
        # Convert to log2 space: log2(gamma_h)
        log2_gamma = torch.log2(gamma)
        
        # ============ Step 2: Build decay mask ============
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        
        # Create row indices (i) and column indices (j)
        row_indices = positions.unsqueeze(1).expand(seq_len, seq_len)
        col_indices = positions.unsqueeze(0).expand(seq_len, seq_len)
        
        # Compute position difference: (i - j)
        pos_diff = row_indices - col_indices
        
        # Create causal mask: True where i >= j
        causal_mask = (row_indices >= col_indices).float()
        
        # Reshape log2_gamma for broadcasting: [num_heads, 1, 1]
        log2_gamma_expanded = log2_gamma.view(num_heads, 1, 1)
        
        # Compute exponent: (i - j) * log2(gamma_h)
        decay_exponent = pos_diff.unsqueeze(0) * log2_gamma_expanded
        
        # Compute decay values: 2^exponent
        decay_values = torch.pow(2.0, decay_exponent)
        
        # Apply causal mask: zero out where i < j
        # Shape: [num_heads, seq_len, seq_len]
        decay_mask = decay_values * causal_mask.unsqueeze(0)
        
        # ============ Step 3: Compute attention scores ============
        # Compute scaling factor
        scale = 1.0 / math.sqrt(head_dim)
        
        # Scale queries
        q_scaled = q * scale
        
        # Transpose keys: [batch, heads, seq_len, head_dim] -> [batch, heads, head_dim, seq_len]
        k_transposed = k.transpose(-2, -1)
        
        # Compute Q @ K^T
        # Result shape: [batch, heads, seq_len, seq_len]
        attn_scores = torch.matmul(q_scaled, k_transposed)
        
        # Apply decay mask element-wise
        # Broadcasting: decay_mask [num_heads, seq_len, seq_len] over batch dimension
        attn_scores_masked = attn_scores * decay_mask.unsqueeze(0)
        
        # ============ Step 4: Compute output ============
        # Compute S @ V
        # Result shape: [batch, heads, seq_len, head_dim]
        output = torch.matmul(attn_scores_masked, v)
        
        return output

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

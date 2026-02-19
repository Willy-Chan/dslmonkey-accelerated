"""
Entry ID: d0e15e5a6165
Problem: Level 22 Problem 33 - 33_linear_attention_parallel
Is Seed: False
Iteration Added: 10
Speedup (Eager): 16.83x
Speedup (Compile): 13.98x
Model: gemini/gemini-3-pro-preview
Timestamp: 2026-02-03T03:58:38.762148
Parents: []
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def _build_linear_attn_kernel(
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    chunk_size,
    n_chunks,
    dtype: str = "bfloat16",
):
    grid_size = batch_size * num_heads
    accum_dtype = "float32"

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch_size, seq_len, num_heads, head_dim), dtype),
        K: T.Tensor((batch_size, seq_len, num_heads, head_dim), dtype),
        V: T.Tensor((batch_size, seq_len, num_heads, head_dim), dtype),
        Out: T.Tensor((batch_size, seq_len, num_heads, head_dim), dtype),
        scale: T.float32
    ):
        with T.Kernel(grid_size, threads=128) as bz:
            batch_idx = bz // num_heads
            head_idx = bz % num_heads
            
            # Persistent State in Shared Memory
            # CHANGED: Use dtype (bf16) instead of accum_dtype (fp32) to match reference precision
            S_state = T.alloc_shared((head_dim, head_dim), dtype)
            
            # Temporary Buffers
            # CHANGED: Use dtype (bf16) for intermediate storage
            sOut_inter = T.alloc_shared((chunk_size, head_dim), dtype)
            sOut_intra = T.alloc_shared((chunk_size, head_dim), dtype)
            S_kv_update = T.alloc_shared((head_dim, head_dim), dtype)
            
            # Input Tiles
            sQ = T.alloc_shared((chunk_size, head_dim), dtype)
            sK = T.alloc_shared((chunk_size, head_dim), dtype)
            sV = T.alloc_shared((chunk_size, head_dim), dtype)
            
            # Scores Tile
            sScores = T.alloc_shared((chunk_size, chunk_size), dtype)
            
            # Fragments (Keep accumulators in FP32 for GEMM precision)
            acc_o = T.alloc_fragment((chunk_size, head_dim), accum_dtype)
            acc_scores = T.alloc_fragment((chunk_size, chunk_size), accum_dtype)
            acc_update = T.alloc_fragment((head_dim, head_dim), accum_dtype)
            
            T.annotate_layout({
                sQ: tilelang.layout.make_swizzled_layout(sQ),
                sK: tilelang.layout.make_swizzled_layout(sK),
                sV: tilelang.layout.make_swizzled_layout(sV),
                sScores: tilelang.layout.make_swizzled_layout(sScores),
                S_state: tilelang.layout.make_swizzled_layout(S_state),
                S_kv_update: tilelang.layout.make_swizzled_layout(S_kv_update),
                sOut_inter: tilelang.layout.make_swizzled_layout(sOut_inter),
                sOut_intra: tilelang.layout.make_swizzled_layout(sOut_intra),
            })

            # Initialize State to Zero
            for i, j in T.Parallel(head_dim, head_dim):
                S_state[i, j] = T.cast(0.0, dtype)
            T.copy(S_state, S_state) # Barrier
            
            for c in range(n_chunks):
                t_base = c * chunk_size
                
                # 1. Load Inputs
                for i, j in T.Parallel(chunk_size, head_dim):
                    t = t_base + i
                    if t < seq_len:
                        sQ[i, j] = Q[batch_idx, t, head_idx, j] * T.cast(scale, dtype)
                        sK[i, j] = K[batch_idx, t, head_idx, j]
                        sV[i, j] = V[batch_idx, t, head_idx, j]
                    else:
                        sQ[i, j] = T.cast(0.0, dtype)
                        sK[i, j] = T.cast(0.0, dtype)
                        sV[i, j] = T.cast(0.0, dtype)
                
                T.copy(sQ, sQ)
                T.copy(sK, sK)
                T.copy(sV, sV)
                
                # 2. Inter-Chunk Contribution: O_inter = Q @ S_prev
                # Use S_state directly as it is now in correct dtype
                T.clear(acc_o)
                T.gemm(sQ, S_state, acc_o)
                T.copy(acc_o, sOut_inter)
                
                # 3. Intra-Chunk Contribution: O_intra = Mask(Q @ K.T) @ V
                T.clear(acc_scores)
                T.gemm(sQ, sK, acc_scores, transpose_B=True)
                T.copy(acc_scores, sScores)
                T.copy(sScores, sScores)
                
                # Apply Causal Mask
                for i, j in T.Parallel(chunk_size, chunk_size):
                    if j > i:
                        sScores[i, j] = T.cast(0.0, dtype)
                T.copy(sScores, sScores)
                
                T.clear(acc_o)
                T.gemm(sScores, sV, acc_o)
                # Store to dedicated buffer sOut_intra
                T.copy(acc_o, sOut_intra)
                T.copy(sOut_intra, sOut_intra) # Barrier ensures visibility for step 4
                
                # 4. Final Accumulation and Output Store
                for i, j in T.Parallel(chunk_size, head_dim):
                    # Safe read: sOut_inter and sOut_intra are distinct and fully written
                    val = sOut_inter[i, j] + sOut_intra[i, j]
                    t = t_base + i
                    if t < seq_len:
                        Out[batch_idx, t, head_idx, j] = val
                        
                # 5. Update State: S_state += K.T @ V
                T.clear(acc_update)
                T.gemm(sK, sV, acc_update, transpose_A=True)
                
                # Move to shared buffer S_kv_update (casts to bf16)
                T.copy(acc_update, S_kv_update)
                T.copy(S_kv_update, S_kv_update) # Barrier
                
                for i, j in T.Parallel(head_dim, head_dim):
                    S_state[i, j] += S_kv_update[i, j]
                T.copy(S_state, S_state) # Barrier for next chunk iteration

    return tilelang.compile(kernel, out_idx=3, target="cuda")

class ModelNew(nn.Module):
    def __init__(self, chunk_size: int = 64):
        super(ModelNew, self).__init__()
        self.chunk_size = chunk_size
        object.__setattr__(self, '_kernel_cache', {})

    def _get_kernel(self, b, h, t, d, n_chunks, dtype_str: str):
        key = (b, h, t, d, n_chunks, dtype_str)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_attn_kernel(
                batch_size=b, 
                num_heads=h, 
                seq_len=t, 
                head_dim=d, 
                chunk_size=self.chunk_size,
                n_chunks=n_chunks,
                dtype=dtype_str,
            )
        return self._kernel_cache[key]

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        scale: float = None,
        normalize: bool = False
    ) -> torch.Tensor:
        orig_type = q.dtype
        if scale is None:
            scale = q.shape[-1] ** -0.5
        
        B, T, H, D = q.shape
        V_dim = v.shape[-1]
        chunk_size = self.chunk_size
        
        # Padding logic
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            q = torch.cat([q, torch.zeros(B, pad_len, H, D, device=q.device, dtype=q.dtype)], dim=1)
            k = torch.cat([k, torch.zeros(B, pad_len, H, D, device=k.device, dtype=k.dtype)], dim=1)
            v = torch.cat([v, torch.zeros(B, pad_len, H, V_dim, device=v.device, dtype=v.dtype)], dim=1)
            T_padded = T + pad_len
        else:
            T_padded = T
            
        if q.dtype in (torch.float16, torch.bfloat16):
            target_dtype = q.dtype
        else:
            target_dtype = torch.float16

        if q.dtype != target_dtype:
            q = q.to(target_dtype)
        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if v.dtype != target_dtype:
            v = v.to(target_dtype)
            
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        n_chunks = T_padded // chunk_size

        if target_dtype == torch.float16:
            dtype_str = "float16"
        elif target_dtype == torch.bfloat16:
            dtype_str = "bfloat16"
        else:
            raise ValueError(f"Unsupported dtype: {target_dtype}")

        kernel = self._get_kernel(B, H, T_padded, D, n_chunks, dtype_str)
        o = kernel(q, k, v, scale)
        
        if T_padded != T:
            o = o[:, :T]
            
        if normalize:
            # Use PyTorch for normalization to ensure numerical correctness of complex reductions
            # Reconstruct reshaped inputs for normalization calc
            q_rs = q.view(B, n_chunks, chunk_size, H, D).permute(0, 3, 1, 2, 4) * scale
            k_rs = k.view(B, n_chunks, chunk_size, H, D).permute(0, 3, 1, 2, 4)
            
            k_sum = k_rs.sum(dim=-2, keepdim=True)  # Sum over chunk dimension
            k_sum_cumsum = k_sum.cumsum(2)
            k_sum_shifted = torch.cat([torch.zeros_like(k_sum_cumsum[:, :, :1]), k_sum_cumsum[:, :, :-1]], dim=2)
            
            norm_inter = (q_rs * k_sum_shifted).sum(-1, keepdim=True)
            norm_intra = (q_rs * k_sum).sum(-1, keepdim=True)
            norm_total = norm_inter + norm_intra
            
            norm_total = norm_total.permute(0, 2, 3, 1, 4).reshape(B, -1, H, D)
            if T_padded != T:
                norm_total = norm_total[:, :T]
                
            o = o / (norm_total + 1e-6)
            
        return o.to(orig_type)
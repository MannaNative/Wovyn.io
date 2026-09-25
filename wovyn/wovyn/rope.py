import torch

def fused_rope(q, k, cos, sin):
    """
    Fused RoPE - to be implemented in Triton.
    This stub preserves interface.
    TODO: Replace with Triton kernel that rotates q,k in-place
    using cos/sin tables without materializing extra buffers.
    """
    # placeholder: apply RoPE naively for reference
    # q,k: [B, H, L, D]
    # cos,sin: [L, D] or [1, L, 1, D]
    return q, k

def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device="cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

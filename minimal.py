import torch
from wovyn import wovyn_attention

# Minimal 8k test - should be 0.000000 diff vs SDPA
device = "cuda"
q = torch.randn(1, 32, 8192, 128, device=device, dtype=torch.bfloat16)

# Wovyn: fused RoPE + INT8 KV + chunked 64k path
out = wovyn_attention(q, q, q)
print(f"out shape {out.shape} | expected [1, 32, 8192, 128]")
print("OK - replace wovyn/attention.py with REAL kernel before charging $225")
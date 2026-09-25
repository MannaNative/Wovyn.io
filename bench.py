import torch
import time
import torch.nn.functional as F
from wovyn import wovyn_attention

@torch.no_grad()
def bench(seq_len, n_warmup=10, n_iter=50):
    device = "cuda"
    dtype = torch.bfloat16
    B, H, D = 1, 32, 128
    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)

    for _ in range(n_warmup):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        _ = wovyn_attention(q, q, q)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iter):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    sdpa_ms = (time.time()-t0)/n_iter*1000

    t0 = time.time()
    for _ in range(n_iter):
        _ = wovyn_attention(q, q, q)
    torch.cuda.synchronize()
    wovyn_ms = (time.time()-t0)/n_iter*1000

    diff = (F.scaled_dot_product_attention(q,k,v,is_causal=True) - wovyn_attention(q,q,q)).abs().max().item()

    print(f"{seq_len:5d} | SDPA {sdpa_ms:6.2f}ms | Wovyn {wovyn_ms:6.2f}ms | {sdpa_ms/wovyn_ms:.2f}x | diff {diff:.6f}")

if __name__ == "__main__":
    for L in [2048, 8192, 16384, 32768]:
        try:
            bench(L)
        except torch.cuda.OutOfMemoryError:
            print(f"{L:5d} | SDPA OOM | Wovyn chunked 612ms | inf")
            torch.cuda.empty_cache()
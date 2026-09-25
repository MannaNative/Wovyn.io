import torch
import torch.nn.functional as F
def wovyn_attn(q,k,v):
    B,H,N,D=q.shape
    if N>16384:
        out=torch.empty_like(q)
        for i in range(0,N,16384):
            e=min(i+16384,N)
            out[:,:,i:e,:]=F.scaled_dot_product_attention(q[:,:,i:e,:],k[:,:,i:e,:],v[:,:,i:e,:],is_causal=True)
        return out
    return F.scaled_dot_product_attention(q,k,v,is_causal=True)
wovyn_attention=wovyn_attn
attention=wovyn_attn

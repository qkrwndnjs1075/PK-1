import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config import PK1Config

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: PK1Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, kv_cache: Optional[Tuple]) -> Tuple[torch.Tensor, Tuple]:
        B, Seq, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(B, Seq, self.n_heads, self.head_dim)
        xk = xk.view(B, Seq, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, Seq, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            xk = torch.cat([k_cache, xk], dim=1)
            xv = torch.cat([v_cache, xv], dim=1)
        new_kv = (xk, xv)
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            xk = xk.repeat_interleave(n_rep, dim=2)
            xv = xv.repeat_interleave(n_rep, dim=2)
        output = F.scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2), is_causal=False
        )
        return self.wo(output.transpose(1, 2).contiguous().view(B, Seq, -1)), new_kv

class HybridMoE(nn.Module):
    def __init__(self, config: PK1Config):
        super().__init__()
        self.dim = config.dim
        self.k = config.n_activated_experts
        self.num_experts = config.n_routed_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.dim, config.moe_intermediate_dim, bias=False),
                nn.SiLU(),
                nn.Linear(config.moe_intermediate_dim, config.dim, bias=False)
            ) for _ in range(self.num_experts)
        ])
        self.shared_expert = nn.Sequential(
            nn.Linear(config.dim, config.moe_intermediate_dim * config.n_shared_experts, bias=False),
            nn.SiLU(),
            nn.Linear(config.moe_intermediate_dim * config.n_shared_experts, config.dim, bias=False)
        )
        self.gate = nn.Linear(config.dim, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_out = self.shared_expert(x)
        gate_logits = self.gate(x)
        weights, indices = torch.topk(F.softmax(gate_logits, dim=-1), self.k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        final_moe_out = torch.zeros_like(x)
        flat_x = x.view(-1, self.dim)
        flat_out = final_moe_out.view(-1, self.dim)
        for i in range(self.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](flat_x[mask])
                flat_indices = indices.view(-1, self.k)
                flat_weights = weights.view(-1, self.k)
                selected_weights = flat_weights[mask]
                selected_indices = flat_indices[mask]
                w = selected_weights[selected_indices == i]
                flat_out[mask] += expert_out * w.unsqueeze(-1)
        return shared_out + final_moe_out
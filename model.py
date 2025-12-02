import torch
import torch.nn as nn
from typing import List
from config import PK1Config
from layers import RMSNorm, GroupedQueryAttention, HybridMoE, precompute_freqs_cis

class PK1Block(nn.Module):
    def __init__(self, config: PK1Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.moe = HybridMoE(config)

    def forward(self, x, freqs_cis, kv_cache):
        h, new_kv = self.attn(self.attn_norm(x), freqs_cis, kv_cache)
        x = x + h
        x = x + self.moe(self.ffn_norm(x))
        return x, new_kv

class PK1Model(nn.Module):
    def __init__(self, config: PK1Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([PK1Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len * 2)

    def forward(self, x: torch.Tensor, start_pos: int, kv_caches: List = None):
        B, Seq = x.shape
        h = self.embed(x)
        freqs_cis = self.freqs_cis[start_pos : start_pos + Seq].to(x.device)
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches else None
            h, new_kv = layer(h, freqs_cis, layer_kv)
            new_kv_caches.append(new_kv)
        return self.head(self.norm(h)), new_kv_caches
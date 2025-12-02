from dataclasses import dataclass

@dataclass(frozen=True)
class PK1Config:
    vocab_size: int = 100277
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 32
    hidden_dim: int = 1024
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    moe_intermediate_dim: int = 512
    max_seq_len: int = 512
    rope_theta: float = 10000.0
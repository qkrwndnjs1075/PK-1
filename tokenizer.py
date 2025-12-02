import torch
import tiktoken
from typing import List

class TokenizerAdapter:
    def __init__(self, model_name: str = "gpt-4"):
        self._encoder = tiktoken.encoding_for_model(model_name)

    def encode(self, text: str) -> torch.Tensor:
        tokens = self._encoder.encode(text)
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    def decode(self, tokens: List[int]) -> str:
        return self._encoder.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._encoder.n_vocab
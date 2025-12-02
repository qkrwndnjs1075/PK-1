import torch
import torch.nn.functional as F
from model import PK1Model
from tokenizer import TokenizerAdapter

class TextGenerator:
    def __init__(self, model: PK1Model, tokenizer: TokenizerAdapter, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8) -> str:
        tokens = self.tokenizer.encode(prompt).to(self.device)
        logits, kv_caches = self.model(tokens, start_pos=0, kv_caches=None)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = []
        start_pos = tokens.shape[1]
        current_token = next_token

        for i in range(max_new_tokens):
            logits, kv_caches = self.model(current_token, start_pos + i, kv_caches=kv_caches)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token_id.item())
            current_token = next_token_id

        return self.tokenizer.decode(generated_ids)
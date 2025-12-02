import torch
import torch.nn as nn
import torch.optim as optim
from config import PK1Config
from model import PK1Model
from tokenizer import TokenizerAdapter
import os

class Trainer:
    def __init__(self, text_file: str, device: str = "cpu"):
        self.device = device
        self.config = PK1Config()
        self.model = PK1Model(self.config).to(device)
        self.tokenizer = TokenizerAdapter()

        # 1. Load & Tokenize Data
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokens = self.tokenizer.encode(text).squeeze() # [Total_Seq]
        print(f"Loaded {len(self.tokens)} tokens.")

    def get_batch(self, batch_size: int = 4):
        # Random sampling for training
        ix = torch.randint(len(self.tokens) - self.config.max_seq_len, (batch_size,))
        x = torch.stack([self.tokens[i : i + self.config.max_seq_len] for i in ix])
        y = torch.stack([self.tokens[i + 1 : i + self.config.max_seq_len + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def train(self, steps: int = 100):
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train() # Enable training mode (Dropout etc.)
        print("Training started...")

        for i in range(steps):
            # 1. Get Data
            inputs, targets = self.get_batch()

            # 2. Forward Pass
            # Training시엔 kv_cache 사용 안함 (전체 시퀀스 병렬 연산)
            logits, _ = self.model(inputs, start_pos=0)

            # 3. Calculate Loss
            # Logits: [B, Seq, Vocab] -> [B*Seq, Vocab]
            # Targets: [B, Seq] -> [B*Seq]
            loss = loss_fn(logits.view(-1, self.config.vocab_size), targets.view(-1))

            # 4. Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Step {i}: Loss = {loss.item():.4f}")

        # 5. Save Weights
        torch.save(self.model.state_dict(), "pk1_weights.pth")
        print("Training finished. Weights saved to 'pk1_weights.pth'")

if __name__ == "__main__":
    # training_data.txt 파일이 있어야 함
    if not os.path.exists("training_data.txt"):
        with open("training_data.txt", "w") as f:
            f.write("Hello world! This is a test training data for PK-1 model. " * 100)

    trainer = Trainer("training_data.txt", device="cpu")
    trainer.train(steps=50)
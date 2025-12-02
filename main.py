import torch
import torch.nn as nn
from config import PK1Config
from model import PK1Model
from tokenizer import TokenizerAdapter
from generator import TextGenerator
import sys

if __name__ == "__main__":
    device = "cpu"
    torch.set_num_threads(4)

    config = PK1Config()
    try:
        tokenizer = TokenizerAdapter()
    except ImportError:
        print("Install tiktoken first.")
        exit(1)

    print(f"Loading PK-1 on {device.upper()}...", end="")
    model = PK1Model(config).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.eval()
    print(" Done!")

    generator = TextGenerator(model, tokenizer, device)

    print("\nChat with PK-1 (type 'exit' to quit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
            response = generator.generate(user_input, max_new_tokens=50)
            print(f"PK-1: {response}")

        except KeyboardInterrupt:
            break
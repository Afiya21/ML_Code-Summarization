import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
import sys
import os

# Add the project root to the system path to allow importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import CodeSeq2Seq
from src.utils import preprocess_function, tokenizer

def train_model():
    # 1. Load Data
    print("Loading dataset...")
    # Using 50,000 examples as decided
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split='train[:50000]')
    
    # 2. Preprocess Data
    print("Tokenizing data...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    train_loader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True)

    # 3. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")
    
    model = CodeSeq2Seq(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 4. Training Loop
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Shift labels for teacher forcing
            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]

            optimizer.zero_grad()
            output = model(input_ids, tgt_input)
            
            # Reshape for loss calculation
            loss = criterion(output.reshape(-1, tokenizer.vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")

    # 5. Save Model
    save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'code_summary_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
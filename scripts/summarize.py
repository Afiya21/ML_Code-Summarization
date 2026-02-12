import torch
import argparse
import sys
import os
from transformers import AutoTokenizer

# Add the project root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the model class (we will create this next)
from src.model import CodeSeq2Seq

def summarize_code(code_snippet, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Initialize Model
    model = CodeSeq2Seq(vocab_size=tokenizer.vocab_size).to(device)
    
    # Load Weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please download code_summary_model.pth from Colab and put it in the 'models' folder.")
        return

    model.eval()
    
    # Tokenize input
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=128).input_ids.to(device)
    
    # Start generation
    decoder_input = torch.tensor([[tokenizer.cls_token_id]]).to(device)
    
    print("Generating summary...")
    with torch.no_grad():
        for _ in range(20): 
            output = model(inputs, decoder_input)
            next_token = output.argmax(dim=-1)[:, -1].unsqueeze(0)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            if next_token.item() == tokenizer.sep_token_id:
                break
                
    summary = tokenizer.decode(decoder_input[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type=str, required=True, help="The Python code string to summarize")
    args = parser.parse_args()

    # Look for the model in the 'models' folder
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'code_summary_model.pth')
    result = summarize_code(args.input, model_path)
    print(f"\nSummary: {result}")
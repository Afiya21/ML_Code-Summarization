from transformers import AutoTokenizer

# Initialize the tokenizer globally or pass it in
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def preprocess_function(examples):
    """
    Tokenizes the code (input) and docstring (target).
    Truncates and pads to a max length of 128.
    """
    # Tokenize Code (Input) using the 'code' column
    inputs = tokenizer(examples["code"], padding="max_length", truncation=True, max_length=128)
    
    # Tokenize Summary (Target) using the 'docstring' column
    targets = tokenizer(examples["docstring"], padding="max_length", truncation=True, max_length=128)
    
    return {"input_ids": inputs["input_ids"], "labels": targets["input_ids"]}
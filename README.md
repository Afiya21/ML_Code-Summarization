# ML-Based Python Code Summarization

## 1. Project Overview
This project implements a Sequence-to-Sequence (Seq2Seq) Transformer model using PyTorch to generate natural language summaries for Python code snippets. The model is trained on the CodeXGLUE dataset (a subset of CodeSearchNet) to interpret Python syntax and produce readable docstrings.

## 2. Requirements
**Python:** 3.x
**Libraries:** PyTorch, Transformers, Datasets, Evaluate, Rouge_score, NumPy
**Hardware:** A GPU is recommended for training (e.g., Google Colab T4).

## 3. Installation
1.  **Clone or unzip the project folder.**
2. **Install dependencies** using the provided requirements file:
    ```bash
    pip install -r requirements.txt
    ```

## 4. Usage Instructions

### A. Train the Model
Run the training script to download the dataset, preprocess it, and train the Transformer model. The trained model artifact will be saved to the `models/` directory.
```bash
python scripts/train.py
```

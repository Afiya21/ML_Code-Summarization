# ML-Based Python Code Summarization

## 1. Project Overview
This project implements a Sequence-to-Sequence (Seq2Seq) Transformer model using PyTorch to generate natural language summaries for Python code snippets. [cite_start]The model is trained on the CodeXGLUE dataset (a subset of CodeSearchNet) to interpret Python syntax and produce readable docstrings[cite: 12, 19].

## 2. Requirements
* [cite_start]**Python:** 3.x [cite: 61]
* [cite_start]**Libraries:** PyTorch, Transformers, Datasets, Evaluate, Rouge_score, NumPy [cite: 31, 62]
* **Hardware:** A GPU is recommended for training (e.g., Google Colab T4).

## 3. Installation
1.  **Clone or unzip the project folder.**
2.  [cite_start]**Install dependencies** using the provided requirements file[cite: 31]:
    ```bash
    pip install -r requirements.txt
    ```

## 4. Usage Instructions

### A. Train the Model
Run the training script to download the dataset, preprocess it, and train the Transformer model. [cite_start]The trained model artifact will be saved to the `models/` directory[cite: 32].
```bash
python scripts/train.py
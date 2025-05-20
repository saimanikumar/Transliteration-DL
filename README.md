# Assignment 3: Seq2Seq Transliteration with and without Attention

**Name:** Sai Mani Kumar Devathi
**Roll No:** DA24M016

---

## Project Overview

This repository contains two implementations of a character-level Seq2Seq transliteration system (Latin → Devanagari): one using a vanilla RNN encoder–decoder, and the other augmented with an attention mechanism. The goal is to compare performance, inspect predictions, and analyze the gains achieved by attention.

## Links

* **WandB Report:** [View the full training & evaluation dashboard](https://api.wandb.ai/links/da24m016-indian-institute-of-technology-madras/p4fit53s)
* **GitHub Repository:** [https://github.com/saimanikumar-da24m016/da6401\_assignment3](https://github.com/saimanikumar-da24m016/da6401_assignment3)


## Repository Structure

```
|--- assignment_3_attention.ipynb          # Notebook: training + eval with attention
|--- assignment_3_vannila.ipynb            # Notebook: training + eval without attention
|--- model_attention.py                    # Seq2Seq model with attention
|--- model_vannila.py                      # Vanilla Seq2Seq model
|--- train_attention.py                    # Training script for attention model
|--- train_vannila.py                      # Training script for vanilla model
|--- res_attention_predictions/
|    |--- best_model_attention.pt          # Saved checkpoint
|    |--- test_predictions_attention.csv   # Test set predictions with attention
|    |--- visual_examples.csv              # Sample visual examples
|--- res_vannila_predictions/
|    |--- best_model_vanilla.pt            # Saved checkpoint
|    |--- test_predictions_vanilla.csv     # Test set predictions without attention
|--- vocab/
|    |--- best_model.pt                    # Best model for vocab building (optional)
|    |--- src_vocab.json                   # Source vocabulary
|    |--- tgt_vocab.json                   # Target vocabulary
|--- README.md                             # This file
```

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/saimanikumar-da24m016/da6401_assignment3.git
   cd da6401_assignment3
   ```
2. Create a Python environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Quickstart

* **Train Attention Model:**

  ```bash
  python train_attention.py --config configs/attention.yaml
  ```
* **Train Vanilla Model:**

  ```bash
  python train_vannila.py --config configs/vanilla.yaml
  ```
* **Evaluate & Visualize:**

  * Open `assignment_3_attention.ipynb` or `assignment_3_vannila.ipynb` in Jupyter.
  * Generate accuracy metrics, confusion matrices, and inspect example predictions.

## Results & Predictions

* The `res_attention_predictions/` folder contains the best checkpoint and test predictions for the attention model.
* The `res_vannila_predictions/` folder contains the same for the vanilla model.
* You can compare `test_predictions_attention.csv` vs. `test_predictions_vanilla.csv` to see where attention helps.



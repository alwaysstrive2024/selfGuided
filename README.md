
# Self-Guided Attention for Explainable Natural Language Inference (NLI)

## 📌 Overview

This repository contains the implementation of **Self-Guided Attention**, a novel mechanism designed to enhance the interpretability and predictive performance of Transformer-based models (BERT) by integrating human rationales directly into the attention layers.

Traditional attention mechanisms often fail to align with human reasoning. This project addresses this gap by implementing an explicit intervention strategy on the **e-SNLI dataset**, demonstrating that guiding model focus toward human-annotated tokens significantly improves both explainability metrics and classification accuracy.

## 🚀 Key Features


**Direct Attention Intervention**: Implements "Soft Multiplicative Biasing" to inject human knowledge into the last four encoder layers of BERT.


**Explainability Metrics Pipeline**: Integrated evaluation for XAI, including:
  
**Top-k Attention IoU**: Measuring alignment between model focus and human rationales.



**AUPRC**: Assessing token importance discrimination.



**Comprehensiveness & Sufficiency**: Quantifying the causal impact of highlighted tokens on model confidence.





**Robust Preprocessing**: Custom pipeline for aligning natural language explanations with WordPiece tokenization using BERT's offset mapping.



**Comparative Analysis Framework**: Built-in support for benchmarking against **Vanilla Fine-tuning** and **Random Attention Masking** controls.



## 🏗️ Technical Architecture

The core of this project is the modification of the standard BERT self-attention formula:


Where  is a binary gold mask derived from human explanations, and  is a hyperparameter controlling the guidance strength.

## 📂 Project Structure

```text
selfGuided/
├── data/               # Scripts for e-SNLI dataset downloading and preprocessing
├── models/             # Implementation of GuidedBERT and custom attention layers
├── utils/              # Metrics calculation (IoU, AUPRC, Faithfulness)
├── experiments/        # Training scripts and hyperparameter configurations
├── notebook/           # Visualization of attention heatmaps and qualitative analysis
├── requirements.txt    # Project dependencies
└── main.py             # Entry point for training and evaluation

```

## 📊 Performance & Results

Experiments on the **e-SNLI** dataset prove that the Guided Model outperforms baselines in both task effectiveness and transparency:

| Model | Val Accuracy | Attention IoU | AUPRC | Comprehensiveness |
| --- | --- | --- | --- | --- |
| **Guided (Ours)** | **0.9067** | **0.3784** | **0.6561** | **0.3199** |
| Vanilla BERT | 0.9011 | 0.2665 | 0.4881 | 0.2135 |
| Random Control | 0.8926 | 0.3181 | 0.3053 | 0.1535 |

The results confirm that structured human intervention serves as a practical strategy for enhancing model reasoning.

## 🛠️ Installation & Usage

### Prerequisites

* Python 3.8+
* PyTorch 1.12+
* Transformers (Hugging Face)

### Setup

```bash
git clone git@github.com:alwaysstrive2024/selfGuided.git
cd selfGuided
pip install -r requirements.txt

```

### Training

To train the model with guided attention:

```bash
python main.py --mode guided --layers 4 --lambda 1.5

```

## 📖 Citation

If you find this work useful for your research or projects, please cite the original paper:

> Phil Gu. (2024). *Self-Guided Attention for Explainable Natural Language Inference.*

## 📄 License

This project is licensed under the MIT License.

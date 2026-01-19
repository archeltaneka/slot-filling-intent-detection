# Spoken Language Understanding (SLU) for Task-Oriented Dialogue

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://archeltaneka-slot-filling-intent-detection-app-vcbymi.streamlit.app/)
![Tests](https://github.com/archeltaneka/slot-filling-intent-detection/actions/workflows/main.yml/badge.svg)

## 📃 Overview

This repository contains an end-to-end Spoken Language Understanding (SLU) production-ready pipeline for Joint Intent Detection and Slot Filling. This project implements multiple architectures from CRF baselines to Joint BiLSTM and BERT-based models, integrated with a Streamlit interface and automated CI/CD. The system performs two tasks simultaneously:

- Intent Classification: Determining what the user wants to do (e.g., flight query).
- Slot Filling: Identifying key entities in the sentence (e.g., origin, destination, date, etc.).

## 🚀 Features

- Joint Modeling: Simultaneously predicts intents and slot labels using shared representations.
- Multiple Architecture Comparisons:
    - Baseline: CRF (Slots) + Random Forest (Intent).
    - Neural: BiLSTM and BiLSTM-Attention with GloVe embeddings.
    - Transformer: Joint-BERT fine-tuning using HuggingFace.
- Robust Pipeline: Custom data loaders, feature engineers, and group-aware data splitters.
- Production Quality: Automated unit testing suite and GitHub Actions CI/CD pipeline.

## 🛠️Tech Stack

- Python
- Streamlit
- NumPy
- Scikit-learn
- PyTorch
- Transformers
- HuggingFace

## 📃Requirements

- Python 3.10+

## 📦Installation & Setup

1. Clone the repo

```{bash}
git clone https://github.com/archeltaneka/slot-filling-intent-detection.git
cd slot-filling-intent-detection
```

2. Install dependencies

```{bash}
pip install -r requirements.txt
```

3. Run Streamlit app locally

```{bash}
streamlit run app.py
```

## 📊 Training/Experimenting Models

To run the full training pipeline (Baseline, BiLSTM, and BERT):

```{bash}
python train.py
```

Training artifacts (model weights and vocabularies) will be saved to files/checkpoints/.

To experiment with different hyperparameters, modify the config files `config.yaml`

```
# config.yaml
...
embed_dim: 100
hidden_dim: 256
num_layers: 5
dropout: 0.5
```

Then rerun the training pipeline `train.py`

## 🧪 Testing

We maintain high code quality through automated testing and continuous integration.

- Unit Testing (WIP): Comprehensive tests for data-processing modules (Splitter, Builder, Feature Engineer) using pytest.
- Automated Workflow: GitHub Actions runs the test suite on every Pull Request to ensure no regressions.
- Test Coverage (WIP): We are currently integrating Codecov to track test coverage and identify untested code paths.




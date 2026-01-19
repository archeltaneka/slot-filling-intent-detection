# Spoken Language Understanding (SLU) for Task-Oriented Dialogue

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://archeltaneka-slot-filling-intent-detection-app-vcbymi.streamlit.app/)
![Tests](https://github.com/archeltaneka/slot-filling-intent-detection/actions/workflows/main.yml/badge.svg)

## 📃 Overview

This repository contains an end-to-end Spoken Language Understanding (SLU) production-ready pipeline for Joint Intent Detection and Slot Filling. This project implements multiple architectures from CRF baselines to Joint BiLSTM and BERT-based models, integrated with a Streamlit interface and automated CI/CD. The system performs two tasks simultaneously:

- Intent Classification: Determining what the user wants to do (e.g., flight query).
- Slot Filling: Identifying key entities in the sentence (e.g., origin, destination, date, etc.).

🔗 Live App: https://archeltaneka-slot-filling-intent-detection-app-vcbymi.streamlit.app/

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

## 🛠 Project Structure

```
slot-filling-intent-detection/
├── data/                      # Raw data files
├── files/
│   ├── checkpoints/           # Saved model checkpoints
│   └── embedding/             # Created when training models
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── builder.py          # Transform raw data 
│   │   ├── data_utils.py       # Data processing utilities
│   │   ├── feature_engineer.py # Feature engineer transformed data
│   │   ├── loader.py           # Data loader
│   │   └── splitter.py         # Data splitter using group-aware splitter
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py         # CRF baseline model
│   │   ├── model_utils.py      # Model utilities
│   │   ├── models.py           # Joint BILSTM and BERT models
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train_utils.py      # Model training utilities
│   │   ├── trainer.py          # Model trainer
│   └── __init__.py
│   └── evaluation.py           # Model evaluation
│   └── inference.py            # Model inference
│   └── pipeline.py             # Data transformation pipeline
│   └── utils.py                # General utilities
├── tests/
├── app.py                      # Streamlit web app
├── config.yaml                 # Model configuration file
├── download_models.py          # Download pre-trained models
├── requirements.txt
├── README.md
├── train.py                    # Training script
```

## 🍿Demo Video
[streamlit-app-2026-01-19-19-36-04.webm](https://github.com/user-attachments/assets/b507bb2e-02a8-4a82-b31d-9423095efb7b)

## 📄 License

MIT License © 2025 Archel Taneka

## ⚙️ Want to contribute?

PRs, suggestions, and issues are welcome.


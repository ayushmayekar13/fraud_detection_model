# Credit Card Fraud Detection 

## Overview

This project provides a complete machine learning pipeline for credit card fraud detection. It includes:

- Model training using XGBoost with SMOTE to handle class imbalance.
- Feature scaling for numerical features (`Time` and `Amount`).
- Saving and loading trained models and scalers.
- An interactive Streamlit dashboard for:
  - Single transaction prediction
  - Batch CSV predictions
  - Model evaluation with confusion matrix and feature importance
- Optional Chrome extension and WebSocket integration for real-time streaming.

- The model files included in this dataset are already trained on a dataset

### Installation and Setup

Clone the repository:
```bash
git clone https://github.com/yourusername/fraud_detection.git
cd fraud_detection


## Create and activate a virtual environment

python -m venv venv
source venv/bin/activate


## Install dependencies

pip install -r requirements.txt

## Make sure to load a dataset for training pupose

## Training the Model

python train_model.py

## Running the Streamlit App

streamlit run fraud_app.py


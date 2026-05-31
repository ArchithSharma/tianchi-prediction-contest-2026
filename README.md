# Tianchi Prediction Contest 2026 – Aftershock Prediction

This repository contains our team's solution for the **International Contest on Aftershock Forecasting**.

## Approach

This solution follows a supervised machine learning framework:

### Feature Engineering

Features were generated from the historical earthquake sequence, including:

* Mainshock characteristics
* Temporal statistics of prior aftershocks
* Magnitude distribution summaries
* Spatial characteristics of the sequence
* Sequence-level aggregation metrics

### Models Evaluated

Three machine learning models were compared:

* LightGBM
* XGBoost
* Neural Network

Models were evaluated using:

* Mean Absolute Error (MAE) for aftershock magnitude prediction
* Mean Absolute Error (MAE) for aftershock timing prediction

### Final Model

Based on validation performance, **XGBoost** was selected as the final submission model due to its strong overall performance and consistency across all prediction windows.

## Repository Structure

```text
.
├── code_notebook.ipynb        # Full modeling workflow
├── data/                      # Competition datasets
├── submissions/               # Generated submission files
├── models/                    # Saved model artifacts
└── README.md
```

## Results

The final system predicts both:

* Time of the largest aftershock
* Magnitude of the largest aftershock

for each of the three forecast horizons (T1–T3).

Model selection was performed using cross-validation and held-out validation earthquake sequences.

## Reproducing the Results

### Requirements

```bash
pip install -r requirements.txt
```

### Training

Run the notebook:

```bash
jupyter notebook code_notebook.ipynb
```

or execute the training pipeline directly:

```bash
python train.py
```

### Generating Submission Files

```bash
python generate_submission.py
```

This produces the required competition files:

```text
YYYYMMDDHHMMSS-T1-T2.csv
YYYYMMDDHHMMSS-T3.csv
```

for every earthquake sequence in the test set.

## Competition

**International Competition on Prediction Technology for Aftershocks (Tianchi 2026)**

The competition aims to improve forecasting of strong aftershocks through statistical, machine learning, and AI-based methods, helping advance earthquake hazard assessment and disaster response.

## Author

**Archith Sharma**

GitHub: https://github.com/ArchithSharma

# Tianchi Prediction Contest 2026 – Aftershock Prediction

This repository contains our team's solution for the **International Contest on Aftershock Forecasting**.

The goal is to predict:

* **Maximum aftershock magnitude**
* **Time of occurrence of the maximum aftershock**

for three forecasting horizons after a mainshock event:

| Window | Forecast Horizon |
| ------ | ---------------- |
| T1     | 0–24 hours       |
| T2     | 24–72 hours      |
| T3     | 72–168 hours     |

---

## Methodology

### Data Processing

Raw earthquake sequence data are transformed into sequence-level features describing:

* Mainshock characteristics
* Historical aftershock activity
* Magnitude statistics
* Temporal behavior
* Spatial properties of the sequence

Processed datasets are stored in the `cleaned_data/` directory.

### Models Evaluated

Three machine learning approaches were explored:

* LightGBM
* XGBoost
* Feed-Forward Neural Network

Models were trained separately for:

1. Predicting the occurrence time of the strongest aftershock
2. Predicting the magnitude of the strongest aftershock

for each forecasting window (T1–T3).

### Model Selection

Validation experiments compared timing and magnitude prediction errors across all models. XGBoost was selected as the final submission model due to its strong overall performance and robustness across forecasting horizons.

---

## Repository Structure

```text
.
├── cleaned_data/          # Processed training and test datasets
├── models/                # Saved model/prediction artifacts
├── submission_files/      # Competition submission CSV files
├── code_notebook.ipynb    # Complete workflow: preprocessing, training, evaluation
├── createfiles.py         # Generates competition submission files
├── test_eq_data.zip       # Competition test dataset
└── README.md
```

---

## Running the Project

### Open the notebook

```bash
jupyter notebook code_notebook.ipynb
```

The notebook contains:

* Data preprocessing
* Feature engineering
* Model training
* Validation experiments
* Prediction generation

### Generate Submission Files

```bash
python createfiles.py
```

This script converts model predictions into the competition submission format:

```text
YYYYMMDDHHMMSS-T1-T2.csv
YYYYMMDDHHMMSS-T3.csv
```

for every earthquake sequence in the test set.

---

## Competition Submission Format

For each earthquake sequence:

### File 1: T1 and T2

```text
YYYYMMDDHHMMSS-T1-T2.csv
```

Contains:

* Maximum aftershock magnitude prediction
* Predicted occurrence time

for the T1 and T2 forecasting windows.

### File 2: T3

```text
YYYYMMDDHHMMSS-T3.csv
```

Contains the prediction for the T3 forecasting window.

---

GitHub: https://github.com/ArchithSharma

This repository was developed as part of the Tianchi 2026 Aftershock Prediction Competition.

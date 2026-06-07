# ANOMALY-IDS: Explainable Hybrid Intrusion Detection System

## Overview

This repository contains the implementation of an **Explainable Hybrid Intrusion Detection System (IDS)** developed as part of a Master's Thesis.

The proposed system combines:

- **Dense Autoencoder (DAE)** for anomaly detection
- **XGBoost** for attack classification
- **SHAP (SHapley Additive Explanations)** for explainable AI
- **Streamlit** for interactive deployment

The system is evaluated using the **NSL-KDD dataset** and demonstrates strong performance in detecting network intrusions while providing interpretable explanations for security analysts.

---

## Research Objective

Traditional signature-based IDS struggle to detect:

- Zero-day attacks
- Unknown attack patterns
- Evolving cyber threats

The objective of this research is to develop a hybrid IDS capable of:

1. Learning normal network behaviour
2. Detecting anomalous traffic
3. Classifying attack types
4. Explaining predictions using XAI techniques

---

## System Architecture

```text
NSL-KDD Dataset
        │
        ▼
Data Preprocessing
(Label Encoding + Standardization)
        │
        ▼
Dense Autoencoder
(Anomaly Detection)
        │
        ▼
Reconstruction Error
        │
        ▼
Threshold-Based Detection
        │
        ▼
XGBoost Classifier
(Attack Classification)
        │
        ▼
SHAP Explainability
        │
        ▼
Streamlit Dashboard
```

---

## Project Structure

```text
ANOMALY-IDS/
│
├── data/
│   ├── raw/
│   │   └── nsl-kdd/
│   ├── processed/
│   └── generated/
│
├── models/
│   ├── dense_autoencoder_best.pth
│   ├── attack_classifier_full_xgb.pkl
│   ├── attack_classifier_xgb.pkl
│   ├── vae.pth
│   ├── lstm_autoencoder_best.pth
│   └── anomaly_threshold.txt
│
├── notebooks/
│   ├── eda.ipynb
│   ├── GNN.ipynb
│   └── TrainTestLabels.ipynb
│
├── results/
│
├── src/
│   ├── preprocessing.py
│   ├── DenseAutoEncoderModel.py
│   ├── DenseAutoEncoderTrain.py
│   ├── autoencoder_model.py
│   ├── attack_classifier.py
│   ├── explain_dense_shap.py
│   ├── FeatureImportance.py
│   ├── FeatureImportancePearson.py
│   ├── vae_model.py
│   ├── lstm_model.py
│   └── train_lstm.py
│
├── app.py
├── requirements.txt
├── test_app_csv.py
└── test_train_app.py
```

---

## Dataset

### NSL-KDD

The NSL-KDD dataset is an improved version of the KDD Cup 1999 dataset and is widely used for benchmarking intrusion detection systems.

### Attack Categories

| Category | Examples |
|-----------|-----------|
| DoS | Neptune, Smurf, Teardrop |
| Probe | Satan, Portsweep, Nmap |
| R2L | Guess_Password, FTP_Write |
| U2R | Buffer_Overflow, Rootkit |

---

## Methodology

### 1. Data Preprocessing

The preprocessing pipeline performs:

- Missing value handling
- Duplicate removal
- Label encoding
- Feature scaling using StandardScaler
- Label mapping generation

Output:

```text
train_processed.csv
test_processed.csv
label_mappings.json
standard_scaler.pkl
```

---

### 2. Dense Autoencoder

The Dense Autoencoder learns normal network behaviour and identifies anomalies using reconstruction error.

Architecture:

```text
Input Layer (41 Features)
        ↓
Dense (128)
        ↓
Dense (64)
        ↓
Bottleneck (32)
        ↓
Dense (64)
        ↓
Dense (128)
        ↓
Output Layer
```

Features:

- Batch Normalization
- Dropout
- Adam Optimizer
- Learning Rate Scheduling

---

### 3. Threshold-Based Anomaly Detection

Network traffic is classified using reconstruction error:

```text
Reconstruction Error > Threshold
        ↓
Attack

Reconstruction Error ≤ Threshold
        ↓
Normal
```

The threshold is optimized experimentally using percentile analysis.

---

### 4. XGBoost Attack Classification

Detected attacks are further classified using XGBoost.

Advantages:

- High accuracy
- Fast inference
- Handles imbalanced datasets
- Robust feature selection

Output:

```text
Attack Type
↓
Neptune
Smurf
Satan
Guess_Password
...
```

---

### 5. Explainable AI (SHAP)

SHAP is used to explain:

- Global feature importance
- Local prediction explanations
- Attack-specific feature contributions

Example influential features:

- service
- flag
- num_root
- num_compromised
- num_file_creations

---

## Experimental Models

### Machine Learning

- Random Forest
- XGBoost
- Isolation Forest
- One-Class SVM
- Local Outlier Factor

### Deep Learning

- Dense Autoencoder
- Variational Autoencoder (VAE)
- LSTM Autoencoder

### Experimental

- Graph Neural Network (GNN)

---

## Results

### Dense Autoencoder

| Metric | Value |
|----------|----------|
| ROC-AUC | 0.9259 |
| Average Precision | 0.9289 |

### Hybrid IDS

| Metric | Value |
|----------|----------|
| Accuracy | 81.95% |
| Precision | 96.9% |
| Recall | 70.5% |

### Explainability

SHAP successfully identified key features responsible for anomaly detection and attack classification.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/anomaly-ids.git

cd anomaly-ids
```

### Create Virtual Environment

```bash
python -m venv venv

source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

### Preprocess Dataset

```bash
python src/preprocessing.py
```

### Train Dense Autoencoder

```bash
python src/DenseAutoEncoderTrain.py
```

### Train XGBoost Classifier

```bash
python src/attack_classifier.py
```

### Generate SHAP Explanations

```bash
python src/explain_dense_shap.py
```

---

## Running the Application

Launch Streamlit:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The application allows:

- Network traffic analysis
- Attack detection
- Attack classification
- Explainable AI visualization

---

## Thesis Contributions

- Developed a hybrid IDS using Dense Autoencoder and XGBoost
- Integrated SHAP-based explainability
- Evaluated multiple ML and DL models
- Performed feature importance validation using five independent methods
- Built an interactive Streamlit-based deployment interface
- Demonstrated effective detection of anomalous network traffic

---

## Future Work

Potential improvements include:

- Evaluation on modern datasets (CICIDS2017, CICIDS2018, UNSW-NB15)
- Real-time network packet monitoring
- Federated learning for distributed IDS
- Online learning for adaptive threat detection
- Advanced Graph Neural Network architectures

---

## Author

**Dhyaneswar Bachu**

Master's Thesis  
**Intrusion Detection Using Explainable Hybrid Deep Learning Models**

---

## License

This project is developed for academic and research purposes. Use and modify freely with proper citation.
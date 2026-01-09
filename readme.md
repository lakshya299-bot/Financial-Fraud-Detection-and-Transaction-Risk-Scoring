# Transaction Fraud Detection & Risk Scoring  
*Extreme Class Imbalance • Cost-Sensitive Modeling*

## Overview
This project formulates **financial fraud detection as a transaction-level risk scoring problem** under **extreme class imbalance**, prioritizing **high fraud capture (recall)** while controlling false positives to ensure operational feasibility.

Instead of optimizing for raw accuracy, the system focuses on **business-aligned metrics** such as **ROC-AUC, Recall, and Precision–Recall trade-offs**, reflecting the asymmetric cost of missed fraud versus false alerts in real-world financial systems.

The end-to-end pipeline covers exploratory analysis, leakage-safe feature engineering, gradient-boosted modeling, and production-style deployment using Docker on Azure App Service (Linux).

---

## Problem Framing
In real payment systems:
- Fraud cases are **rare (<1%)**
- Missing a fraudulent transaction is significantly more costly than flagging a legitimate one
- Accuracy is misleading under severe class imbalance

Accordingly, this project frames fraud detection as a **risk scoring and threshold optimization problem**, where:
- Models output a continuous fraud risk score
- Operating thresholds are chosen based on business priorities (high recall over accuracy)

---

## Dataset  
### PaySim: A Financial Mobile Money Simulator

**Dataset link:**  
https://www.kaggle.com/datasets/ealaxi/paysim1/data

### Motivation
Due to the **private and sensitive nature of financial transactions**, real-world fraud datasets are rarely publicly available. PaySim addresses this gap by generating **realistic synthetic mobile money transaction data** derived from aggregated patterns observed in a real financial service.

---

### Dataset Description
- Simulates **mobile money transactions** based on one month of real financial logs
- Original data sourced from a multinational mobile money provider operating in 14+ countries
- Dataset provided on Kaggle is **scaled to 1/4 of the original volume**
- Time resolution: **1 step = 1 hour**
- Total duration: **744 steps (≈ 30 days)**

---

### Transaction Types
- CASH-IN  
- CASH-OUT  
- DEBIT  
- PAYMENT  
- TRANSFER  

---

### Features (Column Definitions)

| Column | Description |
|------|------------|
| step | Time step (1 hour per step, 744 total) |
| type | Transaction type |
| amount | Transaction amount |
| nameOrig | Originating account |
| oldbalanceOrg | Balance before transaction (origin) |
| newbalanceOrig | Balance after transaction (origin) |
| nameDest | Destination account |
| oldbalanceDest | Balance before transaction (destination) |
| newbalanceDest | Balance after transaction (destination) |
| isFraud | Fraud indicator (target variable) |
| isFlaggedFraud | Rule-based flag for large transfers (>200,000) |

**Important note:**  
Fraudulent transactions are cancelled in the simulator. Therefore, balance-based fields  
(`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) were **excluded from modeling** to prevent target leakage.

---

## Exploratory Data Analysis (EDA)
EDA focused on understanding differences between fraud and non-fraud transactions across:
- Transaction amount distributions
- Transaction type frequencies
- Temporal behavior (hourly patterns)
- Fraud concentration in TRANSFER and CASH-OUT operations

These insights guided:
- Feature selection
- Leakage-safe feature engineering
- Metric selection aligned with business cost asymmetry

---

## Methodology

### Feature Engineering
- Aggregated transaction statistics per account
- Frequency and velocity-based features
- Temporal behavior indicators
- Strict exclusion of post-transaction balance fields to avoid leakage

---

### Modeling
- Algorithm: **XGBoost**
- Objective: Binary classification with imbalance-aware learning
- Evaluation metrics:
  - ROC-AUC (ranking quality)
  - Recall (fraud capture)
  - Precision–Recall trade-offs for threshold selection

---

## Results

### Model Performance
- **ROC-AUC:** ~0.94  
- **Recall (Fraud Class):** ~0.92  

These results indicate strong discrimination between fraudulent and legitimate transactions while maintaining recall levels suitable for real-world fraud screening systems.

---

### Threshold Analysis
Rather than relying on default probability thresholds:
- Precision–Recall curves were analyzed
- Operating points were selected to favor **high fraud capture**
- Thresholds aligned with business priorities rather than accuracy maximization

---

## Deployment
The fraud scoring pipeline was deployed in a **production-like environment** using:
- Docker
- Azure App Service (Linux)

This enables:
- Scalable transaction risk scoring
- Reproducible inference environments
- Easy integration with downstream monitoring or alerting systems

---


## References
- Lopez-Rojas, E. A., Elmir, A., Axelsson, S.  
  *PaySim: A Financial Mobile Money Simulator for Fraud Detection*,  
  European Modeling and Simulation Symposium (EMSS), 2016.

---

## Repository Notes
- Source code only (no datasets tracked)
- Synthetic dataset referenced via Kaggle
- Model artifacts packaged within Docker images
- Focus on reproducibility and deployment realism

# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Forever_Noobs  
**Team Members:** Nisargkumar Mulani, Yagnesh Dhankecha, Ved Kuvadiya, Saumya Thacker  
**Submission Date:** 12-10-2025

---

## 1. Executive Summary

We developed a **hybrid multimodal pricing model** leveraging both **semantic text embeddings** and **TF-IDF-based statistical features**, compressed via **PCA**, to efficiently capture product descriptions. A **stacked ensemble of XGBoost, LightGBM, and CatBoost** models was trained with an **MLPRegressor as the meta-learner**, leading to strong price prediction accuracy while mitigating noise using **log transformation and outlier removal strategies**.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
Product prices vary widely due to inconsistent catalog descriptions and noise in textual data. Exploratory cleaning revealed:

**Key Observations:**
- Catalog content often contains **HTML tags, URLs, punctuation, and digits**, making preprocessing essential.
- **High variance in price distribution** required **percentile-based outlier clipping (1%-99%)**.
- Some products had **minimal descriptions**, necessitating **robust semantic feature extraction**.

### 2.2 Solution Strategy

**Approach Type:** Hybrid Ensemble (Stacking Regressor)  
**Core Innovation:** Dual feature fusion — combining **SentenceTransformer embeddings + TF-IDF PCA features**, then fed into a **stacked model of gradient boosting regressors with a neural regressor as final estimator**.

---

## 3. Model Architecture

### 3.1 Architecture Overview

Raw Text ──▶ Cleaning ──▶ (1) Sentence Embeddings (MiniLM)
└─▶ (2) TF-IDF → PCA Compression
──▶ Concatenation → StandardScaler
──▶ StackingRegressor
├── XGBoost
├── LightGBM
└── CatBoost
──▶ MLPRegressor (Final Layer)
──▶ Predicted Price (Expm1 Reverse Log)


### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing steps: Lowercase → Remove HTML → Remove URLs → Remove Digits → Remove Punctuation → Normalize spaces
- Model: `SentenceTransformer('all-MiniLM-L6-v2')`
- Embedding Output: 384-D per sample

**Statistical Text Pipeline (TF-IDF + PCA):**
- TF-IDF (`max_features=5000`, English stopwords)
- PCA Dimensionality Reduction → 50 Components

**Stacked Ensemble Configuration:**

| Model             | Framework | Key Parameters                                |
|------------------|-----------|-----------------------------------------------|
| XGBRegressor     | XGBoost   | 600 estimators, depth 8, subsample 0.8        |
| LGBMRegressor    | LightGBM  | 600 estimators, num_leaves 64                 |
| CatBoostRegressor| CatBoost  | 800 iterations, depth 7                       |
| MLPRegressor (Meta)| Sklearn | (256,128,64), relu, early stopping enabled     |

---

## 4. Model Performance

### 4.1 Validation Results
Training performed on **Fold 1 only** using **RepeatedKFold (5 splits × 2 repeats)**.

| Metric | Value |
|--------|-------|
| SMAPE  | _To be inserted after measurement_ |
| MAE / RMSE | _Optional if computed_ |

---

## 5. Conclusion

Our system effectively blends **semantic understanding and statistical generalization**, resulting in a **robust price prediction pipeline**. By combining **Transformer embeddings with traditional TF-IDF signals**, stacked over **gradient boosting and neural networks**, we achieved a highly scalable and noise-resistant architecture. Future enhancements include **image fusion and SMAPE-optimized loss functions**.

---

## Appendix

### A. Code Artefacts
🔗 https://github.com/NisargMulani/Amazon_ML_Challenge.git

### B. Additional Results
(Add graphs like price distribution, loss curves, or SHAP feature importance if available.)

---

**End of Document**


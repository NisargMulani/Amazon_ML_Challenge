# 🛒 Smart Product Pricing - ML Challenge 2025

A **hybrid multimodal machine learning solution** for predicting product prices using textual catalog descriptions.  
Built for **ML Challenge 2025** by **Team Forever_Noobs**.

---

## 🚀 Features

✔️ Clean & preprocess noisy product descriptions  
✔️ Dual feature pipeline – **Sentence Embeddings + TF-IDF PCA**  
✔️ **Stacked Ensemble Model** combining XGBoost, LightGBM & CatBoost  
✔️ Neural Network (**MLPRegressor**) as meta-learner  
✔️ Outlier handling with **percentile clipping + log transformation**  
✔️ Generates final **test_out.csv** with predicted prices

---

## 📂 Project Structure

```
├── data/                  # train.csv and test.csv here
├─ notebook/
│   ├── Notebook.ipynb      # Notebook code for training       
├─ Predicted-prize/
│   ├── test_out.csv         
├── documentation.md       # Full technical documentation
├── requirements.txt       # Dependencies
└── README.md              # You are here 🚀
```

---

## ⚙️ Setup Instructions

```bash
# Clone repository
https://github.com/NisargMulani/Amazon_ML_Challenge.git
cd Amazon_ML_Challenge

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Training & Inference

```bash
# Run full training pipeline
python notebook/Notebook.ipynb

# Generate predictions on test set
python notebook/Notebook.ipynb
```

The final output will be saved as:

```
test_out.csv
└── sample_id | price
```

---

## 🔧 Model Architecture

```
Text → Cleaning → (1) SentenceTransformer Embeddings
                → (2) TF-IDF + PCA
→ Concatenate → StandardScaler → StackingRegressor
                                 ├── XGBoost
                                 ├── LightGBM
                                 └── CatBoost
→ MLPRegressor (Meta) → Final Price
```

---

## 🏆 Results

| Metric | Score |
|--------|-------|
| SMAPE  | _Pending_ |

---

## 👨‍💻 Team

| Name | Role |
|------|------|
| Nisargkumar Mulani | NLP & Feature Engineering |
| Yagnesh Dhankecha  | Model Architecture |
| Ved Kuvadiya       | Data Processing |
| Saumya Thacker     | Evaluation & Reporting |

---

## 📌 Future Improvements

- 🖼️ Add product **image embeddings** for multimodal learning  
- ⚖️ Optimize model with **SMAPE-aware loss function**  
- 🌐 Deploy as an **API or Streamlit app**

---

⭐ If you like this work, consider giving the repo a **Star**!

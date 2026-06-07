# Heart Disease Prediction using IBM Watson AutoAI
> A clinical machine learning project that predicts the presence or absence of heart disease using IBM Watson Studio's AutoAI that's built during a virtual AI internship under IBM.

---

##  Project Overview

This project builds an end to end **binary classification pipeline** to predict whether a patient has heart disease (`0` = No, `1` = Yes), based on key cardiovascular and clinical indicators.

The entire ML workflow: from data preprocessing to model selection and deployment  was automated using **IBM Watson AutoAI**, which generated and compared multiple pipelines (P1–P8) and selected the best performing model.

**Best Model:** Snap SVM Classifier (Pipeline P8)  
**Selection Criteria:** Best balance of ROC-AUC and Recall critical for minimizing missed positive cases in a screening context.

---

## 🧰 Tech Stack

| Tool/Platform | Purpose |
|---|---|
| IBM Watson Studio | Cloud ML workspace |
| IBM AutoAI | Automated pipeline generation & model selection |
| Python | Scripting & API client |
| Pandas, NumPy | Data preprocessing |
| Matplotlib, Seaborn | Exploratory data analysis |
| Scikit-learn (via AutoAI) | Model training internals |
| IBM Cloud REST API | Model deployment & inference |

---

## 📂 Dataset

**File:** `heart.csv`  
**Task:** Binary Classification  
**Target column:** `target` (0 = No Disease, 1 = Disease Present)

| Feature | Description |
|---|---|
| `age` | Patient age in years |
| `sex` | Biological sex (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia status (0–3) |

---

## 🔄 Project Workflow

1. **Data Preprocessing** : cleaned dataset, encoded categoricals, standardized numeric features
2. **Watson Studio Setup** : uploaded `heart.csv` to IBM cloud workspace
3. **AutoAI Experiment** : ran binary classification, AutoAI generated 8 pipelines (P1–P8)
4. **Model Selection** : compared Accuracy, Recall, F1, ROC-AUC across pipelines
5. **Deployment** : deployed best model (P8, Snap SVM) as a secured REST API endpoint
6. **Validation** : batch tested on holdout set, tuned threshold to prioritize Recall
7. **Documentation** : maintained notebooks, metric dashboards, and deployment payloads
## 🚀 How to Run the Scoring Client

### Prerequisites
```bash
pip install requests
```

### Set Your API Key
Replace `<Your API Key>` in the script with your IBM Cloud API key.

### Run
```bash
python scoring_client.py
```

## 📊 Results

| Pipeline | Model | ROC-AUC | Recall | F1 |
|---|---|---|---|---|
| P8 ⭐ | Snap SVM Classifier | Best | Best | Strong |
| P7 | Snap Logistic Regression | High | High | Strong |
| P1–P6 | Various | Moderate | Moderate | Moderate |

> ⭐ P8 selected as final model. Prioritized Recall to minimize missed positive diagnoses.

---

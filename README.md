# 📦 E-Commerce Customer Churn Prediction

> A complete end-to-end Data Mining & Analytics project — from raw data to an interactive prediction dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-1.3+-150458?style=flat&logo=pandas&logoColor=white)
![HTML](https://img.shields.io/badge/Dashboard-HTML%2FJS-E34F26?style=flat&logo=html5&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Project Overview

Customer churn is when a customer stops doing business with a company. In e-commerce, retaining an existing customer is significantly cheaper than acquiring a new one. This project uses **machine learning** to predict which customers are at risk of churning, enabling businesses to take proactive retention action.

This project covers the full data mining pipeline:

```
Data Generation → EDA → Preprocessing → Model Training → Evaluation → Interactive Dashboard
```

---

## 🗂️ Project Structure

```
📁 Data Mining and Analytics/
│
├── 📄 churn_model.py          # Main ML pipeline — run this first
├── 📄 churn_dashboard.html    # Interactive web dashboard
├── 📄 churn_data.json         # Auto-generated output (created by churn_model.py)
└── 📄 README.md               # You are here
```

> **Note:** `churn_data.json` is auto-generated when you run `churn_model.py`. Do not edit it manually.

---

## ✨ Features

- 🔢 **Synthetic Dataset** — 2,000 realistic e-commerce customer records with 19 features
- 📊 **Exploratory Data Analysis** — Churn patterns across tenure, satisfaction, orders, complaints, and more
- ⚙️ **Full Preprocessing Pipeline** — Missing value imputation, label encoding, and feature scaling
- 🤖 **3 ML Models Trained** — Logistic Regression, Random Forest, Gradient Boosting
- 📈 **Comprehensive Evaluation** — Accuracy, AUC, Precision, Recall, F1, Confusion Matrix, 5-Fold CV
- 🌟 **Feature Importance Analysis** — Identify top churn drivers
- 🖥️ **Interactive Dashboard** — 7-tab web dashboard with live churn predictor

---

## 📊 Dataset

The dataset is synthetically generated to simulate a real e-commerce platform. It includes 2,000 customers with the following features:

| Feature | Description |
|---|---|
| `Tenure` | Number of months the customer has been on the platform |
| `SatisfactionScore` | Customer satisfaction rating (1–5) |
| `OrderCount` | Total number of orders placed |
| `DaySinceLastOrder` | Days since the customer last placed an order |
| `HourSpendOnApp` | Average hours spent on the app per day |
| `Complain` | Whether the customer filed a complaint (0/1) |
| `CashbackAmount` | Total cashback received |
| `CouponUsed` | Number of coupons used |
| `CityTier` | City classification (Tier 1 / 2 / 3) |
| `WarehouseToHome` | Distance from warehouse to customer's home |
| `PreferredLoginDevice` | Mobile Phone / Computer / Tablet |
| `PreferredPaymentMode` | Credit Card / UPI / E-Wallet / etc. |
| `Gender` | Male / Female |
| `MaritalStatus` | Married / Single / Divorced |
| `PreferedOrderCat` | Preferred product category |
| `NumberOfDeviceRegistered` | Devices linked to the account |
| `NumberOfAddress` | Number of saved addresses |
| `OrderAmountHikeFromlastYear` | % increase in order amount from last year |
| `Churn` | **Target variable** — 1 = Churned, 0 = Retained |

**Class Distribution:** 70.25% Retained · 29.75% Churned

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed. Then install the required libraries:

```bash
pip install scikit-learn pandas numpy
```

### Step 1 — Run the ML Pipeline

```bash
python churn_model.py
```

This will:
- Generate the synthetic dataset
- Perform EDA and compute statistics
- Preprocess the data
- Train all 3 models
- Evaluate and export results to `churn_data.json`

Expected output:
```
⚙  Generating dataset...
📊 Computing EDA stats...
🔧 Preprocessing...
🤖 Training models...
✅ Saved to: .../churn_data.json
✅ Done! Models trained successfully.
   Logistic Regression: Acc=0.9975 | AUC=1.0 | F1=0.9958
   Random Forest:       Acc=1.0    | AUC=1.0 | F1=1.0
   Gradient Boosting:   Acc=1.0    | AUC=1.0 | F1=1.0
```

### Step 2 — Launch the Dashboard

You need a local server to load the dashboard (browsers block local JSON file reads for security).

**Option A — Python (recommended, no extra install):**
```bash
python -m http.server 8000
```
Then open: [http://localhost:8000/churn_dashboard.html](http://localhost:8000/churn_dashboard.html)

**Option B — VS Code Live Server:**
1. Install the **Live Server** extension in VS Code
2. Right-click `churn_dashboard.html`
3. Click **"Open with Live Server"**

---

## 🤖 Models Used

| Model | Type | Key Strength |
|---|---|---|
| **Logistic Regression** | Linear | Fast, interpretable baseline |
| **Random Forest** | Ensemble (Bagging) | Handles non-linearity, gives feature importance |
| **Gradient Boosting** | Ensemble (Boosting) | Sequential error correction, high accuracy |

---

## 📈 Model Results

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Logistic Regression | 99.75% | 1.00 | 1.00 | 99.16% | 99.58% |
| Random Forest | 100% | 1.00 | 1.00 | 100% | 100% |
| Gradient Boosting | 100% | 1.00 | 1.00 | 100% | 100% |

> ⚠️ **Note on high scores:** The dataset is synthetically generated with clearly defined patterns, which leads to near-perfect results. In real-world datasets with noise and overlapping patterns, expect AUC scores in the 0.80–0.92 range. The focus of this project is the complete pipeline, not the metric scores.

---

## 🌟 Top Churn Drivers (Feature Importance — Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | SatisfactionScore | 41.3% |
| 2 | DaySinceLastOrder | 23.7% |
| 3 | OrderCount | 17.9% |
| 4 | Tenure | 9.2% |
| 5 | HourSpendOnApp | 5.0% |
| 6 | Complain | 1.8% |

**Key Insight:** Customers who are unsatisfied AND haven't ordered recently are by far the highest churn risk group.

---

## 🖥️ Dashboard Tabs

The interactive dashboard (`churn_dashboard.html`) has 7 sections:

| Tab | Contents |
|---|---|
| **Overview** | KPI cards, churn donut chart, model comparison bar chart |
| **EDA** | Side-by-side comparisons of churned vs retained customers |
| **Models** | Clickable model cards, confusion matrix, ROC curve, PRF scores |
| **Feature Importance** | Horizontal bar chart + importance breakdown |
| **Live Predictor** | Enter customer data → instant churn risk % with key factors |
| **Pipeline** | Visual ML workflow diagram |
| **Dataset** | Preview of raw data with churn labels |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| ML Library | scikit-learn |
| Data Processing | pandas, numpy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js |
| Fonts | Google Fonts (Syne + DM Mono) |

---

## 📚 Concepts Covered

- Binary Classification
- Exploratory Data Analysis (EDA)
- Data Preprocessing (Imputation, Encoding, Scaling)
- Ensemble Methods (Bagging & Boosting)
- Model Evaluation Metrics (AUC, F1, Confusion Matrix)
- K-Fold Cross Validation
- Feature Importance Analysis

---

## 👨‍💻 Author

Made as part of a **Data Mining and Analytics** university project.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

"""
E-Commerce Customer Churn Prediction
Data Mining & Analytics Project
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.impute import SimpleImputer
import json
import math
import warnings
warnings.filterwarnings('ignore')

# Custom JSON encoder: converts NaN/Inf to null so the output is valid JSON
class SafeEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._clean(o), _one_shot)
    def _clean(self, obj):
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._clean(v) for v in obj]
        return obj

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC E-COMMERCE DATASET GENERATION
# ─────────────────────────────────────────────
def generate_ecommerce_data(n=2000):
    churn = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

    data = {
        'CustomerID': [f'CUST{str(i).zfill(5)}' for i in range(1, n+1)],
        'Tenure':              np.where(churn, np.random.randint(1, 24, n),   np.random.randint(6, 72, n)),
        'CityTier':            np.random.choice([1, 2, 3], size=n, p=[0.4, 0.35, 0.25]),
        'WarehouseToHome':     np.random.randint(5, 35, n),
        'HourSpendOnApp':      np.clip(np.where(churn, np.random.normal(1.5, 1, n), np.random.normal(3, 1, n)), 0, 6).round(1),
        'NumberOfDeviceRegistered': np.random.randint(1, 6, n),
        'SatisfactionScore':   np.where(churn, np.random.randint(1, 3, n),    np.random.randint(3, 6, n)),
        'NumberOfAddress':     np.random.randint(1, 8, n),
        'Complain':            np.where(churn, np.random.choice([0,1], n, p=[0.4,0.6]), np.random.choice([0,1], n, p=[0.85,0.15])),
        'OrderAmountHikeFromlastYear': np.clip(np.random.normal(15, 8, n), 0, 40).round(1),
        'CouponUsed':          np.random.randint(0, 10, n),
        'OrderCount':          np.where(churn, np.random.randint(1, 5, n),    np.random.randint(3, 20, n)),
        'DaySinceLastOrder':   np.where(churn, np.random.randint(10, 45, n),  np.random.randint(0, 15, n)),
        'CashbackAmount':      np.clip(np.random.normal(180, 80, n), 0, 400).round(2),
        'PreferredLoginDevice': np.random.choice(['Mobile Phone', 'Computer', 'Tablet'], size=n, p=[0.6, 0.3, 0.1]),
        'PreferredPaymentMode': np.random.choice(['Credit Card', 'Debit Card', 'E-Wallet', 'UPI', 'Cash on Delivery'], size=n, p=[0.25, 0.2, 0.25, 0.2, 0.1]),
        'Gender':              np.random.choice(['Male', 'Female'], size=n, p=[0.6, 0.4]),
        'PreferedOrderCat':    np.random.choice(['Laptop & Accessory', 'Mobile', 'Fashion', 'Grocery', 'Others'], size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'MaritalStatus':       np.random.choice(['Married', 'Single', 'Divorced'], size=n, p=[0.55, 0.35, 0.1]),
        'Churn':               churn
    }

    df = pd.DataFrame(data)
    # Inject some missing values (realistic)
    for col in ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']:
        mask = np.random.rand(n) < 0.04
        df.loc[mask, col] = np.nan
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df.drop(columns=['CustomerID'], inplace=True)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.difference(['Churn']).tolist()

    # Impute
    num_imp = SimpleImputer(strategy='median')
    df[num_cols] = num_imp.fit_transform(df[num_cols])

    # Encode categoricals
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, X.columns.tolist()


# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test, feature_names):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=150, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Feature importance
        feat_imp = None
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            feat_imp = sorted(zip(feature_names, imp.tolist()), key=lambda x: x[1], reverse=True)[:12]

        results[name] = {
            'accuracy':    round(accuracy_score(y_test, y_pred), 4),
            'roc_auc':     round(roc_auc_score(y_test, y_prob), 4),
            'cv_mean':     round(cv_scores.mean(), 4),
            'cv_std':      round(cv_scores.std(), 4),
            'precision':   round(report['1']['precision'], 4),
            'recall':      round(report['1']['recall'], 4),
            'f1':          round(report['1']['f1-score'], 4),
            'cm':          cm,
            'fpr':         fpr.tolist()[::5],
            'tpr':         tpr.tolist()[::5],
            'feat_imp':    feat_imp,
        }

    return results


# ─────────────────────────────────────────────
# 4. EDA STATS
# ─────────────────────────────────────────────
def compute_eda(df):
    churn_rate = round(df['Churn'].mean() * 100, 2)
    total = len(df)
    churned = int(df['Churn'].sum())
    retained = total - churned

    # Churn by category
    cat_breakdowns = {}
    for col in ['PreferedOrderCat', 'PreferredLoginDevice', 'MaritalStatus', 'Gender', 'CityTier']:
        grp = df.groupby(col)['Churn'].mean().round(4) * 100
        cat_breakdowns[col] = grp.to_dict()

    # Numeric averages by churn
    num_cols = ['Tenure', 'SatisfactionScore', 'OrderCount', 'DaySinceLastOrder',
                'HourSpendOnApp', 'CashbackAmount', 'Complain']
    num_comparison = {}
    for col in num_cols:
        num_comparison[col] = {
            'churned':  round(df[df['Churn']==1][col].mean(), 2),
            'retained': round(df[df['Churn']==0][col].mean(), 2),
        }

    return {
        'total': total,
        'churned': churned,
        'retained': retained,
        'churn_rate': churn_rate,
        'cat_breakdowns': cat_breakdowns,
        'num_comparison': num_comparison,
    }


# ─────────────────────────────────────────────
# 5. RUN & EXPORT
# ─────────────────────────────────────────────
def run():
    print("⚙  Generating dataset...")
    df = generate_ecommerce_data(2000)

    print("📊 Computing EDA stats...")
    eda = compute_eda(df)

    print("🔧 Preprocessing...")
    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("🤖 Training models...")
    model_results = train_models(X_train, X_test, y_train, y_test, feature_names)

    output = {
        'eda': eda,
        'models': model_results,
        'feature_names': feature_names,
        'dataset_sample': df.head(10).to_dict(orient='records'),
    }

    # Replace NaN in sample with None so JSON is valid
    clean_sample = []
    for row in output['dataset_sample']:
        clean_row = {k: (None if isinstance(v, float) and math.isnan(v) else v)
                     for k, v in row.items()}
        clean_sample.append(clean_row)
    output['dataset_sample'] = clean_sample

    # Save in the same folder as this script (works on Windows, Mac, Linux)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'churn_data.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, cls=SafeEncoder)
    print(f"✅ Saved to: {output_path}")

    print("✅ Done! Models trained successfully.")
    for name, r in model_results.items():
        print(f"   {name}: Acc={r['accuracy']} | AUC={r['roc_auc']} | F1={r['f1']}")

    return output

if __name__ == '__main__':
    run()
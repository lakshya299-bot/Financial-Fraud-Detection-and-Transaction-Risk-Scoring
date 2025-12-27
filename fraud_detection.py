import os
import datasets
import pandas as pd
from datasets import Dataset, DatasetDict   
import numpy as np

# -------------------- Load data --------------------
train_identity = pd.read_csv(
    '/Users/lakshyasmac/Desktop/Fraud detection pipeline/ieee-fraud-detection/train_identity.csv'
)
train_transaction = pd.read_csv(
    '/Users/lakshyasmac/Desktop/Fraud detection pipeline/ieee-fraud-detection/train_transaction.csv'
)

train_identity.info()
train_transaction.info()

# -------------------- Merge --------------------
train = pd.merge(
    train_transaction,
    train_identity,
    on='TransactionID',
    how='left'
)

train.info()

# -------------------- Drop ultra-sparse columns --------------------
missing_ratio = train.isna().mean()
cols_to_drop = missing_ratio[missing_ratio > 0.90].index
train = train.drop(columns=cols_to_drop)

# -------------------- Separate target & features --------------------
y = train["isFraud"]
X = train.drop(columns=["isFraud", "TransactionID"])

# -------------------- Train-validation split --------------------
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------- Feature engineering --------------------
for df in [X_train, X_val]:
    df["Transaction_hour"] = (df["TransactionDT"] / 3600) % 24
    df["Transaction_day"]  = (df["TransactionDT"] / 86400).astype(int)
    df["Transaction_week"] = (df["TransactionDT"] / 604800).astype(int)

    df["Log_TransactionAmt"] = np.log1p(df["TransactionAmt"])

    df.drop(columns=["TransactionDT"], inplace=True)

# -------------------- Detect feature types --------------------
num_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

# -------------------- Preprocessing pipelines --------------------
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
froom sklearn.decomposition import PCA

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=42))
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# -------------------- Model --------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs'
)

# -------------------- Full pipeline --------------------
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# -------------------- Train --------------------
clf.fit(X_train, y_train)

# -------------------- Evaluate --------------------
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_val, clf.predict_proba(X_val)[:, 1])
y_val_proba = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_proba)

print(f"Validation AUC: {auc:.4f}")
print(classification_report(
    y_val,
    (y_val_proba >= 0.85).astype(int)
))

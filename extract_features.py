import pandas as pd
import numpy as np

# Load and process data exactly like training
train_identity = pd.read_csv('train_identity.csv')
train_transaction = pd.read_csv('train_transaction.csv')

train = pd.merge(
    train_transaction,
    train_identity,
    on='TransactionID',
    how='left'
)

# Drop ultra-sparse columns
missing_ratio = train.isna().mean()
cols_to_drop = missing_ratio[missing_ratio > 0.90].index
train = train.drop(columns=cols_to_drop)

# Separate target & features
X = train.drop(columns=["isFraud", "TransactionID"])

# Feature engineering
X["Transaction_hour"] = (X["TransactionDT"] / 3600) % 24
X["Transaction_day"] = (X["TransactionDT"] / 86400).astype(int)
X["Transaction_week"] = (X["TransactionDT"] / 604800).astype(int)
X["Log_TransactionAmt"] = np.log1p(X["TransactionAmt"])
X = X.drop(columns=["TransactionDT"])

# Get feature types
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Expected columns:", list(X.columns))
print("\nTotal columns:", len(X.columns))
print("\nNumeric features:", len(num_features))
print("Categorical features:", len(cat_features))

# Save to file for reference
with open('expected_features.txt', 'w') as f:
    f.write('\n'.join(X.columns.tolist()))

print("\nFeature list saved to 'expected_features.txt'")


"""Preprocess `creditcard.csv`: inspect and normalize numeric features.
Saves:
- `creditcard_normalized.csv` (same columns, numeric features scaled)
- `models/scalers.joblib` (fitted ColumnTransformer)
- prints dataset inspection summary
"""
import os
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler

ROOT = r"C:/Documents/project_ML"
INPUT = os.path.join(ROOT, 'creditcard.csv')
OUT_CSV = os.path.join(ROOT, 'creditcard_normalized.csv')
MODELS_DIR = os.path.join(ROOT, 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scalers.joblib')

os.makedirs(MODELS_DIR, exist_ok=True)

print('Loading dataset:', INPUT)
df = pd.read_csv(INPUT)

# Basic inspection
print('\nShape:', df.shape)
print('\nColumns and dtypes:\n', df.dtypes)
print('\nMissing values per column:\n', df.isnull().sum())
print('\nClass counts:\n', df['Class'].value_counts())
print('\nClass ratio (normalized):\n', df['Class'].value_counts(normalize=True))

# Identify PCA columns (V1..V28)
vcols = [c for c in df.columns if c.startswith('V')]
print('\nFound V columns:', len(vcols))

# Columns to scale: V1..V28 and 'Time' and 'Amount'
numeric_cols = vcols.copy()
if 'Time' in df.columns:
    numeric_cols.append('Time')
if 'Amount' in df.columns:
    numeric_cols.append('Amount')

print('\nNumeric columns to scale (sample):', numeric_cols[:6], '... total', len(numeric_cols))

# Build ColumnTransformer: StandardScaler for V and Time, RobustScaler for Amount
transformers = []
if vcols:
    transformers.append(('v_std', StandardScaler(), vcols))
if 'Time' in df.columns:
    transformers.append(('time_std', StandardScaler(), ['Time']))
if 'Amount' in df.columns:
    transformers.append(('amt_robust', RobustScaler(), ['Amount']))

ct = ColumnTransformer(transformers, remainder='passthrough')

# Fit and transform
X = df.copy()
cols_order = list(df.columns)

print('\nFitting scalers and transforming data...')
X_scaled = ct.fit_transform(X)

# ColumnTransformer with remainder='passthrough' may reorder columns; reconstruct DataFrame carefully.
# Determine output column names
out_cols = []
for name, trans, cols in transformers:
    out_cols.extend(cols)
# remainder columns
remainder_cols = [c for c in df.columns if c not in sum([cols for _,_,cols in transformers], [])]
out_cols.extend(remainder_cols)

# X_scaled is numpy array with same number of columns as out_cols
X_scaled_df = pd.DataFrame(X_scaled, columns=out_cols)

# Reorder columns to original order
X_scaled_df = X_scaled_df[cols_order]

# Keep same dtypes for Class
if 'Class' in df.columns:
    X_scaled_df['Class'] = df['Class'].astype(int)

# Save normalized CSV
X_scaled_df.to_csv(OUT_CSV, index=False)
print('\nSaved normalized dataset to:', OUT_CSV)

# Save scaler
joblib.dump(ct, SCALER_PATH)
print('Saved fitted scalers to:', SCALER_PATH)

print('\nSample of normalized data (head):')
print(X_scaled_df.head().to_string())

print('\nDone.')

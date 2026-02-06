"""
TabNet Experiment v12 - Weighted (Class Balancing)
==================================================
Training TabNetClassifier on v12 dataset with Class Balancing (weights=1).
This addresses the 1:14 class imbalance by upsampling the minority class (Winning Horse).
"""
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import joblib
import os
import gc

# Config
EXP_NAME = 'exp_tabnet_v12_weighted'
OUTPUT_DIR = f'models/experiments/{EXP_NAME}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['odds'] > 0) & (df['odds'].notna())].reset_index(drop=True)

# Target
df['target'] = (df['rank'] == 1).astype(int)

# --- Feature Engineering for TabNet ---

# 1. Categorical Features
cat_cols = [
    'jockey_id', 'trainer_id', 'sire_id', 
    'venue', 'grade_code', 'kyoso_joken_code', 
    'surface', 'sex'
]

# Ensure they are strings
for c in cat_cols:
    df[c] = df[c].astype(str)

# Label Encoding
nunique = df[cat_cols].nunique()
types = df[cat_cols].dtypes
print("\nCategorical Features:")
print(nunique)

cat_idxs = []
cat_dims = []
label_encoders = {}

for i, col in enumerate(cat_cols):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    cat_idxs.append(i) # Will be readjusted after reordering columns
    cat_dims.append(len(le.classes_))

# 2. Numerical Features
# Exclude ID columns and targets
exclude_cols = [
    'id', 'race_id', 'horse_id', 'date', 'rank', 'target', 'year', 'is_train',
    'fukusho_rank', 'popularity', 'odds', 'date', 'race_name', 'horse_name',
    'jockey_name', 'trainer_name', 'owner_name', 'cushion_chi', 'gan_suiritsu',
    # dynamic exclusion of raw object cols not in cat_cols
] + cat_cols # cat_cols handled separately

# Get numeric features from LambdaRank list (intersection)
try:
    lgbm_features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()
    # Filter only those present in df
    num_cols = [c for c in lgbm_features if c in df.columns and c not in cat_cols and c not in exclude_cols]
except:
    print("Warning: Could not load LambdaRank features. Using auto-selection.")
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in exclude_cols and c not in cat_cols]

print(f"\nNumerical Features: {len(num_cols)}")
print(f"Categorical Features: {len(cat_cols)}")

# Feature List Order: Categorical + Numerical
# TabNet expects cat features to be specified by index. 
# We will put cat features FIRST to make indices easy (0 to len(cat_cols)-1).
feature_order = cat_cols + num_cols
cat_idxs = [i for i in range(len(cat_cols))]

df_train = df[df['year'] <= 2022].copy()
df_valid = df[df['year'] == 2023].copy()
df_test = df[df['year'] == 2024].copy()

# Fill NaNs in Numerical Features
print("Filling NaNs...")
# Simple mean filling for now (std scaling happens inside TabNet batch norm usually, but filling is needed)
# Using median from train set
fill_values = df_train[num_cols].median()
for col in num_cols:
    val = fill_values[col]
    if pd.isna(val): val = 0
    df_train[col] = df_train[col].fillna(val)
    df_valid[col] = df_valid[col].fillna(val)
    df_test[col] = df_test[col].fillna(val)

X_train = df_train[feature_order].values
y_train = df_train['target'].values
X_valid = df_valid[feature_order].values
y_valid = df_valid['target'].values
X_test = df_test[feature_order].values
y_test = df_test['target'].values

print(f"\nTrain shape: {X_train.shape}")
print(f"Valid shape: {X_valid.shape}")

# Training
clf = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=10, # Keep v2 setting
    lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":50, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax', # or 'sparsemax'
    device_name='auto',
    verbose=1
)

print("\nTraining TabNet (Weighted)...")
# weights=1 enables automatic class balancing
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['auc', 'logloss'],
    max_epochs=100, # Increased epochs for full convergence
    patience=20,
    batch_size=4096,
    virtual_batch_size=256,
    num_workers=0,
    drop_last=False,
    weights=1 # <--- Key Change: Enable Class Balancing
)

# Evaluation
print("\nEvaluation (Test 2024)...")
# Raw probabilities
raw_preds = clf.predict_proba(X_test)[:, 1]

# Normalize per race
df_test['raw_prob'] = raw_preds
def normalize_probs(group):
    p = group['raw_prob']
    return p / p.sum()

df_test['pred_prob'] = df_test.groupby('race_id', group_keys=False).apply(normalize_probs)
auc = roc_auc_score(y_test, df_test['pred_prob'])
print(f"Test AUC (Normalized): {auc:.4f}")

# Save
print("Saving model...")
saved_filepath = clf.save_model(os.path.join(OUTPUT_DIR, 'tabnet_model_weighted'))
joblib.dump(feature_order, os.path.join(OUTPUT_DIR, 'feature_order.pkl'))
joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, 'label_encoders.pkl'))
joblib.dump(fill_values, os.path.join(OUTPUT_DIR, 'fill_values.pkl'))

# ROI Simulation (Simple)
print("\nROI Check (Test 2024) - Normalized")
df_test['ev'] = df_test['pred_prob'] * df_test['odds']

print("Top 1 analysis:")
# Rank by prob within race
df_test['pred_rank'] = df_test.groupby('race_id')['pred_prob'].rank(ascending=False)
top1 = df_test[df_test['pred_rank'] == 1]

wins = top1[top1['target'] == 1]
roi = wins['odds'].sum() / len(top1) * 100
print(f"Top 1 ROI: {roi:.2f}% (Hit Rate: {len(wins)/len(top1)*100:.1f}%)")

print("\nEV Filter (All Horses):")
for ev_thr in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
    bets = df_test[df_test['ev'] >= ev_thr]
    if len(bets) < 10: continue
    w = bets[bets['target'] == 1]
    roi = w['odds'].sum() / len(bets) * 100
    print(f"EV >= {ev_thr}: Bets={len(bets)}, ROI={roi:.1f}%")

print("\nDONE")

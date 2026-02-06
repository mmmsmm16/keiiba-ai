"""
TabNet Experiment v12 - Rank Score Regression
=============================================
Implementation of User Idea: "Weight top 5 and their order, ignore the rest."
Method: Regression
Target: Normalized Rank Score
    Rank 1 -> 1.0
    Rank 2 -> 0.8
    Rank 3 -> 0.6
    Rank 4 -> 0.4
    Rank 5 -> 0.2
    Rank 6+ -> 0.0

This forces the model to distinguish between Rank 1 and 2, but treats Rank 6 and 18 as equal (0).
"""
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score
import joblib
import os
import gc

# Config
EXP_NAME = 'exp_tabnet_v12_rank_score'
OUTPUT_DIR = f'models/experiments/{EXP_NAME}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['odds'] > 0) & (df['odds'].notna())].reset_index(drop=True)

# --- Target Engineering ---
def calculate_rank_score(rank):
    # Score 5 to 0 for Ranks 1 to 6+
    # Normalized to 0.0 - 1.0
    val = max(0, 6 - rank)
    return val / 5.0

df['target_score'] = df['rank'].apply(calculate_rank_score)

print("Target Score Distribution:")
print(df['target_score'].value_counts().sort_index(ascending=False))

# --- Feature Engineering for TabNet ---
cat_cols = [
    'jockey_id', 'trainer_id', 'sire_id', 
    'venue', 'grade_code', 'kyoso_joken_code', 
    'surface', 'sex'
]
for c in cat_cols:
    df[c] = df[c].astype(str)

nunique = df[cat_cols].nunique()
cat_idxs = []
cat_dims = []
label_encoders = {}

for i, col in enumerate(cat_cols):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    cat_idxs.append(i) 
    cat_dims.append(len(le.classes_))

# Numerical Features
exclude_cols = [
    'id', 'race_id', 'horse_id', 'date', 'rank', 'target', 'year', 'is_train',
    'fukusho_rank', 'popularity', 'odds', 'date', 'race_name', 'horse_name',
    'jockey_name', 'trainer_name', 'owner_name', 'cushion_chi', 'gan_suiritsu',
    'target_score'
] + cat_cols

try:
    lgbm_features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()
    num_cols = [c for c in lgbm_features if c in df.columns and c not in cat_cols and c not in exclude_cols]
except:
    print("Warning: Could not load LambdaRank features. Using auto-selection.")
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in exclude_cols and c not in cat_cols]

feature_order = cat_cols + num_cols
cat_idxs = [i for i in range(len(cat_cols))]

df_train = df[df['year'] <= 2022].copy()
df_valid = df[df['year'] == 2023].copy()
df_test = df[df['year'] == 2024].copy()

# Fill NaNs
print("Filling NaNs...")
fill_values = df_train[num_cols].median()
for col in num_cols:
    val = fill_values[col]
    if pd.isna(val): val = 0
    df_train[col] = df_train[col].fillna(val)
    df_valid[col] = df_valid[col].fillna(val)
    df_test[col] = df_test[col].fillna(val)

# Prepare Arrays (Target is now 'target_score' and it must be 2D array [batch, 1] for Regressor)
X_train = df_train[feature_order].values
y_train = df_train['target_score'].values.reshape(-1, 1)
X_valid = df_valid[feature_order].values
y_valid = df_valid['target_score'].values.reshape(-1, 1)
X_test = df_test[feature_order].values
y_test = df_test['target_score'].values.reshape(-1, 1)

print(f"Train/Valid/Test shapes: {X_train.shape}, {X_valid.shape}, {X_test.shape}")
print(f"Target example: {y_train[:5].flatten()}")

# Training
clf = TabNetRegressor(
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
    mask_type='entmax',
    device_name='auto',
    verbose=1
)

print("\nTraining TabNet Regressor...")
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['mse'], # Minimize Mean Squared Error on scores
    max_epochs=100, 
    patience=20,
    batch_size=4096,
    virtual_batch_size=256,
    num_workers=0,
    drop_last=False
)

# Evaluation (Convert regression score back to Prob/Rank for ROI check)
print("\nEvaluation (Test 2024)...")
preds = clf.predict(X_test).flatten()

# Normalize scores to look like probs (optional, but helps EV calculation intuition)
# Softmax? Or just Linear Normalization?
# Since scores are 0-1, we can use softmax to turn them into valid probs for EV.
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

df_test['pred_score'] = preds
df_test['pred_prob'] = df_test.groupby('race_id')['pred_score'].transform(lambda x: softmax(x.values))

# Check AUC (Top 1 vs Rest) just to compare rankabililty
auc = roc_auc_score(df_test['rank'] == 1, df_test['pred_score'])
print(f"Test AUC (on Rank 1): {auc:.4f}")

# Save
print("Saving model...")
saved_filepath = clf.save_model(os.path.join(OUTPUT_DIR, 'tabnet_model_rank_score'))
joblib.dump(feature_order, os.path.join(OUTPUT_DIR, 'feature_order.pkl'))
joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, 'label_encoders.pkl'))
joblib.dump(fill_values, os.path.join(OUTPUT_DIR, 'fill_values.pkl'))

# ROI Simulation
print("\nROI Check (Test 2024) - Rank Score Based")
df_test['ev'] = df_test['pred_prob'] * df_test['odds'] # EV based on Softmax of scores

print("Top 1 analysis:")
df_test['pred_rank'] = df_test.groupby('race_id')['pred_score'].rank(ascending=False)
top1 = df_test[df_test['pred_rank'] == 1]
wins = top1[top1['rank'] == 1]
roi = wins['odds'].sum() / len(top1) * 100
print(f"Top 1 ROI: {roi:.2f}% (Hit Rate: {len(wins)/len(top1)*100:.1f}%)")

print("\nEV Filter (All Horses):")
for ev_thr in [0.8, 1.0, 1.2, 1.5, 2.0, 2.3, 2.5]:
    bets = df_test[df_test['ev'] >= ev_thr]
    cols = bets.columns
    if len(bets) < 10: continue
    w = bets[bets['rank'] == 1]
    roi = w['odds'].sum() / len(bets) * 100
    print(f"EV >= {ev_thr}: Bets={len(bets)}, ROI={roi:.1f}%")

print("\nDONE")

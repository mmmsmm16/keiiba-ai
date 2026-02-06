"""
TabNet Experiment v12 - Optuna Optimization
===========================================
Hyperparameter optimization for TabNetClassifier using Optuna.
Max epochs increased to 1000 with early stopping.
"""
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import optuna
import joblib
import os
import gc

# Config
EXP_NAME = 'exp_tabnet_v12_optuna'
OUTPUT_DIR = f'models/experiments/{EXP_NAME}'
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_TRIALS = 3  # Reduced for speed, as 1 trial takes ~10min

print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['odds'] > 0) & (df['odds'].notna())].reset_index(drop=True)
df['target'] = (df['rank'] == 1).astype(int)

# --- Preprocessing (Same as baseline) ---
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

# Numerical features
exclude_cols = [
    'id', 'race_id', 'horse_id', 'date', 'rank', 'target', 'year', 'is_train',
    'fukusho_rank', 'popularity', 'odds', 'date', 'race_name', 'horse_name',
    'jockey_name', 'trainer_name', 'owner_name', 'cushion_chi', 'gan_suiritsu'
] + cat_cols

try:
    lgbm_features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()
    num_cols = [c for c in lgbm_features if c in df.columns and c not in cat_cols and c not in exclude_cols]
except:
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in exclude_cols and c not in cat_cols]

feature_order = cat_cols + num_cols
cat_idxs = [i for i in range(len(cat_cols))]

df_train = df[df['year'] <= 2022].copy()
df_valid = df[df['year'] == 2023].copy()
df_test = df[df['year'] == 2024].copy()

# Fill NaNs
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

print(f"Train/Valid/Test shapes: {X_train.shape}, {X_valid.shape}, {X_test.shape}")

# --- Optuna Objective ---
def objective(trial):
    # Hyperparameters
    n_d = trial.suggest_int('n_d', 8, 64, step=8)
    n_a = n_d  # Keep n_d = n_a usually
    n_steps = trial.suggest_int('n_steps', 3, 10)
    gamma = trial.suggest_float('gamma', 1.0, 2.0)
    lambda_sparse = trial.suggest_float('lambda_sparse', 1e-6, 1e-1, log=True)
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    cat_emb_dim = trial.suggest_int('cat_emb_dim', 1, 32)
    batch_size = trial.suggest_categorical('batch_size', [2048, 4096, 8192])
    
    clf = TabNetClassifier(
        n_d=n_d, n_a=n_a, n_steps=n_steps,
        gamma=gamma, lambda_sparse=lambda_sparse,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        device_name='auto',
        verbose=1 # Show progress
    )
    
    # Train with early stopping
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=['valid'],
        eval_metric=['auc'],
        max_epochs=1000, 
        patience=30, # Stop if no improvement for 30 epochs
        batch_size=batch_size,
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False
    )
    
    preds = clf.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    return auc

print("\nStarting Optuna Optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)

print("\nBest params:")
print(study.best_params)
best_params = study.best_params

# --- Retrain with Best Params ---
print("\nRetraining with BEST parameters...")
final_clf = TabNetClassifier(
    n_d=best_params['n_d'], n_a=best_params['n_d'], 
    n_steps=best_params['n_steps'],
    gamma=best_params['gamma'], 
    lambda_sparse=best_params['lambda_sparse'],
    cat_idxs=cat_idxs, cat_dims=cat_dims, 
    cat_emb_dim=best_params['cat_emb_dim'],
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=best_params['lr']),
    scheduler_params={"step_size":50, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',
    device_name='auto',
    verbose=1 # Ensure user sees progress
)

final_clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['auc', 'logloss'],
    max_epochs=1000,
    patience=50, # Slightly more patience for final run
    batch_size=best_params['batch_size'],
    virtual_batch_size=256,
    num_workers=0,
    drop_last=False
)

# Evaluation
raw_preds = final_clf.predict_proba(X_test)[:, 1]
df_test['raw_prob'] = raw_preds
def normalize_probs(group):
    p = group['raw_prob']
    return p / p.sum()

df_test['pred_prob'] = df_test.groupby('race_id', group_keys=False).apply(normalize_probs)
auc = roc_auc_score(y_test, df_test['pred_prob'])
print(f"Final Test AUC (Normalized): {auc:.4f}")

# Valid AUC
valid_preds = final_clf.predict_proba(X_valid)[:, 1]
valid_auc = roc_auc_score(y_valid, valid_preds)
print(f"Final Valid AUC: {valid_auc:.4f}")

# Save
final_clf.save_model(os.path.join(OUTPUT_DIR, 'tabnet_model_optuna'))
joblib.dump(feature_order, os.path.join(OUTPUT_DIR, 'feature_order.pkl'))
joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, 'label_encoders.pkl'))
joblib.dump(fill_values, os.path.join(OUTPUT_DIR, 'fill_values.pkl'))
joblib.dump(best_params, os.path.join(OUTPUT_DIR, 'best_params.pkl'))

# Check ROI
print(f"\nFinal ROI Check (Test 2024)")
df_test['ev'] = df_test['pred_prob'] * df_test['odds']
df_test['pred_rank'] = df_test.groupby('race_id')['pred_prob'].rank(ascending=False)

top1 = df_test[df_test['pred_rank'] == 1]
wins = top1[top1['target'] == 1]
roi = wins['odds'].sum() / len(top1) * 100
print(f"Top 1 ROI: {roi:.2f}% (Hit Rate: {len(wins)/len(top1)*100:.1f}%)")

print("\nEV Filter (All Horses):")
for ev_thr in [0.8, 1.0, 1.2, 1.5, 2.0, 2.3, 2.5]:
    bets = df_test[df_test['ev'] >= ev_thr]
    cols = bets.columns
    if len(bets) < 10: continue
    w = bets[bets['target'] == 1]
    roi = w['odds'].sum() / len(bets) * 100
    print(f"EV >= {ev_thr}: Bets={len(bets)}, ROI={roi:.1f}%")

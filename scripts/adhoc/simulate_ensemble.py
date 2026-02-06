"""
Ensemble: LambdaRank + TabNet (Simplified)
==========================================
Uses pre-computed TabNet v2 model predictions.
Optimized preprocessing for speed.
"""
import pandas as pd
import numpy as np
import joblib
import os
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

# Load data
print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df_test = df[df['year'] == 2024].copy()
df_test = df_test[(df_test['odds'] > 0) & (df_test['odds'].notna())].reset_index(drop=True)
df_test['target'] = (df_test['rank'] == 1).astype(int)

print(f"Test Data: {len(df_test)} rows")

# --- 1. LambdaRank Predictions ---
print("\n=== Loading LambdaRank ===")
lgbm_model_path = 'models/experiments/exp_lambdarank_v12_batch4_optuna/model.pkl'
lgbm_features_path = 'models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv'

model_lgbm = joblib.load(lgbm_model_path)
features_lgbm = pd.read_csv(lgbm_features_path)['0'].tolist()

# Predict
lgbm_scores = model_lgbm.predict(df_test[features_lgbm].values)
df_test['lgbm_score'] = lgbm_scores

# Normalize per race using softmax
df_test['lgbm_prob'] = df_test.groupby('race_id')['lgbm_score'].transform(lambda x: softmax(x.values))
lgbm_auc = roc_auc_score(df_test['target'], df_test['lgbm_prob'])
print(f"LambdaRank AUC: {lgbm_auc:.4f}")

# --- 2. TabNet Predictions (Using v2 Baseline) ---
print("\n=== Loading TabNet (v2 Baseline) ===")
tabnet_dir = 'models/experiments/exp_tabnet_v12_baseline'
tabnet_zip = os.path.join(tabnet_dir, 'tabnet_model_v2.zip')

# Load artifacts
feature_order = joblib.load(os.path.join(tabnet_dir, 'feature_order.pkl'))
label_encoders = joblib.load(os.path.join(tabnet_dir, 'label_encoders.pkl'))
fill_values = joblib.load(os.path.join(tabnet_dir, 'fill_values.pkl'))

# Preprocess TabNet Test Data (Vectorized)
df_tab = df_test.copy()

# Categorical (Vectorized)
cat_cols = [c for c in feature_order if c in label_encoders]
for col in cat_cols:
    le = label_encoders[col]
    known_map = {v: i for i, v in enumerate(le.classes_)}
    df_tab[col] = df_tab[col].astype(str).map(known_map).fillna(0).astype(int)

# Numerical
num_cols = [c for c in feature_order if c not in cat_cols]
for col in num_cols:
    val = fill_values.get(col, 0)
    if pd.isna(val): val = 0
    df_tab[col] = df_tab[col].fillna(val)

X_test_tab = df_tab[feature_order].values.astype(np.float32)

# Predict
print("Loading TabNet model...")
clf = TabNetClassifier()
clf.load_model(tabnet_zip)
print("Predicting with TabNet...")
tab_probs = clf.predict_proba(X_test_tab)[:, 1]
df_test['tab_prob_raw'] = tab_probs

# Normalize per race
df_test['tab_prob'] = df_test.groupby('race_id')['tab_prob_raw'].transform(lambda x: x / x.sum())
tab_auc = roc_auc_score(df_test['target'], df_test['tab_prob'])
print(f"TabNet AUC: {tab_auc:.4f}")

# --- 3. Ensemble with Various Weights ---
print("\n=== Ensemble Results ===")
print(f"{'Weight (LGBM)':<14} {'AUC':<8} {'Top1 ROI':<10} {'Hit Rate':<10}")
print("-" * 50)

for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    # Combine (using normalized probabilities)
    df_test['ens_prob'] = alpha * df_test['lgbm_prob'] + (1 - alpha) * df_test['tab_prob']
    
    # Normalize again
    df_test['ens_prob_norm'] = df_test.groupby('race_id')['ens_prob'].transform(lambda x: x / x.sum())
    
    auc = roc_auc_score(df_test['target'], df_test['ens_prob_norm'])
    
    # Top 1 ROI
    df_test['ens_rank'] = df_test.groupby('race_id')['ens_prob_norm'].rank(ascending=False)
    top1 = df_test[df_test['ens_rank'] == 1]
    wins = top1[top1['target'] == 1]
    roi = wins['odds'].sum() / len(top1) * 100
    hit_rate = len(wins) / len(top1) * 100
    
    print(f"{alpha:<14.1f} {auc:<8.4f} {roi:<10.2f}% {hit_rate:<10.1f}%")

print("\n=== Best Ensemble EV Filter Check ===")
# Test with α=0.5 (equal weight)
alpha = 0.5
df_test['ens_prob'] = alpha * df_test['lgbm_prob'] + (1 - alpha) * df_test['tab_prob']
df_test['ens_prob_norm'] = df_test.groupby('race_id')['ens_prob'].transform(lambda x: x / x.sum())
df_test['ens_ev'] = df_test['ens_prob_norm'] * df_test['odds']

print(f"\nEV Filter (α={alpha}):")
for ev_thr in [0.8, 1.0, 1.2, 1.5, 2.0, 2.3, 2.5]:
    bets = df_test[df_test['ens_ev'] >= ev_thr]
    if len(bets) < 10: continue
    w = bets[bets['target'] == 1]
    roi = w['odds'].sum() / len(bets) * 100
    print(f"EV >= {ev_thr}: Bets={len(bets)}, ROI={roi:.1f}%")

print("\nDONE")

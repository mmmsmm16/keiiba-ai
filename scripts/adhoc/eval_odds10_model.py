"""
Evaluate Odds10-Weighted LambdaRank Model
==========================================
Tests ROI performance of the new model with odds_10min weighting.
"""
import pandas as pd
import numpy as np
import joblib
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

# Load NEW Model (Odds10-Weighted)
print("\n=== Loading Odds10-Weighted LambdaRank ===")
new_model_dir = 'models/experiments/exp_lambdarank_v12_batch4_optuna_odds10_weighted'
model_new = joblib.load(f'{new_model_dir}/model.pkl')
features_new = pd.read_csv(f'{new_model_dir}/features.csv')['0'].tolist()

# Predict
new_scores = model_new.predict(df_test[features_new].values)
df_test['new_score'] = new_scores
df_test['new_prob'] = df_test.groupby('race_id')['new_score'].transform(lambda x: softmax(x.values))

new_auc = roc_auc_score(df_test['target'], df_test['new_prob'])
print(f"NEW Model AUC: {new_auc:.4f}")

# Load OLD Model (Baseline)
print("\n=== Loading Baseline LambdaRank (for comparison) ===")
old_model_dir = 'models/experiments/exp_lambdarank_v12_batch4_optuna'
model_old = joblib.load(f'{old_model_dir}/model.pkl')
features_old = pd.read_csv(f'{old_model_dir}/features.csv')['0'].tolist()

old_scores = model_old.predict(df_test[features_old].values)
df_test['old_score'] = old_scores
df_test['old_prob'] = df_test.groupby('race_id')['old_score'].transform(lambda x: softmax(x.values))

old_auc = roc_auc_score(df_test['target'], df_test['old_prob'])
print(f"OLD Model AUC: {old_auc:.4f}")

# --- ROI Comparison ---
print("\n=== ROI Comparison ===")
print(f"{'Model':<20} {'AUC':<8} {'Top1 ROI':<10} {'Hit Rate':<10}")
print("-" * 55)

for name, score_col, prob_col in [('OLD (Baseline)', 'old_score', 'old_prob'), ('NEW (Odds10-Wgt)', 'new_score', 'new_prob')]:
    df_test['pred_rank'] = df_test.groupby('race_id')[score_col].rank(ascending=False)
    top1 = df_test[df_test['pred_rank'] == 1]
    wins = top1[top1['target'] == 1]
    roi = wins['odds'].sum() / len(top1) * 100
    hit_rate = len(wins) / len(top1) * 100
    auc = roc_auc_score(df_test['target'], df_test[prob_col])
    print(f"{name:<20} {auc:<8.4f} {roi:<10.2f}% {hit_rate:<10.1f}%")

# --- EV Filter Check ---
print("\n=== EV Filter Check (NEW Model) ===")
df_test['ev'] = df_test['new_prob'] * df_test['odds']

for ev_thr in [0.8, 1.0, 1.2, 1.5, 2.0, 2.3, 2.5]:
    bets = df_test[df_test['ev'] >= ev_thr]
    if len(bets) < 10: continue
    w = bets[bets['target'] == 1]
    roi = w['odds'].sum() / len(bets) * 100
    print(f"EV >= {ev_thr}: Bets={len(bets)}, ROI={roi:.1f}%")

# --- EV Filter with Odds Cap ---
print("\n=== EV + Odds Cap Check (NEW Model) ===")
for ev_thr in [2.0, 2.3, 2.5]:
    for max_odds in [10, 15, 20]:
        bets = df_test[(df_test['ev'] >= ev_thr) & (df_test['odds'] <= max_odds)]
        if len(bets) < 10: continue
        w = bets[bets['target'] == 1]
        roi = w['odds'].sum() / len(bets) * 100
        print(f"EV >= {ev_thr}, Odds <= {max_odds}: Bets={len(bets)}, ROI={roi:.1f}%")

print("\nDONE")

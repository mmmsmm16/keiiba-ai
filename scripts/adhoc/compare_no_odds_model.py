"""
Compare predictions between models with and without odds features.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

# Load test data
df = pd.read_parquet('data/temp_t2/T2_features.parquet')
tgt = pd.read_parquet('data/temp_t2/T2_targets.parquet')

df['race_id'] = df['race_id'].astype(str)
tgt['race_id'] = tgt['race_id'].astype(str)
df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce').fillna(0).astype(int)
tgt['horse_number'] = pd.to_numeric(tgt['horse_number'], errors='coerce').fillna(0).astype(int)

merged = pd.merge(df, tgt, on=['race_id', 'horse_number'])
merged['date'] = pd.to_datetime(merged['date'])

# Use December 2024 test data
test_df = merged[(merged['date'] >= '2024-12-01')].copy()
n_races = test_df['race_id'].nunique()
print(f'Test set (Dec 2024): {len(test_df)} records, {n_races} races')

# Load models
m_with = joblib.load('models/experiments/exp_t2_track_bias/model.pkl')
m_no = joblib.load('models/experiments/exp_t2_no_odds/model.pkl')

# Prepare features
feats_with = m_with.feature_name()
feats_no = m_no.feature_name()

# Fill missing
for f in feats_with:
    if f not in test_df.columns:
        test_df[f] = 0
for f in feats_no:
    if f not in test_df.columns:
        test_df[f] = 0

# Convert to numpy for prediction (avoid categorical issues)
X_with = test_df[feats_with].copy()
X_no = test_df[feats_no].copy()

for col in X_with.columns:
    if X_with[col].dtype == 'object' or str(X_with[col].dtype) == 'category':
        X_with[col] = pd.to_numeric(X_with[col].astype(str), errors='coerce').fillna(0)
    if X_with[col].isna().any():
        X_with[col] = X_with[col].fillna(0)

for col in X_no.columns:
    if X_no[col].dtype == 'object' or str(X_no[col].dtype) == 'category':
        X_no[col] = pd.to_numeric(X_no[col].astype(str), errors='coerce').fillna(0)
    if X_no[col].isna().any():
        X_no[col] = X_no[col].fillna(0)

# Predictions
test_df['pred_with_odds'] = m_with.predict(X_with.values)
test_df['pred_no_odds'] = m_no.predict(X_no.values)
test_df['is_win'] = (test_df['rank'] == 1).astype(int)

# Compare AUC
auc_with = roc_auc_score(test_df['is_win'], test_df['pred_with_odds'])
auc_no = roc_auc_score(test_df['is_win'], test_df['pred_no_odds'])

print()
print('=== AUC Comparison (December 2024) ===')
print(f'With bias_adversity: {auc_with:.4f}')
print(f'Without (No-Odds):   {auc_no:.4f}')
print(f'Difference:          {auc_with - auc_no:.4f}')

# Correlation
corr = test_df['pred_with_odds'].corr(test_df['pred_no_odds'])
print(f'\nPrediction Correlation: {corr:.4f}')

# Normalize predictions per race
test_df['norm_with'] = test_df.groupby('race_id')['pred_with_odds'].transform(lambda x: x / x.sum())
test_df['norm_no'] = test_df.groupby('race_id')['pred_no_odds'].transform(lambda x: x / x.sum())

# Calculate rank within race
test_df['pred_rank_with'] = test_df.groupby('race_id')['pred_with_odds'].rank(ascending=False)
test_df['pred_rank_no'] = test_df.groupby('race_id')['pred_no_odds'].rank(ascending=False)

# Top-1 Accuracy
wins = (test_df['rank'] == 1).sum()
top1_with = ((test_df['pred_rank_with'] == 1) & (test_df['rank'] == 1)).sum()
top1_no = ((test_df['pred_rank_no'] == 1) & (test_df['rank'] == 1)).sum()

print()
print('=== Top-1 Prediction Accuracy ===')
print(f'With bias_adversity: {top1_with}/{wins} = {100*top1_with/wins:.2f}%')
print(f'Without (No-Odds):   {top1_no}/{wins} = {100*top1_no/wins:.2f}%')

# Show a real race comparison with variance
race_ids = test_df['race_id'].unique()
for race_id in race_ids[10:11]:
    race = test_df[test_df['race_id'] == race_id].copy()
    race = race.sort_values('pred_rank_no')
    print()
    print(f'=== Sample Race: {race_id} ===')
    cols = ['horse_number', 'rank', 'pred_with_odds', 'pred_no_odds', 'pred_rank_with', 'pred_rank_no']
    print(race[cols].to_string(index=False))


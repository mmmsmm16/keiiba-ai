"""
Compare all 3 model variants: With IDs, No Odds, No IDs
"""
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

# Load all 3 models
models = {
    'With Odds + IDs': 'models/experiments/exp_t2_track_bias/model.pkl',
    'No Odds, With IDs': 'models/experiments/exp_t2_no_odds/model.pkl',
    'No Odds, No IDs': 'models/experiments/exp_t2_no_ids/model.pkl'
}

print('=== Model Comparison ===')
for name, path in models.items():
    m = joblib.load(path)
    n_feats = len(m.feature_name())
    print(f'{name}: {n_feats} features')

# Load test data
df = pd.read_parquet('data/temp_t2/T2_features.parquet')
tgt = pd.read_parquet('data/temp_t2/T2_targets.parquet')
df['race_id'] = df['race_id'].astype(str)
tgt['race_id'] = tgt['race_id'].astype(str)
merged = pd.merge(df, tgt, on=['race_id', 'horse_number'])
merged['date'] = pd.to_datetime(merged['date'])
test_df = merged[merged['date'] >= '2024-12-01'].copy()

n_races = test_df['race_id'].nunique()
print()
print(f'Test set: {len(test_df)} records, {n_races} races')

# Evaluate each
print()
print('=== AUC Comparison (Dec 2024) ===')
results = {}
for name, path in models.items():
    m = joblib.load(path)
    feats = m.feature_name()
    
    # Fill missing features with 0
    for f in feats:
        if f not in test_df.columns:
            test_df[f] = 0
    
    X = test_df[feats].copy()
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            X[col] = pd.to_numeric(X[col].astype(str), errors='coerce').fillna(0)
        X[col] = X[col].fillna(0)
    y = (test_df['rank'] == 1).astype(int)
    preds = m.predict(X.values)
    auc = roc_auc_score(y, preds)
    results[name] = auc
    print(f'{name}: AUC = {auc:.4f}')

# Summary
print()
print('=== Summary ===')
base = results['With Odds + IDs']
for name, auc in results.items():
    diff = auc - base
    diff_str = f'+{diff:.4f}' if diff >= 0 else f'{diff:.4f}'
    print(f'{name}: {diff_str} vs baseline')

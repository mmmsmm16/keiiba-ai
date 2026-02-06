"""Check calibration of model predictions"""
import pandas as pd
import numpy as np
import joblib

# Load model and data
print("Loading model...")
model = joblib.load('models/experiments/exp_t2_refined_v3/model.pkl')
print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
targets = pd.read_parquet('data/temp_t2/T2_targets.parquet')

df['race_id'] = df['race_id'].astype(str)
targets['race_id'] = targets['race_id'].astype(str)
df = df.merge(targets[['race_id', 'horse_number', 'rank']], on=['race_id', 'horse_number'], how='left')
df['date'] = pd.to_datetime(df['date'])

# 2024 test set
df_test = df[df['date'].dt.year == 2024].copy()
print(f"Test set: {len(df_test)} records")

feature_names = model.feature_name()
X = df_test[feature_names].copy()
for c in X.columns:
    if X[c].dtype == 'object' or X[c].dtype.name == 'category':
        X[c] = X[c].astype('category').cat.codes
    X[c] = X[c].fillna(-999)

# Predict
print("Predicting...")
raw_preds = model.predict(X.values.astype(np.float64))
df_test['pred'] = raw_preds
df_test['pred_norm'] = df_test.groupby('race_id')['pred'].transform(lambda x: x / x.sum())

# Calibration check: bin by predicted prob, compare to actual win rate
df_test['is_win'] = (df_test['rank'] == 1).astype(int)
df_test['pred_bin'] = pd.cut(df_test['pred_norm'], bins=[0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0])

calib = df_test.groupby('pred_bin', observed=True).agg(
    count=('is_win', 'count'),
    actual_rate=('is_win', 'mean'),
    pred_mean=('pred_norm', 'mean')
).reset_index()

print('\n' + '='*60)
print('Calibration Check (2024 Test Set) - Win Model')
print('='*60)
print(f"{'Pred Bin':<20} | {'Count':<8} | {'Actual%':<10} | {'Pred%':<10} | {'Gap':<10}")
print('-'*60)
for _, row in calib.iterrows():
    gap = (row['pred_mean'] - row['actual_rate']) * 100
    sign = '+' if gap >= 0 else ''
    print(f"{str(row['pred_bin']):<20} | {row['count']:<8} | {row['actual_rate']*100:<9.1f}% | {row['pred_mean']*100:<9.1f}% | {sign}{gap:.1f}%")

# Summary
print('\n' + '='*60)
print('Summary')
print('='*60)
mean_pred = df_test['pred_norm'].mean()
mean_actual = df_test['is_win'].mean()
print(f"Mean predicted prob: {mean_pred*100:.2f}%")
print(f"Mean actual win rate: {mean_actual*100:.2f}%")
print(f"Gap: {(mean_pred - mean_actual)*100:+.2f}%")

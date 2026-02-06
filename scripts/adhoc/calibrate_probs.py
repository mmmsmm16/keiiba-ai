"""
Probability Calibration for EV Betting
=======================================
Calibrates model probabilities using isotonic regression
to match actual win rates, then recalculates EV.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression

# Load data
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['odds'] > 0) & (df['odds'].notna())].copy()
df['is_win'] = (df['rank'] == 1).astype(int)

# Split: 2019-2022 train, 2023 calibration, 2024 test
train_df = df[df['year'] <= 2022].copy()
calib_df = df[df['year'] == 2023].copy()
test_df = df[df['year'] == 2024].copy()

print(f"Train: {len(train_df)}, Calib: {len(calib_df)}, Test: {len(test_df)}")

# Load model and predict
model = joblib.load('models/experiments/exp_lambdarank_v12_batch4_optuna/model.pkl')
features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()

def softmax_probs(group):
    scores = group['pred_score'].values
    exp_scores = np.exp(scores - np.max(scores))
    return pd.Series(exp_scores / exp_scores.sum(), index=group.index)

# Predict and get raw probs
for df_subset in [calib_df, test_df]:
    y_pred = model.predict(df_subset[features].values)
    df_subset['pred_score'] = y_pred
    df_subset['raw_prob'] = df_subset.groupby('race_id', group_keys=False).apply(softmax_probs)

# Calibration on 2023 data
print("\n=== Calibration (Isotonic Regression on 2023 data) ===")
iso_reg = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
iso_reg.fit(calib_df['raw_prob'].values, calib_df['is_win'].values)

# Apply calibration to test data
test_df['calib_prob'] = iso_reg.predict(test_df['raw_prob'].values)
test_df['calib_ev'] = test_df['calib_prob'] * test_df['odds']
test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(ascending=False)

# Check calibration quality
print("\nCalibration check (binned):")
test_df['prob_bin'] = pd.cut(test_df['calib_prob'], bins=10)
calib_check = test_df.groupby('prob_bin', observed=True).agg({
    'is_win': ['mean', 'count'],
    'calib_prob': 'mean'
}).round(3)
print(calib_check)

# EV simulation with calibrated probs
print("\n=== EV Simulation with Calibrated Probs ===")

# Top 1 by model, filter by calibrated EV
top1 = test_df[test_df['pred_rank'] == 1].copy()
print(f"Top1 calib_ev stats: min={top1['calib_ev'].min():.2f}, max={top1['calib_ev'].max():.2f}, mean={top1['calib_ev'].mean():.2f}")

for min_ev in [0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.5]:
    for max_odds in [10, 15, 20, 30, 999]:
        bets = top1[(top1['calib_ev'] >= min_ev) & (top1['odds'] <= max_odds)]
        if len(bets) < 30:
            continue
        wins = bets[bets['is_win'] == 1]
        roi = wins['odds'].sum() * 100 / (len(bets) * 100) * 100
        marker = ' <-- PROFIT!' if roi >= 100 else ''
        print(f'EV>={min_ev} MaxOdds={max_odds:3d}: Bets={len(bets):4d}, Wins={len(wins):3d}, HitRate={len(wins)/len(bets)*100:5.1f}%, ROI={roi:6.1f}%{marker}')

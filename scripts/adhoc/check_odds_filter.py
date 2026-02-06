"""Check and filter zero odds"""
import pandas as pd
import numpy as np
import joblib

df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
test_df = df[df['year'] == 2024].copy()

print('=== Odds Analysis ===')
print('Total rows:', len(test_df))
print('Total races:', test_df['race_id'].nunique())

# Check for zero or null odds
zero_odds = test_df[test_df['odds'] == 0]
null_odds = test_df[test_df['odds'].isna()]
print('Rows with odds = 0:', len(zero_odds))
print('Rows with odds = NaN:', len(null_odds))

# Races with any zero odds
races_with_zero = zero_odds['race_id'].nunique()
print('Races with any odds = 0:', races_with_zero)

# After excluding
valid_df = test_df[(test_df['odds'] > 0) & (test_df['odds'].notna())]
valid_races = valid_df['race_id'].nunique()
print()
print('After excluding odds <= 0:')
print('Valid rows:', len(valid_df))
print('Valid races:', valid_races)

# Now run EV simulation with filtered data
print()
print('=== EV Simulation (filtered) ===')
model = joblib.load('models/experiments/exp_lambdarank_v12_batch4_optuna/model.pkl')
features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()

y_pred = model.predict(valid_df[features].values)
valid_df['pred_score'] = y_pred

def softmax_probs(group):
    scores = group['pred_score'].values
    exp_scores = np.exp(scores - np.max(scores))
    return pd.Series(exp_scores / exp_scores.sum(), index=group.index)

valid_df['win_prob'] = valid_df.groupby('race_id', group_keys=False).apply(softmax_probs)
valid_df['ev'] = valid_df['win_prob'] * valid_df['odds']
valid_df['is_win'] = (valid_df['rank'] == 1).astype(int)

# EV grid search
for min_ev in [1.0, 1.5, 2.0, 2.3, 2.5, 3.0]:
    for max_odds in [10, 15, 20, 30, 999]:
        ev_bets = valid_df[(valid_df['ev'] >= min_ev) & (valid_df['odds'] <= max_odds)].copy()
        if len(ev_bets) < 20:
            continue
        wins = ev_bets[ev_bets['is_win'] == 1]
        roi = wins['odds'].sum() * 100 / (len(ev_bets) * 100) * 100
        profit = wins['odds'].sum() * 100 - len(ev_bets) * 100
        marker = ' ✓' if roi >= 100 else ''
        print(f'EV>={min_ev} MaxOdds={max_odds:3d}: Bets={len(ev_bets):4d}, Wins={len(wins):3d}, ROI={roi:6.1f}%{marker}')

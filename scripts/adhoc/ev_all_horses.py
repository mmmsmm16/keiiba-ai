"""EV filtering across all horses"""
import pandas as pd
import numpy as np
import joblib

df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['is_win'] = (df['rank'] == 1).astype(int)
test_df = df[df['year'] == 2024].copy()

model = joblib.load('models/experiments/exp_lambdarank_v12_batch4_optuna/model.pkl')
features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()
y_pred = model.predict(test_df[features].values)
test_df['pred_score'] = y_pred

# Softmax per race
def softmax_probs(group):
    scores = group['pred_score'].values
    exp_scores = np.exp(scores - np.max(scores))
    return pd.Series(exp_scores / exp_scores.sum(), index=group.index)

test_df['win_prob'] = test_df.groupby('race_id', group_keys=False).apply(softmax_probs)
test_df['ev'] = test_df['win_prob'] * test_df['odds']
test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(ascending=False)

print(f'All horses EV stats:')
print(f'  min={test_df["ev"].min():.3f}, max={test_df["ev"].max():.3f}, mean={test_df["ev"].mean():.3f}')
print()

# Grid search EV thresholds (all horses)
for min_ev in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
    for max_odds in [10, 20, 50, 999]:
        ev_bets = test_df[(test_df['ev'] >= min_ev) & (test_df['odds'] <= max_odds)].copy()
        if len(ev_bets) < 50:
            continue
        
        total_bet = len(ev_bets) * 100
        wins = ev_bets[ev_bets['is_win'] == 1]
        returns = wins['odds'].sum() * 100
        roi = returns / total_bet * 100
        hit_rate = len(wins) / len(ev_bets) * 100
        
        print(f'EV>={min_ev} MaxOdds={max_odds}: Bets={len(ev_bets):4d}, Wins={len(wins):3d}, HitRate={hit_rate:5.1f}%, ROI={roi:6.1f}%')

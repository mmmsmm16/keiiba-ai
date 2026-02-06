"""Grid Search ROI - Top 1 Only"""
import pandas as pd
import numpy as np
import joblib
import itertools

# Load data and model
print('Loading data and model...')
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['is_win'] = (df['rank'] == 1).astype(int)

test_df = df[df['year'] == 2024].copy()

model = joblib.load('models/experiments/exp_lambdarank_v12_batch4_optuna/model.pkl')
features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()

X_test = test_df[features].values
y_pred = model.predict(X_test)
test_df['pred_score'] = y_pred
test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(ascending=False)

# Normalize per race
test_df['pred_prob'] = test_df.groupby('race_id')['pred_score'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
)

# TOP 1 only
top1_df = test_df[test_df['pred_rank'] == 1].copy()
print(f'Top 1 predictions: {len(top1_df)} races')

# Grid Search
MAX_ODDS_LIST = [3, 5, 7, 10, 15, 20, 30, 50, 100, 999]
MIN_CONF_LIST = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

results = []

for max_odds, min_conf in itertools.product(MAX_ODDS_LIST, MIN_CONF_LIST):
    bets = top1_df[(top1_df['pred_prob'] >= min_conf) & (top1_df['odds'] <= max_odds)].copy()
    
    if len(bets) < 50:
        continue
    
    total_bet = len(bets) * 100
    wins = bets[bets['is_win'] == 1]
    returns = wins['odds'].sum() * 100
    roi = returns / total_bet * 100
    hit_rate = len(wins) / len(bets) * 100
    
    results.append({
        'max_odds': max_odds,
        'min_conf': min_conf,
        'num_bets': len(bets),
        'num_wins': len(wins),
        'hit_rate': hit_rate,
        'roi': roi,
        'profit': returns - total_bet
    })

results_df = pd.DataFrame(results).sort_values('roi', ascending=False)

print()
print('=' * 80)
print('TOP 20 STRATEGIES (TOP 1 ONLY)')
print('=' * 80)
print(results_df.head(20).to_string(index=False))

print()
print('=' * 80)
print('PROFITABLE STRATEGIES (ROI >= 100%)')
print('=' * 80)
profitable = results_df[results_df['roi'] >= 100]
if len(profitable) > 0:
    print(profitable.to_string(index=False))
else:
    print('No profitable strategies found')

print()
print('=' * 80)
print('BEST STABLE STRATEGY (>= 1000 bets)')
print('=' * 80)
stable = results_df[results_df['num_bets'] >= 1000].head(1)
if len(stable) > 0:
    row = stable.iloc[0]
    print('Max Odds:', row['max_odds'])
    print('Min Conf:', row['min_conf'])
    print('Bets:', int(row['num_bets']))
    print('Wins:', int(row['num_wins']))
    print('Hit Rate:', f"{row['hit_rate']:.2f}%")
    print('ROI:', f"{row['roi']:.2f}%")
    print('Profit:', f"{int(row['profit']):,}")

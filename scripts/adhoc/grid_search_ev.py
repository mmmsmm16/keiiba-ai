"""Grid Search ROI with EV (Expected Value) Filtering"""
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

# Convert scores to probabilities via softmax per race
def softmax_probs(group):
    scores = group['pred_score'].values
    exp_scores = np.exp(scores - np.max(scores))  # numerical stability
    probs = exp_scores / exp_scores.sum()
    return pd.Series(probs, index=group.index)

test_df['win_prob'] = test_df.groupby('race_id', group_keys=False).apply(softmax_probs)
test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(ascending=False)

# Calculate Expected Value
test_df['ev'] = test_df['win_prob'] * test_df['odds']

print(f'Test Set: {len(test_df)} rows, {test_df["race_id"].nunique()} races')
print(f'EV stats: min={test_df["ev"].min():.3f}, max={test_df["ev"].max():.3f}, mean={test_df["ev"].mean():.3f}')

# TOP 1 only
top1_df = test_df[test_df['pred_rank'] == 1].copy()
print(f'Top 1 predictions: {len(top1_df)} races')
print(f'Top 1 EV stats: min={top1_df["ev"].min():.3f}, max={top1_df["ev"].max():.3f}, mean={top1_df["ev"].mean():.3f}')

# Grid Search with EV filter
MIN_EV_LIST = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
MAX_ODDS_LIST = [5, 10, 15, 20, 30, 50, 100, 999]

results = []

for min_ev, max_odds in itertools.product(MIN_EV_LIST, MAX_ODDS_LIST):
    bets = top1_df[(top1_df['ev'] >= min_ev) & (top1_df['odds'] <= max_odds)].copy()
    
    if len(bets) < 20:
        continue
    
    total_bet = len(bets) * 100
    wins = bets[bets['is_win'] == 1]
    returns = wins['odds'].sum() * 100
    roi = returns / total_bet * 100
    hit_rate = len(wins) / len(bets) * 100
    avg_odds = bets['odds'].mean()
    avg_ev = bets['ev'].mean()
    
    results.append({
        'min_ev': min_ev,
        'max_odds': max_odds,
        'num_bets': len(bets),
        'num_wins': len(wins),
        'hit_rate': hit_rate,
        'avg_odds': avg_odds,
        'avg_ev': avg_ev,
        'roi': roi,
        'profit': returns - total_bet
    })

results_df = pd.DataFrame(results).sort_values('roi', ascending=False)

print()
print('=' * 90)
print('TOP 20 EV-FILTERED STRATEGIES')
print('=' * 90)
print(results_df.head(20).to_string(index=False))

print()
print('=' * 90)
print('PROFITABLE STRATEGIES (ROI >= 100%)')
print('=' * 90)
profitable = results_df[results_df['roi'] >= 100]
if len(profitable) > 0:
    print(profitable.to_string(index=False))
else:
    print('No profitable strategies found')

print()
print('=' * 90)
print('BEST STRATEGY BY PROFIT (>= 100 bets)')
print('=' * 90)
stable = results_df[results_df['num_bets'] >= 100].sort_values('profit', ascending=False).head(1)
if len(stable) > 0:
    row = stable.iloc[0]
    print('Min EV:', row['min_ev'])
    print('Max Odds:', row['max_odds'])
    print('Bets:', int(row['num_bets']))
    print('Wins:', int(row['num_wins']))
    print('Hit Rate:', f"{row['hit_rate']:.2f}%")
    print('Avg Odds:', f"{row['avg_odds']:.2f}")
    print('Avg EV:', f"{row['avg_ev']:.3f}")
    print('ROI:', f"{row['roi']:.2f}%")
    print('Profit:', f"{int(row['profit']):,}")

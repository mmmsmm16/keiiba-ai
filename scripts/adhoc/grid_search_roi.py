"""
Grid Search ROI Simulation for v12 Optuna Model
================================================
Tests various betting parameters to find optimal strategy.
"""
import pandas as pd
import numpy as np
import joblib
import itertools
from tqdm import tqdm

# Load data and model
print("Loading data and model...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['is_win'] = (df['rank'] == 1).astype(int)
df['is_top3'] = (df['rank'] <= 3).astype(int)

test_df = df[df['year'] == 2024].copy()

model = joblib.load('models/experiments/exp_lambdarank_v12_batch4_optuna/model.pkl')
features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()

X_test = test_df[features].values
y_pred = model.predict(X_test)
test_df['pred_score'] = y_pred
test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(ascending=False)

# Normalize pred_score per race for confidence
test_df['pred_prob'] = test_df.groupby('race_id')['pred_score'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
)

print(f"Test Set: {len(test_df)} rows, {test_df['race_id'].nunique()} races")

# Grid Search Parameters
MAX_ODDS_LIST = [5, 10, 15, 20, 30, 50, 100, 999]
MIN_CONFIDENCE_LIST = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
TOP_N_LIST = [1, 2, 3]
BET_TYPE_LIST = ['win', 'top3']  # win = 単勝, top3 = 複勝

results = []

print("\nRunning Grid Search...")
total = len(MAX_ODDS_LIST) * len(MIN_CONFIDENCE_LIST) * len(TOP_N_LIST) * len(BET_TYPE_LIST)

for max_odds, min_conf, top_n, bet_type in tqdm(itertools.product(
    MAX_ODDS_LIST, MIN_CONFIDENCE_LIST, TOP_N_LIST, BET_TYPE_LIST
), total=total):
    
    # Filter bets
    bets = test_df[
        (test_df['pred_rank'] <= top_n) &
        (test_df['pred_prob'] >= min_conf) &
        (test_df['odds'] <= max_odds)
    ].copy()
    
    if len(bets) == 0:
        continue
    
    total_bet = len(bets) * 100
    
    if bet_type == 'win':
        # Win bet: only wins count
        wins = bets[bets['is_win'] == 1]
        returns = wins['odds'].sum() * 100
    else:
        # Top3 bet: need place odds (approximate as odds / 3)
        wins = bets[bets['is_top3'] == 1]
        returns = (wins['odds'] / 3).sum() * 100
    
    roi = returns / total_bet * 100 if total_bet > 0 else 0
    hit_rate = len(wins) / len(bets) * 100 if len(bets) > 0 else 0
    
    results.append({
        'max_odds': max_odds,
        'min_confidence': min_conf,
        'top_n': top_n,
        'bet_type': bet_type,
        'num_bets': len(bets),
        'num_wins': len(wins),
        'hit_rate': hit_rate,
        'total_bet': total_bet,
        'returns': returns,
        'roi': roi,
        'profit': returns - total_bet
    })

results_df = pd.DataFrame(results)

# Filter to meaningful results (at least 100 bets)
results_df = results_df[results_df['num_bets'] >= 100].sort_values('roi', ascending=False)

print("\n" + "=" * 80)
print("TOP 20 STRATEGIES BY ROI (単勝/Win)")
print("=" * 80)
win_results = results_df[results_df['bet_type'] == 'win'].head(20)
print(win_results[['max_odds', 'min_confidence', 'top_n', 'num_bets', 'hit_rate', 'roi', 'profit']].to_string(index=False))

print("\n" + "=" * 80)
print("TOP 20 STRATEGIES BY ROI (複勝/Top3)")
print("=" * 80)
top3_results = results_df[results_df['bet_type'] == 'top3'].head(20)
print(top3_results[['max_odds', 'min_confidence', 'top_n', 'num_bets', 'hit_rate', 'roi', 'profit']].to_string(index=False))

print("\n" + "=" * 80)
print("PROFITABLE STRATEGIES (ROI > 100%)")
print("=" * 80)
profitable = results_df[results_df['roi'] > 100]
if len(profitable) > 0:
    print(profitable[['bet_type', 'max_odds', 'min_confidence', 'top_n', 'num_bets', 'hit_rate', 'roi', 'profit']].to_string(index=False))
else:
    print("No strategies with ROI > 100%")

print("\n" + "=" * 80)
print("RECOMMENDED STRATEGY")
print("=" * 80)
# Find best ROI with at least 500 bets for stability
stable = results_df[(results_df['num_bets'] >= 500) & (results_df['bet_type'] == 'win')]
if len(stable) > 0:
    best = stable.iloc[0]
    print(f"Bet Type: Win (単勝)")
    print(f"Max Odds: {best['max_odds']}")
    print(f"Min Confidence: {best['min_confidence']}")
    print(f"Top N: {int(best['top_n'])}")
    print(f"Number of Bets: {int(best['num_bets'])}")
    print(f"Hit Rate: {best['hit_rate']:.2f}%")
    print(f"ROI: {best['roi']:.2f}%")
    print(f"Expected Profit: ¥{int(best['profit']):,}")

# Save full results
results_df.to_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/roi_grid_search.csv', index=False)
print("\nFull results saved to roi_grid_search.csv")

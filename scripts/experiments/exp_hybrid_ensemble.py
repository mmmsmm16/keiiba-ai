
"""
Hybrid Strategy Experiment
==========================
Combines two models:
1. Deep Value Model (Safe, Odds-Weighted LambdaRank) -> Predicts "Rank"
2. Regularized Gap Model (Aggressive, Log-Weighted, Clipped Gap) -> Predicts "Gap"

Strategies to Test:
A. Pure Deep Value (Baseline)
B. Pure Gap Reg (Aggressive)
C. Filtered Gap (Win Potential Filter):
   - Bet on Gap Model Rank 1 IF Deep Value Model Rank <= 5.
   - Logic: "It's an undervalued hole, AND it has decent winning chances."

Goal:
Maximize ROI while keeping Hit Rate acceptable (>5%).
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

# 1. Load Data
print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df_test = df[df['year'] == 2024].copy().reset_index(drop=True)

# 2. Load Models
print("Loading models...")
model_safe = joblib.load('models/experiments/exp_lambdarank_hard_weighted/model.pkl')
model_gap = joblib.load('models/experiments/exp_gap_prediction_reg/model.pkl')

# 3. Generate Predictions
print("Predicting...")
# Features extraction (Naive re-extraction logic, assuming same pipelines used in training scripts)
# We need to ensure we use the same features.
# Ideally we load features.csv from each dir.

def get_features(model_dir):
    try:
        return pd.read_csv(f'{model_dir}/features.csv')['0'].tolist()
    except:
        # Fallback (Manual reconstruction - risky but okay for quick test if code hasn't changed much)
        # Deep Value features: base + odds_rank_vs_elo + is_high/mid - yoso
        # Gap features: same
        exclude = ['rank', 'date', 'race_id', 'horse_id', 'jockey_id', 'trainer_id', 'target', 'year', 'odds', 'yoso_juni', 'yoso_p_std', 'yoso_p_mean']
        base = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        
        # Inject ad-hoc features needed
        df_test['odds_rank'] = df_test.groupby('race_id')['odds'].rank(ascending=True)
        if 'relative_horse_elo_z' in df_test.columns:
            df_test['elo_rank'] = df_test.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
            df_test['odds_rank_vs_elo'] = df_test['odds_rank'] - df_test['elo_rank']
        else:
            df_test['odds_rank_vs_elo'] = 0
            
        df_test['is_high_odds'] = (df_test['odds'] >= 10).astype(int)
        df_test['is_mid_odds'] = ((df_test['odds'] >= 5) & (df_test['odds'] < 10)).astype(int)
        
        added = ['odds_rank_vs_elo', 'is_high_odds', 'is_mid_odds']
        return [c for c in base if 'yoso' not in c] + added

# Ensure ad-hoc columns exist in df_test
df_test['odds_rank'] = df_test.groupby('race_id')['odds'].rank(ascending=True)
if 'relative_horse_elo_z' in df_test.columns:
    df_test['elo_rank'] = df_test.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
    df_test['odds_rank_vs_elo'] = df_test['odds_rank'] - df_test['elo_rank']
else:
    df_test['odds_rank_vs_elo'] = 0
df_test['is_high_odds'] = (df_test['odds'] >= 10).astype(int)
df_test['is_mid_odds'] = ((df_test['odds'] >= 5) & (df_test['odds'] < 10)).astype(int)

cols_safe = get_features('models/experiments/exp_lambdarank_hard_weighted')
cols_gap = get_features('models/experiments/exp_gap_prediction_reg')

# Intersect with df columns
cols_safe = [c for c in cols_safe if c in df_test.columns]
cols_gap = [c for c in cols_gap if c in df_test.columns]

pred_safe = model_safe.predict(df_test[cols_safe].values)
pred_gap = model_gap.predict(df_test[cols_gap].values)

df_test['score_safe'] = pred_safe
df_test['score_gap'] = pred_gap

# Ranks
df_test['rank_safe'] = df_test.groupby('race_id')['score_safe'].rank(ascending=False)
df_test['rank_gap'] = df_test.groupby('race_id')['score_gap'].rank(ascending=False)

# 4. Simulation
print("\n=== Simulation Results (2024 Test) ===")

def eval_strategy(name, condition_mask):
    bets = df_test[condition_mask]
    if len(bets) == 0:
        print(f"[{name}] No bets.")
        return
        
    wins = bets[bets['rank'] == 1]
    roi = wins['odds'].sum() / len(bets) * 100
    hit = len(wins) / len(bets) * 100
    mean_odds = bets['odds'].mean()
    
    print(f"[{name}] Bets: {len(bets)} | ROI: {roi:.2f}% | Hit: {hit:.1f}% | MeanOdds: {mean_odds:.1f}")

# Strategy A: Baseline Safe (Top 1)
eval_strategy("A. Deep Value Top 1", df_test['rank_safe'] == 1)

# Strategy B: Gap Aggressive (Top 1)
eval_strategy("B. Gap Reg Top 1", df_test['rank_gap'] == 1)

# Strategy C: Hybrid Filter (Gap Top 1 IF Safe Rank <= 5)
mask_c = (df_test['rank_gap'] == 1) & (df_test['rank_safe'] <= 5)
eval_strategy("C. Hybrid Filter (Gap=1 & Safe<=5)", mask_c)

# Strategy D: Intersection (Both Top 1) - Strict
mask_d = (df_test['rank_gap'] == 1) & (df_test['rank_safe'] == 1)
eval_strategy("D. Strict Intersection (Both=1)", mask_d)

# Strategy E: Portfolio (Bet A Top 1 AND B Top 1)
# Note: If same horse, bet 2 units? Or 1 unit?
# Simulation assumes 1 unit per strategy trigger. If double trigger, 2 units.
# Just sum usage.
# Simple way: Combine datasets
bets_a = df_test[df_test['rank_safe'] == 1].copy()
bets_b = df_test[df_test['rank_gap'] == 1].copy()
portfolio = pd.concat([bets_a, bets_b])
wins_p = portfolio[portfolio['rank'] == 1]
roi_p = wins_p['odds'].sum() / len(portfolio) * 100
hit_p = len(wins_p) / len(portfolio) * 100
print(f"[E. Portfolio (A+B)] Bets: {len(portfolio)} | ROI: {roi_p:.2f}% | Hit: {hit_p:.1f}%")

print("\nDone.")

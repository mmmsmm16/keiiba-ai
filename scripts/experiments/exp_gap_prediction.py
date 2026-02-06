
"""
Gap Prediction Model Experiment
===============================
Concept:
Instead of predicting raw rank (1st, 2nd, 3rd...), predict the "Gap" between
Popularity Rank and Actual Rank.

Target:
- Target Label = max(0, PopularityRank - Rank + Constant)
- Example:
  - Pop 10 -> Rank 1: Gap +9 -> Label 9 (Huge Undervalued)
  - Pop 1 -> Rank 1: Gap 0 -> Label 0 (Expected)
  - Pop 1 -> Rank 10: Gap -9 -> Label 0 (Overvalued)

Goal:
Find horses that outperform market expectations.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.metrics import ndcg_score

# Load data
print("Loading data...")
# Use v12 features (no yoso_juni in features)
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['odds'] > 0) & (df['odds'].notna())].reset_index(drop=True)

# --- Feature Engineering ---
# Add Undervalued Features (Only those NOT derived from yoso_juni)
print("Injecting features...")
df['odds_rank'] = df.groupby('race_id')['odds'].rank(ascending=True)

if 'relative_horse_elo_z' in df.columns:
    df['elo_rank'] = df.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
    df['odds_rank_vs_elo'] = df['odds_rank'] - df['elo_rank']
else:
    df['odds_rank_vs_elo'] = 0

df['is_high_odds'] = (df['odds'] >= 10).astype(int)
df['is_mid_odds'] = ((df['odds'] >= 5) & (df['odds'] < 10)).astype(int)

# Load baseline features from a known list or define here
# We reuse v12 features list but exclude yoso
try:
    baseline_features = pd.read_csv('models/experiments/exp_lambdarank_hard_weighted/features.csv')['0'].tolist()
except:
    # If not found, load from another experiment or define manually
    # Fallback to loading all numeric columns except targets
    print("Warning: feature list not found. Using auto-detection.")
    exclude = ['rank', 'date', 'race_id', 'horse_id', 'jockey_id', 'trainer_id', 'target', 'year', 'odds', 'yoso_juni', 'yoso_p_std', 'yoso_p_mean']
    baseline_features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

# Ensure yoso is removed
all_features = [f for f in baseline_features if 'yoso' not in f] + ['odds_rank_vs_elo', 'is_high_odds', 'is_mid_odds']
all_features = list(set([c for c in all_features if c in df.columns]))

print(f"Features: {len(all_features)}")

# --- Target Generation (The Core Change) ---
# Popularity Rank (from Odds)
# 1st fav = 1, 10th fav = 10
df['pop_rank'] = df.groupby('race_id')['odds'].rank(ascending=True)

# Actual Rank
# 1st = 1, 10th = 10

# Gap = Pop - Rank
# Positive Gap -> Undervalued (e.g. Pop 10 - Rank 1 = +9)
# Negative Gap -> Overvalued (e.g. Pop 1 - Rank 10 = -9)
df['gap'] = df['pop_rank'] - df['rank']

# Label for LambdaRank (Must be non-negative integer ideally, or float relevance)
# LightGBM LambdaRank requires int labels by default.
df['target_label'] = df['gap'].apply(lambda x: int(max(0, x)))

# --- Split ---
df_train = df[df['year'] <= 2022].copy()
df_valid = df[df['year'] == 2023].copy()
df_test = df[df['year'] == 2024].copy()

# Sort for LambdaRank group (Must be sorted by group_id)
df_train = df_train.sort_values('race_id')
df_valid = df_valid.sort_values('race_id')
df_test = df_test.sort_values('race_id')

train_groups = df_train.groupby('race_id').size().values
valid_groups = df_valid.groupby('race_id').size().values

X_train = df_train[all_features].values
X_valid = df_valid[all_features].values
X_test = df_test[all_features].values

y_train = df_train['target_label'].values
y_valid = df_valid['target_label'].values

# Weights: Use Odds? Or Uniform?
# Since label already encodes "Undervalued" (which implies High Odds usually),
# further weighting by odds might be double counting?
# Gap +9 implies Odds was high (Pop 10).
# So the Label size correlates with Odds.
# Let's try Uniform Weight first to let the Label drive the learning.
# Or Weight by Odds to really emphasize "High Payout Holes"?
# Let's start with Odds Weighting to match "Deep Value" philosophy.
w_train = df_train['odds'].clip(1.0, 50.0).values
# w_train = np.ones(len(y_train)) # Alternative: Uniform

train_data = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=w_train, feature_name=all_features)
valid_data = lgb.Dataset(X_valid, label=y_valid, group=valid_groups, reference=train_data, feature_name=all_features)

# --- Training ---
print("\nTraining Gap Prediction Model...")
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.01,
    'n_estimators': 2000,
    'seed': 42,
    'verbosity': -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=params['n_estimators'],
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# --- Evaluation ---
print("\n=== Evaluation (Gap Prediction) ===")
scores = model.predict(X_test)
df_test['pred_score'] = scores
df_test['pred_rank'] = df_test.groupby('race_id')['pred_score'].rank(ascending=False)

# What are we predicting? We are predicting "Gap".
# So Top Ranked horses should be "Undervalued".

# 1. Check ROI of Top Ranked horses
# If the model works, Top Predicted horses should be "Holes that actually ran well".
top1 = df_test[df_test['pred_rank'] == 1]
wins = top1[top1['rank'] == 1]
places = top1[top1['rank'] <= 3]

print("\nTop 1 Predicted Stats:")
print(top1[['odds', 'rank', 'gap', 'pop_rank']].describe())

roi_win = wins['odds'].sum() / len(top1) * 100
roi_place = 0 # Need place payout calculation, approximation:
# Approx Place ROI: sum(odds_place) / bets
# We don't have place odds here easily effectively without loading detailed data.
# Just use Win ROI and Hit Rate for now.
hit_rate_win = len(wins) / len(top1) * 100
hit_rate_place = len(places) / len(top1) * 100

print(f"Top 1 Win ROI: {roi_win:.2f}% (Hit Rate: {hit_rate_win:.1f}%)")
print(f"Top 1 Place Hit Rate: {hit_rate_place:.1f}%")

# 2. Check "Hole Finding" Capability
# Filter for Popularity > 5 (Holes)
holes = top1[top1['pop_rank'] > 5]
print(f"\nTop 1 Predictions that were unpopular (Pop > 5): {len(holes)} bets")
if len(holes) > 0:
    hole_wins = holes[holes['rank'] == 1]
    hole_roi = hole_wins['odds'].sum() / len(holes) * 100
    print(f"Hole Win ROI: {hole_roi:.2f}% (Hit Rate: {len(hole_wins)/len(holes)*100:.1f}%)")

# Save
OUTPUT_DIR = 'models/experiments/exp_gap_prediction'
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(model, f'{OUTPUT_DIR}/model.pkl')
pd.DataFrame({'0': all_features}).to_csv(f'{OUTPUT_DIR}/features.csv', index=False)
print(f"Model saved to {OUTPUT_DIR}")

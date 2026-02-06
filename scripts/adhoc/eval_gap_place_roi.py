"""
Gap Model Place (複勝) ROI Evaluation
=====================================
Evaluate the improved Gap Model's Place (Top 3) ROI.
"""
import pandas as pd
import numpy as np
import joblib
import psycopg2

# 1. Load Model and Features
print("Loading model...")
model = joblib.load('models/experiments/exp_gap_prediction_reg/model.pkl')
features = pd.read_csv('models/experiments/exp_gap_prediction_reg/features.csv')['0'].tolist()

# 2. Load Test Data
print("Loading test data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['odds'] > 0) & (df['odds'].notna())].reset_index(drop=True)

# Inject features
df['odds_rank'] = df.groupby('race_id')['odds'].rank(ascending=True)
if 'relative_horse_elo_z' in df.columns:
    df['elo_rank'] = df.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
    df['odds_rank_vs_elo'] = df['odds_rank'] - df['elo_rank']
else:
    df['odds_rank_vs_elo'] = 0
df['is_high_odds'] = (df['odds'] >= 10).astype(int)
df['is_mid_odds'] = ((df['odds'] >= 5) & (df['odds'] < 10)).astype(int)

df_test = df[df['year'] == 2024].copy()

# 3. Predict
X_test = df_test[[f for f in features if f in df_test.columns]].fillna(0)
scores = model.predict(X_test)
df_test['pred_score'] = scores
df_test['pred_rank'] = df_test.groupby('race_id')['pred_score'].rank(ascending=False)

# 4. Load Actual Place Odds from DB
print("Loading place odds from DB...")
try:
    conn = psycopg2.connect(
        host='host.docker.internal',
        port=5433,
        dbname='pckeiba',
        user='postgres',
        password='postgres'
    )
    
    # Get race_ids
    race_ids = tuple(df_test['race_id'].unique())
    
    query = f"""
        SELECT race_id, horse_number, odds_place_min, odds_place_max
        FROM jvd_o1
        WHERE race_id IN {race_ids}
    """
    df_place = pd.read_sql(query, conn)
    conn.close()
    
    # Use average of min/max for place odds
    df_place['odds_place'] = (df_place['odds_place_min'].astype(float) + df_place['odds_place_max'].astype(float)) / 2
    
    # Merge
    df_test = df_test.merge(df_place[['race_id', 'horse_number', 'odds_place']], 
                            on=['race_id', 'horse_number'], how='left')
    
    has_place_odds = True
    print(f"Loaded {len(df_place)} place odds records.")
except Exception as e:
    print(f"Could not load place odds: {e}")
    print("Using approximation: Place Odds ≈ Win Odds / 3")
    df_test['odds_place'] = df_test['odds'] / 3
    has_place_odds = False

# 5. Evaluate
print("\n=== Gap Model Place (複勝) ROI Evaluation ===\n")

# Top 1 Prediction Stats
top1 = df_test[df_test['pred_rank'] == 1]

# Place = Rank <= 3
places = top1[top1['rank'] <= 3]

# Win ROI
wins = top1[top1['rank'] == 1]
win_roi = wins['odds'].sum() / len(top1) * 100

# Place ROI
if 'odds_place' in top1.columns and not top1['odds_place'].isna().all():
    place_payout = places['odds_place'].sum()
    place_roi = place_payout / len(top1) * 100
else:
    # Approximation
    place_payout = places['odds'].sum() / 3
    place_roi = place_payout / len(top1) * 100

print(f"Total Bets (Top 1 Picks): {len(top1)}")
print(f"Wins: {len(wins)} ({len(wins)/len(top1)*100:.1f}%)")
print(f"Places (Top 3): {len(places)} ({len(places)/len(top1)*100:.1f}%)")
print(f"Win ROI: {win_roi:.2f}%")
print(f"Place ROI: {place_roi:.2f}%{'*' if not has_place_odds else ''}")
if not has_place_odds:
    print("  * Approximated (actual place odds not available)")

# Stats by Odds Range
print("\n=== Place ROI by Odds Range ===")
for odds_min, odds_max, label in [(1, 10, '1-10倍'), (10, 20, '10-20倍'), (20, 50, '20-50倍'), (50, 1000, '50倍以上')]:
    subset = top1[(top1['odds'] >= odds_min) & (top1['odds'] < odds_max)]
    if len(subset) == 0:
        continue
    sub_places = subset[subset['rank'] <= 3]
    if 'odds_place' in subset.columns and not subset['odds_place'].isna().all():
        sub_roi = sub_places['odds_place'].sum() / len(subset) * 100
    else:
        sub_roi = sub_places['odds'].sum() / 3 / len(subset) * 100
    print(f"{label}: {len(subset)} bets, Place Hit {len(sub_places)/len(subset)*100:.1f}%, ROI {sub_roi:.1f}%")

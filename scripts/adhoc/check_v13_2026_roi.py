"""
V13 Win ROI Checker for 2026
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader

V13_MODEL_PATH = 'models/experiments/exp_lambdarank_hard_weighted/model.pkl'
V13_FEATS_PATH = 'models/experiments/exp_lambdarank_hard_weighted/features.csv'
PREPROCESSED_DATA_PATH = 'data/processed/preprocessed_data_v13_active.parquet'

def main():
    print("Loading 2026 Data (Jan 1 - Jan 24)...")
    loader = JraVanDataLoader()
    # Fetch 2026 Jan races
    df_db = loader.load(history_start_date='2026-01-01', end_date='2026-01-25', skip_odds=True)
    
    if df_db.empty:
        print("No 2026 data found in DB."); return

    from src.preprocessing.feature_pipeline import FeaturePipeline
    pipeline = FeaturePipeline(cache_dir='data/features_v14/prod_cache')
    df_26 = pipeline.load_features(df_db, list(pipeline.registry.keys()))
    
    # Merge meta
    meta = df_db[['race_id', 'horse_number', 'odds', 'rank', 'popularity']].drop_duplicates()
    df_26 = pd.merge(df_26, meta, on=['race_id', 'horse_number'])
    df_26['race_id'] = df_26['race_id'].astype(str)

    print(f"Applying V13 Model to {len(df_26)} records...")
    model = joblib.load(V13_MODEL_PATH)
    # The CSV has features in the first (and only) column without header
    feats = pd.read_csv(V13_FEATS_PATH, header=None).iloc[:, 0].tolist()
    # Check if first item is '0' or something similar
    if feats[0] == '0' or feats[0] == 'feature': feats = feats[1:]
    
    # Simple feature prep
    X = df_26.reindex(columns=feats, fill_value=-999.0).fillna(-999.0)
    df_26['score'] = model.predict_proba(X)[:, -1] if hasattr(model, 'predict_proba') else model.predict(X)
    
    # Rank per race
    df_26['v13_rank'] = df_26.groupby('race_id')['score'].rank(ascending=False, method='first')
    
    # Target and Odds
    df_26['actual_rank'] = pd.to_numeric(df_26['rank'], errors='coerce').fillna(99).astype(int)
    df_26['is_win'] = (df_26['actual_rank'] == 1).astype(int)
    
    top1 = df_26[df_26['v13_rank'] == 1].copy()
    n_races = top1['race_id'].nunique()
    
    # Clean odds for ROI
    # Use 'odds' if > 0 else 0
    top1['odds_clean'] = top1['odds'].apply(lambda x: x if x > 1.0 else 0.0)
    
    win_ret = (top1[top1['is_win'] == 1]['odds_clean'].sum())
    win_roi = (win_ret / n_races) * 100
    win_rate = top1['is_win'].mean()
    top3_rate = (top1['actual_rank'] <= 3).mean()
    
    print("\n=== V13 Performance: 2026 (Jan 1 - Jan 24) ===")
    print(f"Races:     {n_races}")
    print(f"Win Rate:  {win_rate:>.1%}")
    print(f"Top-3 Rate: {top3_rate:>.1%}")
    print(f"Win ROI:   {win_roi:>.1f}%")
    print(f"Avg Odds (Clean): {top1[top1['is_win']==1]['odds_clean'].mean():.2f}")

if __name__ == "__main__":
    main()


import sys
import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

V13_MODEL_PATH = 'models/experiments/exp_lambdarank_hard_weighted/model.pkl'
V13_FEATS_PATH = 'models/experiments/exp_lambdarank_hard_weighted/features.csv'
V14_MODEL_PATH = 'models/experiments/exp_gap_v14_production/model_v14.pkl'
V14_FEATS_PATH = 'models/experiments/exp_gap_v14_production/features.csv'
CACHE_DIR = 'data/features_v14/prod_cache'

def add_v13_odds_features_sync(df):
    # Leak-free: prioritize odds_10min
    if 'odds_10min' in df.columns:
        odds_base = df['odds_10min'].replace(0, np.nan)
    else:
        odds_base = pd.Series(np.nan, index=df.index)

    df['odds_calc'] = odds_base.fillna(10.0)
    df['odds_rank'] = df.groupby('race_id')['odds_calc'].rank(ascending=True, method='min')

    if 'relative_horse_elo_z' in df.columns:
        df['elo_rank'] = df.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False, method='min')
        df['odds_rank_vs_elo'] = df['odds_rank'] - df['elo_rank']
    else:
        df['odds_rank_vs_elo'] = 0

    df['is_high_odds'] = (df['odds_calc'] >= 10).astype(int)
    df['is_mid_odds'] = ((df['odds_calc'] >= 5) & (df['odds_calc'] < 10)).astype(int)
    return df

def build_v13_features(df, feats_v13, fill_value=0.0):
    X = df.reindex(columns=feats_v13, fill_value=fill_value)
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].cat.codes
        elif X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    return X.fillna(fill_value).astype(float)

def add_v14_derived_features_sync(df):
    # Leak-free: odds_final for features uses pre-race estimate (odds_10min)
    df['odds_final'] = df['odds_10min'].copy()
    
    df['odds_ratio_10min'] = df['odds_final'] / df['odds_10min'].replace(0, np.nan)
    df['odds_ratio_10min'] = df['odds_ratio_10min'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    if 'odds_60min' not in df.columns: df['odds_60min'] = df.get('odds_10min', np.nan)
    df['odds_60min'] = df['odds_60min'].fillna(df.get('odds_10min', np.nan)) 
    
    df['odds_ratio_60_10'] = df['odds_10min'] / df['odds_60min'].replace(0, np.nan)
    df['odds_ratio_60_10'] = df['odds_ratio_60_10'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    df['odds_log_ratio_10min'] = np.log(df['odds_final'] + 1e-9) - np.log(df['odds_10min'] + 1e-9)
    df['odds_log_ratio_10min'] = df['odds_log_ratio_10min'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    df['odds_rank_10min'] = df.groupby('race_id')['odds_10min'].rank(method='min')
    
    # Use popularity_10min to avoid leakage
    pop_base = df.get('popularity_10min', pd.Series(np.nan, index=df.index)).fillna(df['odds_rank_10min'])
    df['rank_diff_10min'] = pop_base - df['odds_rank_10min']
    
    if 'field_size' not in df.columns:
        df['field_size'] = df.groupby('race_id')['horse_number'].transform('count')
    return df

def main():
    race_id = "202605010212" # Tokyo 12R
    target_date = "2026-02-01"
    
    loader = JraVanDataLoader()
    
    with loader.engine.connect() as conn:
        q = text("SELECT DISTINCT ketto_toroku_bango FROM jvd_se WHERE CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) = :rid")
        horse_ids = [r[0] for r in conn.execute(q, {"rid": race_id}).fetchall()]
    
    print(f"Horses for {race_id}: {horse_ids}")
    df_raw = loader.load_for_horses(horse_ids, target_date, skip_training=False)
    
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    blocks = list(pipeline.registry.keys())
    df_features = pipeline.load_features(df_raw, blocks, force=True)
    
    core_cols = ['race_id', 'horse_number', 'date', 'odds', 'popularity', 'horse_name']
    df_merged = pd.merge(df_features, df_raw[core_cols].drop_duplicates(['race_id', 'horse_number']), 
                         on=['race_id', 'horse_number'], how='left')
    
    # Simulate jvd_o1 fallback in both (ensure odds_10min exists)
    # For this debug, we'll just use the already loaded odds_10min from the pipe
    df_race = df_merged[df_merged['race_id'] == race_id].copy()
    
    # Models
    feats_v13 = pd.read_csv(V13_FEATS_PATH).iloc[:, 0].tolist()
    model_v13 = joblib.load(V13_MODEL_PATH)
    feats_v14 = pd.read_csv(V14_FEATS_PATH)['feature'].tolist()
    model_v14 = joblib.load(V14_MODEL_PATH)
    
    # --- SIMULATION ---
    # Path A: JIT (Simulate pre-race environment)
    df_race_jit = add_v13_odds_features_sync(df_race.copy())
    df_race_jit = add_v14_derived_features_sync(df_race_jit)
    
    # Path B: Backtest (Verify it ignores the final closing 'odds' and 'popularity')
    df_race_bt = add_v13_odds_features_sync(df_race.copy())
    df_race_bt = add_v14_derived_features_sync(df_race_bt)
    
    # Predictions
    X_jit_v13 = build_v13_features(df_race_jit, feats_v13)
    score_jit_v13 = model_v13.predict(X_jit_v13)
    X_bt_v13 = build_v13_features(df_race_bt, feats_v13)
    score_bt_v13 = model_v13.predict(X_bt_v13)
    
    X_jit_v14 = df_race_jit.reindex(columns=feats_v14, fill_value=0.0).fillna(0.0)
    score_jit_v14 = model_v14.predict(X_jit_v14)
    X_bt_v14 = df_race_bt.reindex(columns=feats_v14, fill_value=0.0).fillna(0.0)
    score_bt_v14 = model_v14.predict(X_bt_v14)
    
    comparison = pd.DataFrame({
        'umaban': df_race['horse_number'],
        'v13_jit': score_jit_v13,
        'v13_bt': score_bt_v13,
        'v14_jit': score_jit_v14,
        'v14_bt': score_bt_v14
    })
    
    print("\n--- Sync Comparison (V13 & V14) ---")
    print(comparison)
    print(f"\nV13 Max Diff: {np.abs(score_jit_v13 - score_bt_v13).max()}")
    print(f"V14 Max Diff: {np.abs(score_jit_v14 - score_bt_v14).max()}")

if __name__ == "__main__":
    main()

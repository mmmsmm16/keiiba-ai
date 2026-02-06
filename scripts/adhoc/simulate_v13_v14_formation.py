
import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime

# Add workspace
sys.path.append('/workspace')

def get_visual_width(s):
    import unicodedata
    width = 0
    for c in str(s):
        if unicodedata.east_asian_width(c) in 'WF':
            width += 2
        else:
            width += 1
    return width

def main():
    V13_PATH = 'data/predictions/v13_oof_2024_clean.parquet'
    V14_MODEL_PATH = 'models/experiments/exp_gap_v14_production/model_v14.pkl'
    V14_FEATURES_CSV = 'models/experiments/exp_gap_v14_production/features.csv'
    DATA_PATH = "data/processed/preprocessed_data_v13_active.parquet"
    
    if not all([os.path.exists(p) for p in [V13_PATH, V14_MODEL_PATH, DATA_PATH]]):
        print("Missing data or model files.")
        return

    V13_2025_PATH = 'data/predictions/v13_oof_2025_clean.parquet'
    
    # 1. Load V13 Predictions
    print(f"Loading V13 OOF from {V13_PATH} and {V13_2025_PATH}...")
    df_v13_24 = pd.read_parquet(V13_PATH)
    df_v13_25 = pd.read_parquet(V13_2025_PATH)
    df_v13 = pd.concat([df_v13_24, df_v13_25], ignore_index=True)
    
    df_v13['race_id'] = df_v13['race_id'].astype(str)
    
    # 2. Load V14 Data and Predict
    print(f"Loading features from {DATA_PATH}...")
    df_all = pd.read_parquet(DATA_PATH)
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    # Ensure we cover both years
    df_year = df_all[(df_all['date'].dt.year >= 2024) & (df_all['date'].dt.year <= 2025)].copy()
    print(f"Feature Data (2024-2025): {len(df_year)} rows.")

    print("Loading V14 model...")
    model_v14 = joblib.load(V14_MODEL_PATH)
    v14_features = pd.read_csv(V14_FEATURES_CSV)['feature'].tolist()
    
    print("Performing feature engineering for V14 (Odds Alias Fix)...")
    # Feature Fix: V14 Model expects 'odds_final' etc.
    if 'odds_10min' not in df_year.columns and 'odds' in df_year.columns:
        df_year['odds_10min'] = df_year['odds']
        
    df_year['odds_rank_10min'] = df_year.groupby('race_id')['odds_10min'].rank(method='min')
    df_year['rank_diff_10min'] = df_year['popularity'] - df_year['odds_rank_10min']
    df_year['odds_log_ratio_10min'] = np.log(df_year['odds'] + 1e-9) - np.log(df_year['odds_10min'] + 1e-9)
    df_year['odds_ratio_60_10'] = 1.0 
    df_year['odds_60min'] = df_year['odds_10min']
    
    if 'odds_final' not in df_year.columns and 'odds' in df_year.columns:
        df_year['odds_final'] = df_year['odds']
    
    print("Running V14 Prediction...")
    X_v14 = df_year[v14_features].fillna(0)
    preds_v14 = model_v14.predict(X_v14)
    df_year['pred_gap'] = preds_v14
    
    # Standardize and Merge
    df_v13 = df_v13[df_v13['race_id'].str.match(r'^\d+$')].copy()
    df_year = df_year[df_year['race_id'].str.match(r'^\d+$')].copy()
    
    # Merge (Inner join ensures both models have data)
    # Check if 'date' is in v13 cols usually.
    v13_cols = ['race_id', 'horse_number', 'pred_prob', 'rank', 'odds']
    if 'date' in df_v13.columns: v13_cols.append('date')
    
    df_combined = pd.merge(
        df_v13[v13_cols],
        df_year[['race_id', 'horse_number', 'pred_gap']], # Don't duplicate date if v13 has it
        on=['race_id', 'horse_number'],
        how='inner'
    )
    
    # If date missing (e.g. v13 didn't have it), use df_year's date? 
    # But df_year was filtered to 2024-2025. 
    # Actually safer to merge date from df_year if needed.
    if 'date' not in df_combined.columns:
        date_map = df_year[['race_id', 'date']].drop_duplicates()
        df_combined = pd.merge(df_combined, date_map, on='race_id', how='left')

    print(f"Combined data: {len(df_combined)} rows. Races: {df_combined['race_id'].nunique()}")
    
    # Filter Invalid Races (Odds=0)
    # Identify race_ids where ANY horse has odds=0
    invalid_rids = df_combined[df_combined['odds'] == 0]['race_id'].unique()
    print(f"Found {len(invalid_rids)} races with 0.0 odds. Excluding...")
    df_combined = df_combined[~df_combined['race_id'].isin(invalid_rids)].copy()
    print(f"Valid data: {len(df_combined)} rows. Races: {df_combined['race_id'].nunique()}")

    # Ranks
    df_combined['v13_rank'] = df_combined.groupby('race_id')['pred_prob'].rank(ascending=False, method='first')
    df_combined['v14_rank'] = df_combined.groupby('race_id')['pred_gap'].rank(ascending=False, method='first')

    # Load Payouts (2024, 2025)
    from src.preprocessing.loader import JraVanDataLoader
    loader = JraVanDataLoader()
    q_hr = """
    SELECT 
        kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
        haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
        haraimodoshi_wide_1a, haraimodoshi_wide_1b,
        haraimodoshi_wide_2a, haraimodoshi_wide_2b,
        haraimodoshi_wide_3a, haraimodoshi_wide_3b,
        haraimodoshi_wide_4a, haraimodoshi_wide_4b,
        haraimodoshi_wide_5a, haraimodoshi_wide_5b
    FROM jvd_hr
    WHERE kaisai_nen IN ('2024', '2025')
    """
    df_hr = pd.read_sql(q_hr, loader.engine)
    
    hr_map = {}
    for _, row in df_hr.iterrows():
        try:
            rid = f"{int(float(row['kaisai_nen']))}{int(float(row['keibajo_code'])):02}{int(float(row['kaisai_kai'])):02}{int(float(row['kaisai_nichime'])):02}{int(float(row['race_bango'])):02}"
        except: continue
        
        umaren = {} # Simplified for speed (chk all slots if needed, using slots 1 is standard for most, but multi-hit exists)
        # Using 1-3 slots just in case
        for i in range(1, 4):
            k_a = f'haraimodoshi_umaren_{i}a'
            k_b = f'haraimodoshi_umaren_{i}b'
            if k_a in row and pd.notna(row[k_a]) and pd.notna(row[k_b]):
                 try:
                    pay = int(float(str(row[k_b]).replace(',', '')))
                    if pay > 0:
                        u1, u2 = int(row[k_a][:2]), int(row[k_a][2:4])
                        umaren[tuple(sorted((u1, u2)))] = pay
                 except: pass
                 
        wide = {}
        for i in range(1, 6):
            k_a = f'haraimodoshi_wide_{i}a'
            k_b = f'haraimodoshi_wide_{i}b'
            if k_a in row and pd.notna(row[k_a]) and pd.notna(row[k_b]):
                 try:
                    pay = int(float(str(row[k_b]).replace(',', '')))
                    if pay > 0:
                        u1, u2 = int(row[k_a][:2]), int(row[k_a][2:4])
                        wide[tuple(sorted((u1, u2)))] = pay
                 except: pass
        hr_map[rid] = {'umaren': umaren, 'wide': wide}

    # Simulation by Year
    if 'date' in df_combined.columns:
        df_combined['date'] = pd.to_datetime(df_combined['date']) # Ensure datetime
        print(f"Year Distribution:\n{df_combined['date'].dt.year.value_counts()}")
        years = sorted(df_combined['date'].dt.year.unique())
    else:
        print("Date column missing for year splitting. Creating from RaceID?")
        # Fallback: RaceID first 4 chars is usually year
        df_combined['year'] = df_combined['race_id'].astype(str).str[:4].astype(int)
        years = sorted(df_combined['year'].unique())
        print(f"Year Distribution (from RaceID):\n{df_combined['year'].value_counts()}")

    scenarios = [(1, 1), (1, 3), (1, 5), (2, 5)] # Focused scenarios
    
    for y in years:
        print(f"\n=== Simulation Year: {y} ===")
        print(f"{'V13':<4} {'V14':<4} {'UmaROI':<8} {'UmaHR':<8} {'WideROI':<8} {'WideHR':<8} {'Bets/R':<6}")
        print("-" * 60)
        
        if 'date' in df_combined.columns:
            df_y = df_combined[df_combined['date'].dt.year == y]
        else:
            df_y = df_combined[df_combined['year'] == y]
            
        rids_y = df_y['race_id'].unique()
        
        for n1, n2 in scenarios:
            bets_uma = 0; ret_uma = 0; hit_uma = 0
            bets_wide = 0; ret_wide = 0; hit_wide = 0
            
            for rid in rids_y:
                if rid not in hr_map: continue
                race_df = df_y[df_y['race_id'] == rid]
                payouts = hr_map[rid]
                
                v13_top = race_df[race_df['v13_rank'] <= n1]['horse_number'].tolist()
                v14_top = race_df[race_df['v14_rank'] <= n2]['horse_number'].tolist()
                
                pairs = []
                for a in v13_top:
                    for b in v14_top:
                        if a != b:
                            pair = tuple(sorted((int(a), int(b))))
                            if pair not in pairs:
                                pairs.append(pair)
                
                if not pairs: continue
                
                # Bet
                bets_uma += len(pairs)
                bets_wide += len(pairs)
                
                # Check Hit
                h_u = False
                h_w = False
                for p in pairs:
                    if p in payouts['umaren']:
                        ret_uma += payouts['umaren'][p]
                        h_u = True
                    if p in payouts['wide']:
                        ret_wide += payouts['wide'][p]
                        h_w = True # At least one wide hit
                
                if h_u: hit_uma += 1
                if h_w: hit_wide += 1
            
            roi_u = ret_uma / (bets_uma * 100) if bets_uma else 0
            roi_w = ret_wide / (bets_wide * 100) if bets_wide else 0
            hr_u = hit_uma / len(rids_y) if len(rids_y) else 0
            hr_w = hit_wide / len(rids_y) if len(rids_y) else 0
            
            print(f"{n1:<4} {n2:<4} {roi_u:<8.1%} {hr_u:<8.1%} {roi_w:<8.1%} {hr_w:<8.1%} {bets_uma/len(rids_y):<6.1f}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

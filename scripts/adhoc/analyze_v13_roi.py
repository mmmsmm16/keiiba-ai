
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader

def main():
    # V13 Model Path
    MODEL_PATH = 'models/experiments/exp_gap_prediction_reg/model.pkl'
    FEATURES_PATH = 'models/experiments/exp_gap_prediction_reg/features.csv'
    DATA_PATH = "data/processed/preprocessed_data_v13_active.parquet"
    
    if not os.path.exists(DATA_PATH):
        print("Data not found.")
        return

    print("Loading V13 Model Analysis...")
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Needs odds_10min fallback logic? 
    # Yes, for strategy filtering.
    need_odds_compute = False
    if 'odds_10min' not in df.columns: need_odds_compute = True
    elif df['odds_10min'].isna().all(): need_odds_compute = True
        
    if need_odds_compute:
        print("Computing odds_fluctuation directly from DB (Fallback)...")
        from src.preprocessing.features.odds_fluctuation import compute_odds_fluctuation
        from src.preprocessing.loader import JraVanDataLoader
        
        if 'start_time_str' not in df.columns:
             print("Fetching start_time_str via Loader...")
             loader = JraVanDataLoader()
             dates = df['date'].astype(str).unique()
             min_date = dates.min()
             max_date = dates.max()
             df_info = loader.load(history_start_date=min_date, end_date=max_date, skip_odds=True, skip_training=True)
             df_info['race_id'] = df_info['race_id'].astype(str)
             df['race_id'] = df['race_id'].astype(str)
             if 'start_time_str' in df_info.columns:
                 df = pd.merge(df, df_info[['race_id', 'start_time_str']].drop_duplicates(), on='race_id', how='left')

        df_odds = compute_odds_fluctuation(df)
        if not df_odds.empty:
            df['race_id'] = df['race_id'].astype(str)
            df['horse_number'] = df['horse_number'].astype(int)
            df_odds['race_id'] = df_odds['race_id'].astype(str)
            df_odds['horse_number'] = df_odds['horse_number'].astype(int)
            
            for c in ['odds_10min', 'odds_final', 'odds_60min', 'odds_ratio_10min', 'rank_diff_10min', 'odds_log_ratio_10min', 'odds_ratio_60_10', 'ninki_10min']:
                if c in df.columns: df = df.drop(columns=[c])
            
            df = pd.merge(df, df_odds.drop(columns=['horse_id'], errors='ignore'), on=['race_id', 'horse_number'], how='left')

    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    # Filter 2025
    test_df = df[df['date'].dt.year == 2025].copy()
    if len(test_df) == 0:
        print("WARN: No 2025 data found. Using 2024 data.")
        test_df = df[df['date'].dt.year == 2024].copy()
        
    # Feature Engineering (V13 Specific)
    # V13 used 'odds_rank' (from 'odds' aka Final Odds)
    # And 'odds_rank_vs_elo'
    if 'odds' in test_df.columns:
        test_df['odds_rank'] = test_df.groupby('race_id')['odds'].rank(ascending=True)
    else:
        test_df['odds_rank'] = 0 # Should not happen if data has result
        
    if 'relative_horse_elo_z' in test_df.columns:
        test_df['elo_rank'] = test_df.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
        test_df['odds_rank_vs_elo'] = test_df['odds_rank'] - test_df['elo_rank']
    else:
        test_df['odds_rank_vs_elo'] = 0
        
    test_df['is_high_odds'] = (test_df['odds'] >= 10).astype(int)
    test_df['is_mid_odds'] = ((test_df['odds'] >= 5) & (test_df['odds'] < 10)).astype(int)
    
    # Load Model
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    
    # Features
    if os.path.exists(FEATURES_PATH):
        features = pd.read_csv(FEATURES_PATH, header=None).iloc[:, 0].tolist()
        # Header might be missing or index? Viewed file shows: row 1 is '0', row 2 'horse_number'...
        # Actually in view_file: line 1: 0. line 2: horse_number.
        # This implies it has a header '0' or index?
        # My view_file output had line numbers added by tool.
        # Original content:
        # 0
        # horse_number
        # ...
        # This looks like `pd.DataFrame(features).to_csv(..., index=True)` resulted in index column being unnamed?
        # Or column name is "0".
        # Let's try read with header=0. if column name is '0'.
        pass
    else:
        print("Features not found.")
        return
        
    # Fix Features List parsing
    # Re-reading properly
    fty = pd.read_csv(FEATURES_PATH) # Assuming header exists
    if '0' in fty.columns:
        features = fty['0'].tolist()
    elif len(fty.columns) == 1:
        features = fty.iloc[:, 0].tolist()
    else:
        # fallback
        features = fty.columns.tolist() 
        
    # Ensure columns exist
    X_test = test_df.copy()
    for c in features:
        if c not in X_test.columns:
            X_test[c] = np.nan
    X_test = X_test[features]
    
    # Predict
    preds = model.predict(X_test)
    test_df['gap_score'] = preds
    test_df['gap_rank'] = test_df.groupby('race_id')['gap_score'].rank(ascending=False)
    
    # --- ROI Analysis ---
    print("\n=== ROI Analysis (V13 Legacy Model) ===")
    
    # Strategy: Top 3 Gap Picks, Odds 10-50
    # REPRODUCING LEAKED ROI: Using 'odds_final' for filtering
    print("Strategy: Gap Top 3 + Final Odds 10-50 (Leak Simulation)")
    picks = test_df[
        (test_df['gap_rank'] <= 3) &
        (test_df['odds_final'] >= 10.0) & 
        (test_df['odds_final'] <= 50.0)
    ].copy()
    
    print(f"Total Bets: {len(picks)}")
    
    # Hits (Rank 1-3)
    hits = picks[picks['rank'] <= 3]
    print(f"Hits: {len(hits)} (Hit Rate: {len(hits)/len(picks):.2%})")
    
    if 'odds_final' in picks.columns:
        # Win Return
        win_hits = picks[picks['rank'] == 1]
        win_return = win_hits['odds_final'].sum() * 100
        win_roi = win_return / (len(picks) * 100)
        print(f"Win ROI (Simulated): {win_roi:.2%}")

if __name__ == "__main__":
    main()

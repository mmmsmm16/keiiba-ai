
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader

def main():
    MODEL_PATH = 'models/experiments/exp_gap_v14_production/model_v14.pkl'
    DATA_PATH = "data/processed/preprocessed_data_v13_active.parquet"
    
    if not os.path.exists(DATA_PATH):
        print("Data not found.")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Needs odds_10min fallback logic or assume updated parquet has it?
    # The 'rebuild' with manual injection worked (Step 4625), so parquet HAS odds since then.
    # Wait, did I run rebuild manually again?
    # Yes, command 29813fda (Step 4625) finished successfully.
    # So DATA_PATH has odds_10min.
    
    # Valid Rows Check & Fallback
    need_odds_compute = False
    if 'odds_10min' not in df.columns:
        print("odds_10min column not found.")
        need_odds_compute = True
    elif df['odds_10min'].isna().all():
        print("odds_10min is 100% NaN.")
        need_odds_compute = True
        
    if need_odds_compute:
        print("Computing odds_fluctuation directly from DB (Fallback)...")
        from src.preprocessing.features.odds_fluctuation import compute_odds_fluctuation
        from src.preprocessing.loader import JraVanDataLoader
        
        # Ensure start_time_str
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
            print(f"Computed {len(df_odds)} odds records. Merging...")
            df['race_id'] = df['race_id'].astype(str)
            df['horse_number'] = df['horse_number'].astype(int)
            df_odds['race_id'] = df_odds['race_id'].astype(str)
            df_odds['horse_number'] = df_odds['horse_number'].astype(int)
            
            # Drop old columns
            for c in ['odds_10min', 'odds_final', 'odds_60min', 'odds_ratio_10min', 'rank_diff_10min', 'odds_log_ratio_10min', 'odds_ratio_60_10', 'ninki_10min']:
                if c in df.columns:
                    df = df.drop(columns=[c])
            
            df = pd.merge(df, df_odds.drop(columns=['horse_id'], errors='ignore'), 
                          on=['race_id', 'horse_number'], how='left')
        else:
            print("ERROR: compute_odds_fluctuation returned empty!")
            return

    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    # Filter 2025 (Test Set)
    test_df = df[df['date'].dt.year == 2025].copy()
    if len(test_df) == 0:
        print("WARN: No 2025 data found. Using 2024 data.")
        test_df = df[df['date'].dt.year == 2024].copy()
    
    # Filter JRA only (Basho 01-10)
    test_df = test_df[test_df['race_id'].str[4:6].isin(['01','02','03','04','05','06','07','08','09','10'])].copy()
    print(f"JRA Test Data rows: {len(test_df)}")
    
    if len(test_df) > 0:
        print("Sample rows from test_df:")
        cols_to_show = ['race_id', 'date', 'horse_number']
        if 'keibajo_code' in test_df.columns: cols_to_show.append('keibajo_code')
        if 'kaisai_kai' in test_df.columns: cols_to_show.append('kaisai_kai')
        if 'kaisai_nichime' in test_df.columns: cols_to_show.append('kaisai_nichime')
        if 'race_bango' in test_df.columns: cols_to_show.append('race_bango')
        print(test_df[cols_to_show].head(5))
        
    # Feature Engineering (Derived)
    # 1. odds_rank_10min
    if 'odds_10min' in test_df.columns:
        test_df['odds_rank_10min'] = test_df.groupby('race_id')['odds_10min'].rank(method='min')
    
    # 2. rank_diff_10min (ninki_final - ninki_10min) if used in model
    # Model uses it. ninki_final is 'popularity'.
    # ninki_10min is likely 'ninki_10min' if present, or we derive from odds_10min rank?
    # odds_fluctuation returns ninki_10min.
    # If missing, use calculated rank.
    if 'rank_diff_10min' not in test_df.columns:
        if 'ninki_10min' in test_df.columns:
             test_df['rank_diff_10min'] = test_df['popularity'] - test_df['ninki_10min']
        else:
             # Approx
             test_df['rank_diff_10min'] = test_df['popularity'] - test_df['odds_rank_10min']
             
    # 3. odds_log_ratio_10min
    if 'odds_log_ratio_10min' not in test_df.columns:
         test_df['odds_log_ratio_10min'] = np.log(test_df['odds'] + 1e-9) - np.log(test_df['odds_10min'] + 1e-9)
         
    # 4. odds_ratio_60_10 (If missing, set to 1.0 or NaN?)
    if 'odds_ratio_60_10' not in test_df.columns:
         test_df['odds_ratio_60_10'] = 1.0 # Buffer
         
    # 5. odds_60min (If missing, set to odds_10min)
    if 'odds_60min' not in test_df.columns:
         test_df['odds_60min'] = test_df['odds_10min']

    # Load Model
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    
    # Features
    # Must match training features exactly
    # We can read from features.csv if saved, or infer from model.
    # The training script saved features.csv.
    features_csv = 'models/experiments/exp_gap_v14_production/features.csv'
    if os.path.exists(features_csv):
        features = pd.read_csv(features_csv)['feature'].tolist()
    else:
        print("Feature list not found!")
        return
        
    print(f"Features count: {len(features)}")
    
    # Predict
    # Handle NaNs in features
    X_test = test_df[features]
    preds = model.predict(X_test)
    test_df['pred_gap'] = preds
    test_df['pred_rank'] = test_df.groupby('race_id')['pred_gap'].rank(ascending=False)
    
    # --- ROI Analysis ---
    print("\n=== ROI Analysis (2025 Test Set) ===")
    
    # Strategy: Top 3 Gap Picks (NO ODDS FILTER)
    picks = test_df[
        (test_df['pred_rank'] <= 3)
    ].copy()
    
    print(f"Total Bets: {len(picks)}")
    
    # Hits (Rank 1-3)
    hits = picks[picks['rank'] <= 3]
    print(f"Hits: {len(hits)} (Hit Rate: {len(hits)/len(picks):.2%})")
    
    # Payoff Calculation
    # We need payoff columns.
    # 'fukusho_payoff' is usually comma separated in raw string if loaded from jvd_se?
    # Or processed into numeric?
    # Let's inspect available columns related to payoff.
    # If standard Loader processed it, it might be in 'result' object or raw columns.
    # 'tansho' is usually standard. 'fukusho' is tricky.
    
    # We will try to fetch Payoff Info from 'jvd_se' or 'jvd_hr' via Loader for these specific race_id/horse_number
    # to be accurate.
    # Or check if df has it.
    
    payoff_cols = [c for c in test_df.columns if 'payoff' in c or 'haito' in c]
    print(f"Available payoff columns: {payoff_cols}")
    
    # Payoff Calculation
    print("Fetching Place Payouts from DB (jvd_hr)...")
    from src.preprocessing.loader import JraVanDataLoader
    import psycopg2
    
    loader = JraVanDataLoader()
    
    # Get List of Race IDs from picks
    target_races = picks['race_id'].unique().tolist()
    # Chunking to avoid massive query if needed, but 2024 is manageable
    
    # Query jvd_hr
    # We need to map (race_id, horse_number) -> Payout
    # jvd_hr has up to 5 place payouts.
    # Columns: 
    # kaisai_nen..race_bango (Key)
    # haraimodoshi_fukusho_1a (Payout), haraimodoshi_fukusho_1b (Umaban)
    # ... 5a, 5b
    
    conn_str = "host='host.docker.internal' port=5433 dbname='pckeiba' user='postgres' password='postgres'"
    
    try:
        conn = psycopg2.connect(conn_str)
        # Fetch all relevant races
        # Construct WHERE clause for year 2024/2025
        years = picks['date'].dt.year.unique()
        year_str = ",".join([f"'{y}'" for y in years])
        
        # Select raw components to construct ID in Python (safer than SQL concat for padding)
        q = f"""
        SELECT 
            kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
            haraimodoshi_fukusho_1a, haraimodoshi_fukusho_1b,
            haraimodoshi_fukusho_2a, haraimodoshi_fukusho_2b,
            haraimodoshi_fukusho_3a, haraimodoshi_fukusho_3b,
            haraimodoshi_fukusho_4a, haraimodoshi_fukusho_4b,
            haraimodoshi_fukusho_5a, haraimodoshi_fukusho_5b
        FROM jvd_hr
        WHERE kaisai_nen IN ({year_str})
        """
        
        df_hr = pd.read_sql(q, conn)
        conn.close()
        
        print(f"Fetched {len(df_hr)} payout records from jvd_hr.")
        if not df_hr.empty:
            sample = df_hr.iloc[0]
            nen = int(float(str(sample['kaisai_nen'])))
            place = int(float(str(sample['keibajo_code'])))
            kai = int(float(str(sample['kaisai_kai'])))
            nichi = int(float(str(sample['kaisai_nichime'])))
            race = int(float(str(sample['race_bango'])))
            rid_sample = f"{nen}{place:02}{kai:02}{nichi:02}{race:02}"
            print(f"Sample Constructed ID: {rid_sample}")
        
        # Build Payout Map
        payout_map = {}
        for _, row in df_hr.iterrows():
            # ID Construction: cast components to int then format
            try:
               nen = int(float(str(row['kaisai_nen'])))
               place = int(float(str(row['keibajo_code'])))
               kai = int(float(str(row['kaisai_kai'])))
               nichi = int(float(str(row['kaisai_nichime'])))
               race = int(float(str(row['race_bango'])))
               
               rid = f"{nen}{place:02}{kai:02}{nichi:02}{race:02}"
            except:
               continue

            # Extract payouts
            payout_found = False
            for i in range(1, 6):
                uma = row.get(f'haraimodoshi_fukusho_{i}a') # a is Horse No
                pay = row.get(f'haraimodoshi_fukusho_{i}b') # b is Payout
                try:
                    if pd.notna(pay) and pd.notna(uma):
                        pay_val = int(float(str(pay).replace(',', '')))
                        uma_val = int(float(str(uma)))
                        if pay_val > 0:
                            payout_map[(rid, uma_val)] = pay_val
                except Exception:
                    pass
                    
        # Apply to picks
        def get_payout(row):
            return payout_map.get((row['race_id'], int(row['horse_number'])), 0)
            
        picks['place_payout'] = picks.apply(get_payout, axis=1)
        
        # Calculate ROI
        total_investment = len(picks) * 100
        total_return = picks['place_payout'].sum()
        roi = total_return / total_investment if total_investment > 0 else 0
        
        print(f"Total Return: {total_return} Yen")
        print(f"Place ROI: {roi:.2%}")
        
        print("\n--- Breakdown by Rank ---")
        for r in [1.0, 2.0, 3.0]:
            r_picks = picks[picks['pred_rank'] == r]
            r_hits = r_picks[r_picks['rank'] <= 3]
            r_total_inv = len(r_picks) * 100
            r_total_ret = r_picks['place_payout'].sum()
            r_roi = r_total_ret / r_total_inv if r_total_inv > 0 else 0
            r_hr = len(r_hits) / len(r_picks) if len(r_picks) > 0 else 0
            print(f"Rank {int(r)}: HR {r_hr:.2%}, ROI {r_roi:.2%}, Bets {len(r_picks)}")
        
    except Exception as e:
        print(f"DB Error: {e}")

if __name__ == "__main__":
    main()

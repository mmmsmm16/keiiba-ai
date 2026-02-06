
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import sys
import pickle
from sqlalchemy import create_engine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
MODEL_PATH = "models/experiments/exp_t2_refined_v3_top3/model.pkl"

def get_db_engine():
    import os
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'postgres')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_place_odds(year=2024):
    """Load pre-race Place Odds (Min) from apd_sokuho_o1"""
    logger.info(f"Loading Place Odds for {year} from DB (apd_sokuho_o1)...")
    
    # We need the LATEST record per race_id
    # Since race_id is constructed, we query raw columns and construct race_id in Python
    # or typically apd_sokuho_o1 doesn't have race_id column, we construct it.
    
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        odds_fukusho,
        happyo_tsukihi_jifun
    FROM apd_sokuho_o1
    WHERE kaisai_nen = '{year}'
    """
    
    engine = get_db_engine()
    try:
        # Load all (might be heavy, but filter by max timestamp later)
        # To optimize, we could use DISTINCT ON in SQL if supported, or just process in pandas
        df = pd.read_sql(query, engine)
        if df.empty: return pd.DataFrame()
        
        # Sort by timestamp desc per race and keep top
        df = df.sort_values(['race_id', 'happyo_tsukihi_jifun'], ascending=[True, False])
        df = df.drop_duplicates(subset=['race_id'], keep='first')
        
        # Parse Function
        def parse_odds(row):
            raw = row['odds_fukusho']
            if not raw or len(raw) < 12: return []
            
            # Chunk by 12
            recs = []
            rid = row['race_id']
            # Remove trailing spaces or garbage
            raw = raw.strip()
            
            for i in range(0, len(raw), 12):
                chunk = raw[i:i+12]
                if len(chunk) < 12: break
                try:
                    h_num = int(chunk[0:2])
                    min_o = int(chunk[2:6]) / 10.0
                    max_o = int(chunk[6:10]) / 10.0
                    pop = int(chunk[10:12])
                    recs.append({
                        'race_id': rid,
                        'horse_number': h_num,
                        'odds_min': min_o,
                        'odds_max': max_o
                    })
                except:
                    continue
            return recs

        parsed_list = []
        for _, row in df.iterrows():
            parsed_list.extend(parse_odds(row))
            
        odds_df = pd.DataFrame(parsed_list)
        return odds_df
        
    except Exception as e:
        logger.error(f"Failed to load odds from apd_sokuho_o1: {e}")
        return pd.DataFrame()

def load_data_and_odds():
    engine = get_db_engine()
    
    # 1. Load Test Data
    logger.info("Loading test data (2024)...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df_test = df[df['date'].dt.year == 2024].copy()
    
    # Filter JRA
    df_test['venue_code'] = df_test['race_id'].astype(str).str[4:6]
    jra_mask = df_test['venue_code'].isin([str(i).zfill(2) for i in range(1, 11)])
    df_test = df_test[jra_mask]
    
    # 2. Load Place Payouts
    logger.info("Loading Place Payouts...")
    q_pay = """
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_fukusho_1a, haraimodoshi_fukusho_1b,
        haraimodoshi_fukusho_2a, haraimodoshi_fukusho_2b,
        haraimodoshi_fukusho_3a, haraimodoshi_fukusho_3b,
        haraimodoshi_fukusho_4a, haraimodoshi_fukusho_4b,
        haraimodoshi_fukusho_5a, haraimodoshi_fukusho_5b
    FROM jvd_hr
    WHERE kaisai_nen = '2024'
    """
    payouts = pd.read_sql(q_pay, engine)
    
    # 3. Load Place Odds
    odds_df = load_place_odds(2024)
        
    return df_test, odds_df, payouts

def main():
    df_test, odds_df, payouts = load_data_and_odds()
    if df_test is None: return

    # Merge Odds
    logger.info("Merging Odds...")
    # Ensure keys match
    df_test['race_id'] = df_test['race_id'].astype(str)
    df_test['horse_number'] = pd.to_numeric(df_test['horse_number'], errors='coerce')
    
    if hasattr(odds_df, 'columns'):
        df = df_test.merge(odds_df[['race_id', 'horse_number', 'odds_min']], 
                           on=['race_id', 'horse_number'], how='left')
    else:
        df = df_test
        df['odds_min'] = 0 # Fail safe
        
    # Load Model (Top3)
    logger.info(f"Loading Top3 Model from {MODEL_PATH}...")
    import joblib
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    # Load Config to identify categorical features
    import os
    config_path = f"{os.path.dirname(MODEL_PATH)}/config_copy.yaml"
    import yaml
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cat_cols = cfg['dataset'].get('categorical_features', [])
    except:
        logger.warning(f"Could not load config from {config_path}, guessing categoricals...")
        cat_cols = []
        
    feature_names = model.feature_name()
    X_test = df[feature_names].copy()
    
    # Preprocess: Convert categoricals to CODES (integers)
    # This aligns with LightGBM's expectation for categorical features if passing valid integers/floats
    for col in X_test.columns:
        if X_test[col].dtype.name == 'category' or X_test[col].dtype == 'object':
             X_test[col] = X_test[col].astype('category').cat.codes
        else:
             # Identify non-numeric and force to numeric
             pass

    logger.info("Predicting Top3 Probabilities (Bypassing Pandas Check)...")
    # Pass .values to bypass strictly pandas metadata check
    # Ensure float32 for safety
    X_values = X_test.values.astype(np.float32)
    raw_preds = model.predict(X_values)
    
    # Apply Calibration if available
    calib_path = f"{os.path.dirname(MODEL_PATH)}/calibrator.pkl"
    if os.path.exists(calib_path):
        logger.info(f"Applying Calibration from {calib_path}...")
        calib = joblib.load(calib_path)
        df['prob_top3'] = calib.transform(raw_preds)
    else:
        logger.warning("No calibrator found! Using raw scores (Miscalibrated).")
        df['prob_top3'] = raw_preds
    
    # Calculate EV
    # EV = Prob(Top3) * Odds_Min
    df['ev'] = df['prob_top3'] * df['odds_min']
    
    # Payout Lookup Map
    logger.info("Preparing Payout Map...")
    # Map race_id -> set of winning horses and their payouts
    payout_map = {}
    for _, row in payouts.iterrows():
        rid = row['race_id']
        wins = {}
        # standard 1-3 places
        for i in range(1, 6): # up to 5 places for safety
            h_col = f"haraimodoshi_fukusho_{i}a"
            p_col = f"haraimodoshi_fukusho_{i}b"
            if h_col in row and pd.notnull(row[h_col]) and row[h_col] != '':
                try:
                    h_num = int(row[h_col])
                    pay = float(row[p_col])
                    wins[h_num] = pay
                except: continue
        payout_map[rid] = wins

    # Grid Search
    thresholds = [0.0, 0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]
    results = []
    
    logger.info("Running Place EV Grid Search...")
    
    for th in thresholds:
        # Bets: All horses where EV > Threshold
        bets = df[df['ev'] >= th].copy()
        
        n_bets = len(bets)
        cost = n_bets * 100
        
        return_amt = 0
        hits = 0
        
        for _, row in bets.iterrows():
            rid = row['race_id']
            h_num = row['horse_number']
            
            if rid in payout_map and h_num in payout_map[rid]:
                # WIN
                amt = payout_map[rid][h_num]
                return_amt += amt
                hits += 1
                
        roi = (return_amt / cost * 100) if cost > 0 else 0
        profit = return_amt - cost
        hit_rate = (hits / n_bets * 100) if n_bets > 0 else 0
        
        results.append({
            "EV_Threshold": th,
            "Bets": n_bets,
            "Hits": hits,
            "HitRate": hit_rate,
            "ROI": roi,
            "Profit": profit,
            "Return": return_amt,
            "BetsPerYear": n_bets
        })
        
    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print(" 🥉 Place (Fukusho) EV Grid Search Results (JRA 2024)")
    print("   Model: Top3 Model (Prob > Top3)")
    print("   EV = Prob * Min_Odds")
    print("="*80)
    print(res_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("="*80)

if __name__ == "__main__":
    main()

"""
LambdaRank ROI Simulation
=========================
Compares ROI of LambdaRank model vs Baseline Binary model.
Uses top-k selection strategy.

Usage:
  python scripts/experiments/lambdarank_roi_sim.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
BINARY_MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
LAMBDA_MODEL_PATH = "models/experiments/exp_lambdarank/model.pkl"

def get_db_engine():
    user = os.getenv("POSTGRES_USER", "postgres")
    pw = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "host.docker.internal")
    port = os.getenv("POSTGRES_PORT", "5433")
    db = os.getenv("POSTGRES_DB", "pckeiba")
    return create_engine(f'postgresql://{user}:{pw}@{host}:{port}/{db}')

def load_payouts(years=[2024]):
    engine = get_db_engine()
    year_list = ','.join([f"'{y}'" for y in years])
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay,
        haraimodoshi_fukusho_1a as place1_horse, haraimodoshi_fukusho_1b as place1_pay,
        haraimodoshi_fukusho_2a as place2_horse, haraimodoshi_fukusho_2b as place2_pay,
        haraimodoshi_fukusho_3a as place3_horse, haraimodoshi_fukusho_3b as place3_pay
    FROM jvd_hr WHERE kaisai_nen IN ({year_list})
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df['race_id'].astype(str)
    return df

def parse_payouts(df_pay):
    payout_dict = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        try:
            payout_dict[rid] = {
                'win': {'horse': int(row['win_horse']), 'pay': int(row['win_pay'])},
                'place': []
            }
            for i in range(1, 4):
                h = row[f'place{i}_horse']
                p = row[f'place{i}_pay']
                if pd.notna(h) and int(h) > 0:
                    payout_dict[rid]['place'].append({'horse': int(h), 'pay': int(p)})
        except:
            pass
    return payout_dict

def load_data():
    logger.info("Loading features...")
    df = pd.read_parquet(DATA_PATH)
    
    # Load targets only for rank if needed
    # targets = pd.read_parquet(TARGET_PATH)
    
    df['race_id'] = df['race_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    return df

def prepare_features(df, feature_names):
    X = df[feature_names].copy()
    for c in X.columns:
        if X[c].dtype == 'object' or X[c].dtype.name == 'category':
            X[c] = X[c].astype('category').cat.codes
        X[c] = X[c].fillna(-999)
    return X.values.astype(np.float64)

def simulate_roi(df_test, preds, payout_dict, model_name):
    """Simulate Top-1 Win and Top-3 Place betting"""
    df = df_test.copy()
    df['score'] = preds
    
    # Ranking
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    win_bets = 0
    win_hits = 0
    win_return = 0
    
    place_bets = 0
    place_hits = 0
    place_return = 0
    
    race_ids = df['race_id'].unique()
    
    for rid in race_ids:
        if rid not in payout_dict:
            continue
            
        grp = df[df['race_id'] == rid]
        pay = payout_dict[rid]
        
        # --- Win Bet (Top 1) ---
        top1 = grp[grp['pred_rank'] == 1]
        if not top1.empty:
            h = int(top1.iloc[0]['horse_number'])
            win_bets += 1
            if pay['win']['horse'] == h:
                win_hits += 1
                win_return += pay['win']['pay']
        
        # --- Place Bet (Top 3) ---
        top3 = grp[grp['pred_rank'] <= 3]
        for _, row in top3.iterrows():
            h = int(row['horse_number'])
            place_bets += 1
            # Check hit
            for p in pay['place']:
                if p['horse'] == h:
                    place_hits += 1
                    place_return += p['pay']
                    break
    
    win_roi = win_return / (win_bets * 100) * 100 if win_bets > 0 else 0
    place_roi = place_return / (place_bets * 100) * 100 if place_bets > 0 else 0
    
    print(f"\n--- {model_name} Results ---")
    print(f"Win (Top 1): Bets={win_bets}, Hits={win_hits} ({win_hits/win_bets*100:.1f}%), ROI={win_roi:.1f}%")
    print(f"Place (Top 3): Bets={place_bets}, Hits={place_hits} ({place_hits/place_bets*100:.1f}%), ROI={place_roi:.1f}%")
    
    return win_roi, place_roi

def run_simulation():
    logger.info("=" * 60)
    logger.info("ROI Simulation: Binary vs LambdaRank")
    logger.info("=" * 60)
    
    # Load Models
    binary_model = joblib.load(BINARY_MODEL_PATH)
    lambda_model = joblib.load(LAMBDA_MODEL_PATH)
    
    features_bin = binary_model.feature_name()
    features_lambda = lambda_model.feature_name() # Should be same, but safe check
    
    # Load Data
    df = load_data()
    df_test = df[df['date'].dt.year == 2024].copy()
    logger.info(f"Test set: {len(df_test)} records")
    
    # Load Payouts
    df_pay = load_payouts([2024])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Payouts loaded: {len(payout_dict)} races")
    
    # Predictions
    logger.info("Predicting Binary Model...")
    X_bin = prepare_features(df_test, features_bin)
    preds_bin = binary_model.predict(X_bin)
    
    logger.info("Predicting LambdaRank Model...")
    X_lambda = prepare_features(df_test, features_lambda)
    preds_lambda = lambda_model.predict(X_lambda)
    
    # Run Sim
    simulate_roi(df_test, preds_bin, payout_dict, "Baseline (Binary)")
    simulate_roi(df_test, preds_lambda, payout_dict, "LambdaRank")

if __name__ == "__main__":
    run_simulation()

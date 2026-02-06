
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
MODEL_PATH = "models/experiments/optuna_best_full/model.pkl"

def get_db_engine():
    import os
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'postgres')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_payouts(year=2024):
    logger.info(f"Loading actual payouts for {year} from DB (jvd_hr)...")
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_umaren_1a as umaren_comb,
        haraimodoshi_umaren_1b as umaren_pay,
        haraimodoshi_umatan_1a as umatan_comb,
        haraimodoshi_umatan_1b as umatan_pay
    FROM jvd_hr
    WHERE kaisai_nen = '{year}'
    """
    engine = get_db_engine()
    try:
        payouts = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(payouts)} payout records.")
        # Ensure proper types
        payouts['umaren_pay'] = pd.to_numeric(payouts['umaren_pay'], errors='coerce').fillna(0)
        payouts['umatan_pay'] = pd.to_numeric(payouts['umatan_pay'], errors='coerce').fillna(0)
        return payouts
    except Exception as e:
        logger.error(f"Failed to load payouts: {e}")
        return pd.DataFrame()

def main():
    logger.info("loading test data...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df_test = df[df['date'].dt.year == 2024].copy()
    
    # Filter JRA
    df_test['venue_code'] = df_test['race_id'].astype(str).str[4:6]
    jra_mask = df_test['venue_code'].isin([str(i).zfill(2) for i in range(1, 11)])
    df_test = df_test[jra_mask]
    logger.info(f"Test Data (JRA 2024): {len(df_test)} rows")
    
    # Load Model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    feature_names = model.feature_name()
    X_test = df_test[feature_names].copy()
    
    # Preprocess (cat.codes)
    for col in X_test.columns:
        if X_test[col].dtype.name == 'category' or X_test[col].dtype == 'object':
             X_test[col] = X_test[col].astype('category').cat.codes
        else:
             X_test[col] = X_test[col].fillna(-999)

    logger.info("Predicting...")
    df_test['prob'] = model.predict(X_test)
    
    # Rank
    df_test['race_id'] = df_test['race_id'].astype(str)
    df_test['pred_rank'] = df_test.groupby('race_id')['prob'].rank(ascending=False, method='first')
    
    # Extract Rank 1 and 2
    r1 = df_test[df_test['pred_rank'] == 1][['race_id', 'horse_number']].rename(columns={'horse_number': 'h1'})
    r2 = df_test[df_test['pred_rank'] == 2][['race_id', 'horse_number']].rename(columns={'horse_number': 'h2'})
    
    bet_df = r1.merge(r2, on='race_id')
    
    # Format combinations for matching
    # Umaren: "0102" (Smallest first)
    # Umatan: "0102" (Exact order)
    def fmt(h):
        return str(int(h)).zfill(2)
        
    bet_df['h1_s'] = bet_df['h1'].apply(fmt)
    bet_df['h2_s'] = bet_df['h2'].apply(fmt)
    
    bet_df['my_umatan'] = bet_df['h1_s'] + bet_df['h2_s']
    
    def make_umaren(row):
        a, b = int(row['h1']), int(row['h2'])
        mn, mx = min(a,b), max(a,b)
        return str(mn).zfill(2) + str(mx).zfill(2)
        
    bet_df['my_umaren'] = bet_df.apply(make_umaren, axis=1)
    
    # Load Payouts
    payouts = load_payouts(2024)
    if payouts.empty:
        logger.error("No payout data found.")
        return

    # Merge
    merged = bet_df.merge(payouts, on='race_id', how='left')
    
    # Calculate Results
    # Umatan (Exacta)
    merged['hit_umatan'] = merged['my_umatan'] == merged['umatan_comb']
    merged['ret_umatan'] = merged.apply(lambda x: x['umatan_pay'] if x['hit_umatan'] else 0, axis=1)
    
    # Umaren (Quinella)
    merged['hit_umaren'] = merged['my_umaren'] == merged['umaren_comb']
    merged['ret_umaren'] = merged.apply(lambda x: x['umaren_pay'] if x['hit_umaren'] else 0, axis=1)
    
    # Stats
    n_races = len(merged)
    cost = n_races * 100
    
    # Exacta
    ret_exacta = merged['ret_umatan'].sum()
    roi_exacta = (ret_exacta / cost) * 100
    acc_exacta = (merged['hit_umatan'].sum() / n_races) * 100
    profit_exacta = ret_exacta - cost
    
    # Quinella
    ret_quinella = merged['ret_umaren'].sum()
    roi_quinella = (ret_quinella / cost) * 100
    acc_quinella = (merged['hit_umaren'].sum() / n_races) * 100
    profit_quinella = ret_quinella - cost
    
    print("\n" + "="*60)
    print(" 🎯 Exacta (Umatan) & Quinella (Umaren) ROI Simulation")
    print(f"    Target: Predicted Rank 1 & Rank 2 (1 Ticket/Race)")
    print("="*60)
    print(f"Total Races: {n_races}")
    print("-" * 60)
    print(f"Exacta   (1->2) : ROI {roi_exacta:.2f}% | HitRate {acc_exacta:.2f}% | Profit {profit_exacta:,.0f}")
    print(f"Quinella (1-2)  : ROI {roi_quinella:.2f}% | HitRate {acc_quinella:.2f}% | Profit {profit_quinella:,.0f}")
    print("="*60)

if __name__ == "__main__":
    main()

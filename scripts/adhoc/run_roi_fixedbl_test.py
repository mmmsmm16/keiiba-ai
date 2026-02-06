"""
ROI Grid Search + 2025 Walk-Forward Test
With Fixed Baseline Features
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'postgres')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_payout_data(year):
    engine = get_db_engine()
    query = f"""
    SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
        haraimodoshi_tansho_1a, haraimodoshi_tansho_1b,
        haraimodoshi_umatan_1a, haraimodoshi_umatan_1b,
        haraimodoshi_umatan_2a, haraimodoshi_umatan_2b,
        haraimodoshi_umatan_3a, haraimodoshi_umatan_3b
    FROM jvd_hr WHERE kaisai_nen = '{year}'
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df.apply(lambda r: f"{r['kaisai_nen']}{str(r['keibajo_code']).zfill(2)}{str(r['kaisai_kai']).zfill(2)}{str(r['kaisai_nichime']).zfill(2)}{str(r['race_bango']).zfill(2)}", axis=1)
    return df

def parse_payouts(df_pay):
    payouts = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        payouts[rid] = {'win': {}, 'exacta': {}}
        
        if pd.notna(row.get('haraimodoshi_tansho_1a')):
            try: payouts[rid]['win'][int(row['haraimodoshi_tansho_1a'])] = int(row['haraimodoshi_tansho_1b'])
            except: pass
            
        for i in range(1, 4):
            if pd.notna(row.get(f'haraimodoshi_umatan_{i}a')):
                try:
                    h_str = str(row[f'haraimodoshi_umatan_{i}a']).zfill(4)
                    payouts[rid]['exacta'][(int(h_str[:2]), int(h_str[2:]))] = int(row[f'haraimodoshi_umatan_{i}b'])
                except: pass
    return payouts

def load_odds_from_db(year):
    engine = get_db_engine()
    query = f"SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, umaban, tansho_odds FROM jvd_se WHERE kaisai_nen = '{year}'"
    df = pd.read_sql(query, engine)
    df['race_id'] = df.apply(lambda r: f"{r['kaisai_nen']}{str(r['keibajo_code']).zfill(2)}{str(r['kaisai_kai']).zfill(2)}{str(r['kaisai_nichime']).zfill(2)}{str(r['race_bango']).zfill(2)}", axis=1)
    df['horse_number'] = df['umaban'].astype(int)
    df['odds_final'] = pd.to_numeric(df['tansho_odds'], errors='coerce').fillna(1.0)
    return df[['race_id', 'horse_number', 'odds_final']]

def simulate_exacta(df_data, payout_dict, conf_th, min_odds, flow_n):
    cost, ret, races_bet = 0, 0, 0
    
    for rid, group in df_data.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        
        group = group.sort_values('pred_prob', ascending=False)
        scores = group['pred_prob'].values
        horses = group['horse_number'].values
        odds = group['odds_final'].values
        
        if len(horses) < flow_n + 1: continue
        if scores[0] < conf_th: continue
        if odds[0] < min_odds: continue
        
        races_bet += 1
        axis = horses[0]
        flow = horses[1:flow_n+1]
        
        for f in flow:
            cost += 100
            key = (axis, f)
            if key in pay['exacta']:
                ret += pay['exacta'][key]
    
    roi = ret / cost * 100 if cost > 0 else 0
    return {'cost': cost, 'return': ret, 'roi': roi, 'races': races_bet}

def main():
    logger.info("=" * 60)
    logger.info("ROI Grid Search + 2025 Walk-Forward (Fixed Baseline)")
    logger.info("=" * 60)
    
    # Load 2024 model
    MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
    model = joblib.load(MODEL_PATH)
    
    # Get feature names
    if hasattr(model, 'booster_'):
        feat_cols = model.booster_.feature_name()
    else:
        feat_cols = model.feature_name()
    
    # Load fixed baseline features
    df = pd.read_parquet('data/features/temp_merge_fixed_baseline.parquet')
    df['race_id'] = df['race_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Load targets
    tgt = pd.read_parquet('data/temp_t2/T2_targets.parquet')
    tgt['race_id'] = tgt['race_id'].astype(str)
    tgt['horse_number'] = pd.to_numeric(tgt['horse_number'], errors='coerce').fillna(0).astype(int)
    
    # Load odds
    df_odds_24 = load_odds_from_db(2024)
    df_odds_25 = load_odds_from_db(2025)
    df_odds = pd.concat([df_odds_24, df_odds_25], ignore_index=True)
    df_odds['race_id'] = df_odds['race_id'].astype(str)
    
    # Prepare 2024 data
    df_2024 = df[df['year'] == 2024].copy()
    df_2024['horse_number'] = pd.to_numeric(df_2024['horse_number'], errors='coerce').fillna(0).astype(int)
    df_2024 = pd.merge(df_2024, tgt[['race_id', 'horse_number', 'rank']], on=['race_id', 'horse_number'], how='left')
    df_2024 = pd.merge(df_2024, df_odds, on=['race_id', 'horse_number'], how='left')
    df_2024['odds_final'] = df_2024['odds_final'].fillna(1.0)
    
    # Predict 2024
    for c in feat_cols:
        if c not in df_2024.columns:
            df_2024[c] = 0
    X_24 = df_2024[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_2024['pred_prob'] = model.predict(X_24.values.astype(np.float32))
    
    logger.info(f"2024: {len(df_2024)} rows, Prob 95%: {df_2024['pred_prob'].quantile(0.95):.4f}")
    
    # Prepare 2025 data
    df_2025 = df[df['year'] == 2025].copy()
    df_2025['horse_number'] = pd.to_numeric(df_2025['horse_number'], errors='coerce').fillna(0).astype(int)
    df_2025 = pd.merge(df_2025, df_odds, on=['race_id', 'horse_number'], how='left')
    df_2025['odds_final'] = df_2025['odds_final'].fillna(1.0)
    
    # Predict 2025 with 2024 model (for comparison)
    for c in feat_cols:
        if c not in df_2025.columns:
            df_2025[c] = 0
    X_25 = df_2025[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_2025['pred_prob'] = model.predict(X_25.values.astype(np.float32))
    
    logger.info(f"2025: {len(df_2025)} rows, Prob 95%: {df_2025['pred_prob'].quantile(0.95):.4f}")
    
    # Load payouts
    pay_2024 = parse_payouts(load_payout_data(2024))
    pay_2025 = parse_payouts(load_payout_data(2025))
    
    logger.info("\n" + "=" * 60)
    logger.info("GRID SEARCH: Exacta Nagashi")
    logger.info("=" * 60)
    
    results = []
    # Lower thresholds due to fixed baseline probability distribution
    for conf in [0.12, 0.15, 0.18, 0.20]:
        for min_odds in [2.0, 3.0]:
            for flow in [6, 7, 8]:
                res_24 = simulate_exacta(df_2024, pay_2024, conf, min_odds, flow)
                res_25 = simulate_exacta(df_2025, pay_2025, conf, min_odds, flow)
                
                results.append({
                    'Strategy': f'Exacta C>{conf} Od>{min_odds} F{flow}',
                    'ROI_24': round(res_24['roi'], 1),
                    'Races_24': res_24['races'],
                    'ROI_25': round(res_25['roi'], 1),
                    'Races_25': res_25['races'],
                    'Profit_25': res_25['return'] - res_25['cost']
                })
    
    df_res = pd.DataFrame(results).sort_values('ROI_24', ascending=False)
    if len(df_res) > 0:
        print("\n========= Strategies with ROI > 95% in 2024 =========")
        print(df_res.to_string(index=False))
    else:
        print("No strategies with ROI > 95% found.")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Fixed Baseline Effect")
    logger.info("=" * 60)
    
    if len(df_res) > 0:
        avg_24 = df_res['ROI_24'].mean()
        avg_25 = df_res['ROI_25'].mean()
        logger.info(f"Average ROI 2024: {avg_24:.1f}%")
        logger.info(f"Average ROI 2025: {avg_25:.1f}%")
        logger.info(f"Degradation: {avg_24 - avg_25:.1f}%")

if __name__ == "__main__":
    main()

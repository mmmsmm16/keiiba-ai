
import os
import sys
import logging
import pandas as pd
import numpy as np
import itertools
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
    SELECT 
        kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
        haraimodoshi_tansho_1a, haraimodoshi_tansho_1b,
        haraimodoshi_fukusho_1a, haraimodoshi_fukusho_1b,
        haraimodoshi_fukusho_2a, haraimodoshi_fukusho_2b,
        haraimodoshi_fukusho_3a, haraimodoshi_fukusho_3b,
        haraimodoshi_wide_1a, haraimodoshi_wide_1b,
        haraimodoshi_wide_2a, haraimodoshi_wide_2b,
        haraimodoshi_wide_3a, haraimodoshi_wide_3b,
        haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
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
        payouts[rid] = {'win': {}, 'place': {}, 'wide': {}, 'quinella': {}, 'exacta': {}}
        
        if pd.notna(row.get('haraimodoshi_tansho_1a')):
            try: payouts[rid]['win'][int(row['haraimodoshi_tansho_1a'])] = int(row['haraimodoshi_tansho_1b'])
            except: pass

        for i in range(1, 4):
            if pd.notna(row.get(f'haraimodoshi_fukusho_{i}a')):
                try: payouts[rid]['place'][int(row[f'haraimodoshi_fukusho_{i}a'])] = int(row[f'haraimodoshi_fukusho_{i}b'])
                except: pass
                
        for i in range(1, 4):
            if pd.notna(row.get(f'haraimodoshi_wide_{i}a')):
                try:
                    h_str = str(row[f'haraimodoshi_wide_{i}a']).zfill(4)
                    payouts[rid]['wide'][frozenset({int(h_str[:2]), int(h_str[2:])})] = int(row[f'haraimodoshi_wide_{i}b'])
                except: pass
                
        if pd.notna(row.get('haraimodoshi_umaren_1a')):
            try:
                h_str = str(row['haraimodoshi_umaren_1a']).zfill(4)
                payouts[rid]['quinella'][frozenset({int(h_str[:2]), int(h_str[2:])})] = int(row['haraimodoshi_umaren_1b'])
            except: pass
            
        for i in range(1, 4):
            if pd.notna(row.get(f'haraimodoshi_umatan_{i}a')):
                try:
                    h_str = str(row[f'haraimodoshi_umatan_{i}a']).zfill(4)
                    payouts[rid]['exacta'][(int(h_str[:2]), int(h_str[2:]))] = int(row[f'haraimodoshi_umatan_{i}b'])
                except: pass
    return payouts

def simulate_simple(df_data, payout_dict, conf_th, min_odds, flow_n):
    """Simplified simulation for Exacta Nagashi strategy."""
    cost, ret, races_bet = 0, 0, 0
    
    for rid, group in df_data.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        
        group = group.sort_values('pred_prob', ascending=False)
        scores = group['pred_prob'].values
        horses = group['horse_number'].values
        odds = group['odds_final'].values
        
        if scores[0] < conf_th: continue
        if odds[0] < min_odds: continue
        if len(horses) < flow_n + 1: continue
        
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

def load_odds_from_db(year):
    engine = get_db_engine()
    query = f"SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, umaban, tansho_odds FROM jvd_se WHERE kaisai_nen = '{year}'"
    df = pd.read_sql(query, engine)
    df['race_id'] = df.apply(lambda r: f"{r['kaisai_nen']}{str(r['keibajo_code']).zfill(2)}{str(r['kaisai_kai']).zfill(2)}{str(r['kaisai_nichime']).zfill(2)}{str(r['race_bango']).zfill(2)}", axis=1)
    df['horse_number'] = df['umaban'].astype(int)
    df['odds_final'] = pd.to_numeric(df['tansho_odds'], errors='coerce').fillna(1.0)
    return df[['race_id', 'horse_number', 'odds_final']]

def main():
    logger.info("🚀 Walk-Forward ROI Validation (Adjusted Thresholds)...")
    
    # Load predictions
    pred_path_24 = "data/temp_t2/T2_predictions_2024_2025.parquet"
    pred_path_25 = "data/temp_t2/T2_predictions_2025_walkforward.parquet"
    
    df_odds_24 = load_odds_from_db(2024)
    df_odds_25 = load_odds_from_db(2025)
    df_odds = pd.concat([df_odds_24, df_odds_25], ignore_index=True)
    df_odds['race_id'] = df_odds['race_id'].astype(str)
    
    # Load 2024
    df_pred_24 = pd.read_parquet(pred_path_24)
    df_pred_24['race_id'] = df_pred_24['race_id'].astype(str)
    df_pred_24['date'] = pd.to_datetime(df_pred_24['date'])
    df_2024 = df_pred_24[df_pred_24['date'].dt.year == 2024].copy()
    df_2024 = pd.merge(df_2024, df_odds, on=['race_id', 'horse_number'], how='left')
    df_2024['odds_final'] = df_2024['odds_final'].fillna(1.0)
    
    # Load 2025
    df_2025 = pd.read_parquet(pred_path_25)
    df_2025['race_id'] = df_2025['race_id'].astype(str)
    if 'odds_final' in df_2025.columns:
        df_2025 = df_2025.drop(columns=['odds_final'])
    df_2025 = pd.merge(df_2025, df_odds, on=['race_id', 'horse_number'], how='left')
    df_2025['odds_final'] = df_2025['odds_final'].fillna(1.0)
    
    logger.info(f"2024 Prob: Mean={df_2024['pred_prob'].mean():.4f}, 95%={df_2024['pred_prob'].quantile(0.95):.4f}")
    logger.info(f"2025 Prob: Mean={df_2025['pred_prob'].mean():.4f}, 95%={df_2025['pred_prob'].quantile(0.95):.4f}")
    
    pay_2024 = parse_payouts(load_payout_data(2024))
    pay_2025 = parse_payouts(load_payout_data(2025))
    
    # Grid Search with adjusted thresholds
    # 2024 thresholds -> 2025 equivalent (scaled by 95% quantile ratio)
    ratio = df_2025['pred_prob'].quantile(0.95) / df_2024['pred_prob'].quantile(0.95)
    logger.info(f"Threshold Ratio (2025/2024): {ratio:.3f}")
    
    results = []
    
    for conf_24 in [0.40, 0.45, 0.50, 0.55]:
        conf_25 = conf_24 * ratio  # Scale threshold
        for min_odds in [2.0, 3.0]:
            for flow in [6, 7, 8]:
                res_24 = simulate_simple(df_2024, pay_2024, conf_24, min_odds, flow)
                res_25 = simulate_simple(df_2025, pay_2025, conf_25, min_odds, flow)
                
                if res_24['roi'] > 95:  # Only promising strategies
                    results.append({
                        'Strategy': f'Exacta C>{conf_24:.2f}(25:{conf_25:.2f}) Od>{min_odds} Flow{flow}',
                        'ROI_24': res_24['roi'],
                        'Races_24': res_24['races'],
                        'ROI_25': res_25['roi'],
                        'Races_25': res_25['races'],
                        'Profit_25': res_25['return'] - res_25['cost']
                    })
    
    df_res = pd.DataFrame(results).sort_values('ROI_24', ascending=False)
    print("\n========= Walk-Forward ROI (Adjusted Thresholds) =========")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()


import sys
import os
import logging
import pandas as pd
import numpy as np
import json
import gc

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.loader import JraVanDataLoader
from utils.payout_loader import PayoutLoader, format_combination

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Parsers ---

def parse_o1_win_all(rid, raw_str):
    if not isinstance(raw_str, str) or not raw_str: return []
    res = []
    for i in range(0, len(raw_str), 8):
        chunk = raw_str[i:i+8]
        if len(chunk) < 8: break
        try:
            h = int(chunk[0:2])
            val = int(chunk[2:6])
            if val > 0: res.append({'race_id': rid, 'horse_number': h, 'win_odds': val / 10.0})
        except: continue
    return res

def parse_o1_place_all(rid, raw_str):
    if not isinstance(raw_str, str) or not raw_str: return []
    res = []
    for i in range(0, len(raw_str), 12):
        chunk = raw_str[i:i+12]
        if len(chunk) < 12: break
        try:
            h = int(chunk[0:2])
            val_min = int(chunk[2:6])
            val_max = int(chunk[6:10])
            if val_min > 0: 
                res.append({
                    'race_id': rid, 
                    'horse_number': h, 
                    'place_odds_min': val_min / 10.0,
                    'place_odds_max': val_max / 10.0,
                    'place_odds_mid': (val_min + val_max) / 20.0
                })
        except: continue
    return res

def parse_o2_umaren_dict(raw_str):
    if not isinstance(raw_str, str) or not raw_str: return {}
    res = {}
    for i in range(0, len(raw_str), 15):
        chunk = raw_str[i:i+15]
        if len(chunk) < 15: break
        try:
            h1 = int(chunk[0:2])
            h2 = int(chunk[2:4])
            val = int(chunk[4:10])
            if val > 0:
                key = tuple(sorted((h1, h2)))
                res[f"{key[0]:02}-{key[1]:02}"] = val / 10.0
        except: continue
    return res

def main():
    logger.info("🚀 Starting M5 Step 5: ROI Dataset Generation (Optimized)")
    
    PRED_PATH = "experiments/exp_m5_walkforward/walkforward_preds_2022_2024.parquet"
    if not os.path.exists(PRED_PATH):
        logger.error(f"Missing predictions: {PRED_PATH}. Run Step 4 first.")
        return
        
    # 1. Load Data
    logger.info("Loading Predictions & Metadata...")
    df_preds = pd.read_parquet(PRED_PATH)
    years = [2022, 2023, 2024]
    
    loader = JraVanDataLoader()
    engine = loader.engine
    
    query_meta = f"""
    SELECT 
        CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
        kaisai_nen as year,
        keibajo_code as venue,
        kyori as distance,
        track_code as track_type,
        tenko_code as weather,
        to_char(kaisai_nen::int, 'FM0000') || to_char(kaisai_tsukihi::int, 'FM0000') as date_str,
        hasso_jikoku as start_time_str
    FROM jvd_ra
    WHERE kaisai_nen IN ('2022', '2023', '2024')
    """
    df_meta = pd.read_sql(query_meta, engine)
    df_meta['date'] = pd.to_datetime(df_meta['date_str'], format='%Y%m%d')
    df_meta = loader._merge_odds_10min(df_meta, "2022-01-01")
    
    pay_loader = PayoutLoader()
    df_payout_raw = pay_loader.load_payout_dataframe(years)
    payout_map = pay_loader.build_payout_map(df_payout_raw)
    
    # 2. Vectorized Parsing of Odds Strings
    logger.info("Parsing Win/Place odds strings...")
    win_odds_list = []
    place_odds_list = []
    unique_races = df_meta[df_meta['odds_win_str'].notna()]
    
    for _, row in unique_races.iterrows():
        win_odds_list.extend(parse_o1_win_all(row['race_id'], row['odds_win_str']))
        place_odds_list.extend(parse_o1_place_all(row['race_id'], row['odds_place_str']))
        
    df_win_odds = pd.DataFrame(win_odds_list)
    df_place_odds = pd.DataFrame(place_odds_list)
    
    # 3. Parsing Payoffs to Horse-level
    logger.info("Expanding Payoffs to horse level...")
    win_pay_list = []
    place_pay_list = []
    for rid, p_data in payout_map.items():
        # Win
        for h_str, amt in p_data.get('tansho', {}).items():
            win_pay_list.append({'race_id': rid, 'horse_number': int(h_str), 'win_payoff': amt})
        # Place
        for h_str, amt in p_data.get('fukusho', {}).items():
            place_pay_list.append({'race_id': rid, 'horse_number': int(h_str), 'place_payoff': amt})
            
    df_win_pay = pd.DataFrame(win_pay_list)
    df_place_pay = pd.DataFrame(place_pay_list)
    
    # 4. Merge Everything
    logger.info("Merging (Vectorized)...")
    # Drop year from meta to avoid collision (df_preds has year)
    df_final = pd.merge(df_preds, df_meta.drop(columns=['odds_win_str', 'odds_place_str', 'year']), on='race_id', how='left')
    df_final = pd.merge(df_final, df_win_odds, on=['race_id', 'horse_number'], how='left')
    df_final = pd.merge(df_final, df_place_odds, on=['race_id', 'horse_number'], how='left')
    df_final = pd.merge(df_final, df_win_pay, on=['race_id', 'horse_number'], how='left')
    df_final = pd.merge(df_final, df_place_pay, on=['race_id', 'horse_number'], how='left')
    
    # Fill NAs
    df_final['win_odds'] = df_final['win_odds'].fillna(0.0)
    df_final[['place_odds_min', 'place_odds_max', 'place_odds_mid']] = df_final[['place_odds_min', 'place_odds_max', 'place_odds_mid']].fillna(0.0)
    df_final[['win_payoff', 'place_payoff']] = df_final[['win_payoff', 'place_payoff']].fillna(0)
    
    # Umaren Strings as JSON per race
    logger.info("Consolidating Umaren Odds/Payoffs...")
    umaren_json_map = {}
    umaren_pay_map = {}
    for rid, row in df_meta.iterrows():
        rid_val = row['race_id']
        umaren_json_map[rid_val] = json.dumps(parse_o2_umaren_dict(row.get('odds_umaren_str')))
        umaren_pay_map[rid_val] = json.dumps(payout_map.get(rid_val, {}).get('umaren', {}))
        
    df_final['umaren_odds_json'] = df_final['race_id'].map(umaren_json_map)
    df_final['umaren_payoff_json'] = df_final['race_id'].map(umaren_pay_map)
    
    # 5. Save
    COLS = [
        'race_id', 'horse_number', 'year', 'date', 'venue', 'distance', 'track_type',
        'p_win', 'p_top2', 'p_top3', 'ensemble_score_w1', 'ensemble_score_avg',
        'win_odds', 'place_odds_mid', 'place_odds_min', 'place_odds_max',
        'is_win', 'is_top3', 'win_payoff', 'place_payoff',
        'umaren_odds_json', 'umaren_payoff_json'
    ]
    df_final = df_final[COLS]
    
    os.makedirs("reports/simulations", exist_ok=True)
    out_path = "reports/simulations/v24_m5_roi_dataset_2022_2024.parquet"
    logger.info(f"Saving Final Dataset: {df_final.shape}")
    df_final.to_parquet(out_path)
    
    logger.info("M5 Step 5 Complete!")

if __name__ == "__main__":
    main()

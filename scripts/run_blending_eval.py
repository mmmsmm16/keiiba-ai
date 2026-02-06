"""
Blend predictions from Binary and Top3 models and evaluate ROI across bet types.
Usage: python scripts/run_blending_eval.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_db_engine():
    user = os.getenv("POSTGRES_USER", "postgres")
    pw = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "host.docker.internal")
    port = os.getenv("POSTGRES_PORT", "5433")
    db = os.getenv("POSTGRES_DB", "pckeiba")
    return create_engine(f'postgresql://{user}:{pw}@{host}:{port}/{db}')

def load_payout_data(engine, years):
    year_list = ','.join([f"'{y}'" for y in years])
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_tansho_1a as win_1_horse, haraimodoshi_tansho_1b as win_1_pay,
        haraimodoshi_fukusho_1a as place_1_horse, haraimodoshi_fukusho_1b as place_1_pay,
        haraimodoshi_fukusho_2a as place_2_horse, haraimodoshi_fukusho_2b as place_2_pay,
        haraimodoshi_fukusho_3a as place_3_horse, haraimodoshi_fukusho_3b as place_3_pay,
        haraimodoshi_wide_1a as wide_1_horses, haraimodoshi_wide_1b as wide_1_pay,
        haraimodoshi_wide_2a as wide_2_horses, haraimodoshi_wide_2b as wide_2_pay,
        haraimodoshi_wide_3a as wide_3_horses, haraimodoshi_wide_3b as wide_3_pay,
        haraimodoshi_umaren_1a as quinella_horses, haraimodoshi_umaren_1b as quinella_pay
    FROM jvd_hr 
    WHERE kaisai_nen IN ({year_list})
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df['race_id'].astype(str)
    return df

def parse_payouts(df_pay):
    payout_dict = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        pay = {'win': {}, 'place': {}, 'wide': {}, 'quinella': {}}
        try:
            h = int(row['win_1_horse'])
            p = int(row['win_1_pay'])
            if p > 0: pay['win'][h] = p
        except: pass
        for i in [1, 2, 3]:
            try:
                h = int(row[f'place_{i}_horse'])
                p = int(row[f'place_{i}_pay'])
                if p > 0: pay['place'][h] = p
            except: pass
        for i in [1, 2, 3]:
            try:
                horses_str = str(row[f'wide_{i}_horses']).zfill(4)
                h1, h2 = int(horses_str[:2]), int(horses_str[2:])
                p = int(row[f'wide_{i}_pay'])
                if p > 0:
                    pay['wide'][(min(h1,h2), max(h1,h2))] = p
            except: pass
        try:
            horses_str = str(row['quinella_horses']).zfill(4)
            h1, h2 = int(horses_str[:2]), int(horses_str[2:])
            p = int(row['quinella_pay'])
            if p > 0:
                pay['quinella'][(min(h1,h2), max(h1,h2))] = p
        except: pass
        payout_dict[rid] = pay
    return payout_dict

def simulate_bets(df_data, payout_dict, bet_type, th=0.0, min_odds=1.0, n_horses=3):
    cost = 0
    returns = 0
    hits = 0
    bets = 0
    
    for rid, group in df_data.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        
        group = group.sort_values('blend_score', ascending=False)
        top_horses = group['horse_number'].to_numpy()[:n_horses]
        scores = group['blend_score'].to_numpy()[:n_horses]
        odds = group['odds_final'].to_numpy()[:n_horses] if 'odds_final' in group.columns else np.ones(len(top_horses))
        
        if bet_type == 'win':
            if len(top_horses) >= 1 and scores[0] >= th and odds[0] >= min_odds:
                h = top_horses[0]
                cost += 100
                bets += 1
                if h in pay['win']:
                    returns += pay['win'][h]
                    hits += 1
        
        elif bet_type == 'place':
            if len(top_horses) >= 1 and scores[0] >= th and odds[0] >= min_odds:
                h = top_horses[0]
                cost += 100
                bets += 1
                if h in pay['place']:
                    returns += pay['place'][h]
                    hits += 1
        
        elif bet_type == 'wide_box':
            if len(top_horses) >= 3 and scores[0] >= th:
                from itertools import combinations
                for h1, h2 in combinations(top_horses[:3], 2):
                    cost += 100
                    bets += 1
                    key = (min(h1,h2), max(h1,h2))
                    if key in pay['wide']:
                        returns += pay['wide'][key]
                        hits += 1
        
        elif bet_type == 'quinella':
            if len(top_horses) >= 2 and scores[0] >= th:
                h1, h2 = top_horses[0], top_horses[1]
                cost += 100
                bets += 1
                key = (min(h1,h2), max(h1,h2))
                if key in pay['quinella']:
                    returns += pay['quinella'][key]
                    hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    return roi

def main():
    logger.info("🔀 Blending Experiment: Binary + Top3")
    
    # Load predictions
    pred_dir = "models/experiments/exp_t2_refined_v3/cv_results"
    df_binary = pd.read_parquet(f"{pred_dir}/preds_binary.parquet")
    df_top3 = pd.read_parquet(f"{pred_dir}/preds_top3.parquet")
    
    # Merge
    df = df_binary[['race_id', 'horse_number', 'date', 'rank', 'odds_final']].copy()
    df = df.rename(columns={'rank': 'actual_rank'})
    df['binary_prob'] = df_binary['pred_prob']
    
    df_top3['race_id'] = df_top3['race_id'].astype(str)
    df['race_id'] = df['race_id'].astype(str)
    df = pd.merge(df, df_top3[['race_id', 'horse_number', 'pred_prob']], 
                  on=['race_id', 'horse_number'], how='left', suffixes=('', '_top3'))
    df['top3_prob'] = df['pred_prob']
    df = df.drop(columns=['pred_prob'])
    
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year == 2023]
    logger.info(f"Loaded {len(df)} predictions for 2023")
    
    # Load payouts
    engine = get_db_engine()
    df_pay = load_payout_data(engine, [2023])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Loaded {len(payout_dict)} payout records")
    
    # Grid Search over alpha
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    bet_types = ['win', 'place', 'wide_box', 'quinella']
    
    print("\n" + "="*80)
    print(" Blending Results: α*Binary + (1-α)*Top3")
    print("="*80)
    print(f"{'α':<8}", end="")
    for bt in bet_types:
        print(f"{bt:<15}", end="")
    print()
    print("-"*80)
    
    best_results = {bt: {'alpha': 0, 'roi': 0} for bt in bet_types}
    
    for alpha in alphas:
        df['blend_score'] = alpha * df['binary_prob'] + (1 - alpha) * df['top3_prob']
        
        print(f"{alpha:<8.1f}", end="")
        for bt in bet_types:
            roi = simulate_bets(df, payout_dict, bt)
            print(f"{roi:<15.1f}", end="")
            if roi > best_results[bt]['roi']:
                best_results[bt]['alpha'] = alpha
                best_results[bt]['roi'] = roi
        print()
    
    print("="*80)
    print("\n🏆 Best α per bet type:")
    for bt in bet_types:
        print(f"  {bt:<15}: α={best_results[bt]['alpha']:.1f} → ROI={best_results[bt]['roi']:.1f}%")
    
    # Compare with single models
    print("\n📊 Improvement over best single model:")
    single_model_best = {'win': 78.0, 'place': 81.0, 'wide_box': 72.9, 'quinella': 65.1}
    for bt in bet_types:
        best_blend = best_results[bt]['roi']
        best_single = single_model_best[bt]
        diff = best_blend - best_single
        sign = "+" if diff >= 0 else ""
        print(f"  {bt:<15}: {best_single:.1f}% → {best_blend:.1f}% ({sign}{diff:.1f}%)")

if __name__ == "__main__":
    main()

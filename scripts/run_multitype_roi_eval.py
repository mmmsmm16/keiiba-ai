"""
Evaluate multiple bet types (Place, Wide, Quinella) using saved predictions.
Usage: python scripts/run_multitype_roi_eval.py --pred_path <path_to_preds.parquet>
"""
import os
import sys
import logging
import argparse
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
    """Load payout data for specified years."""
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
    """Parse payout dataframe into dictionary keyed by race_id."""
    payout_dict = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        pay = {'win': {}, 'place': {}, 'wide': {}, 'quinella': {}}
        
        # Win
        try:
            h = int(row['win_1_horse'])
            p = int(row['win_1_pay'])
            if p > 0: pay['win'][h] = p
        except: pass
        
        # Place (fukusho)
        for i in [1, 2, 3]:
            try:
                h = int(row[f'place_{i}_horse'])
                p = int(row[f'place_{i}_pay'])
                if p > 0: pay['place'][h] = p
            except: pass
        
        # Wide
        for i in [1, 2, 3]:
            try:
                horses_str = str(row[f'wide_{i}_horses']).zfill(4)
                h1, h2 = int(horses_str[:2]), int(horses_str[2:])
                p = int(row[f'wide_{i}_pay'])
                if p > 0:
                    pay['wide'][(min(h1,h2), max(h1,h2))] = p
            except: pass
        
        # Quinella (umaren)
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
    """
    Simulate bets for a given type.
    
    bet_type: 'win', 'place', 'wide_top', 'quinella_top'
    """
    cost = 0
    returns = 0
    hits = 0
    bets = 0
    
    for rid, group in df_data.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        
        group = group.sort_values('pred_prob', ascending=False)
        top_horses = group['horse_number'].to_numpy()[:n_horses]
        scores = group['pred_prob'].to_numpy()[:n_horses]
        odds = group['odds_final'].to_numpy()[:n_horses] if 'odds_final' in group.columns else np.ones(len(top_horses))
        
        if bet_type == 'win':
            # Top1 win bet
            if len(top_horses) >= 1 and scores[0] >= th and odds[0] >= min_odds:
                h = top_horses[0]
                cost += 100
                bets += 1
                if h in pay['win']:
                    returns += pay['win'][h]
                    hits += 1
        
        elif bet_type == 'place':
            # Top1 place bet
            if len(top_horses) >= 1 and scores[0] >= th and odds[0] >= min_odds:
                h = top_horses[0]
                cost += 100
                bets += 1
                if h in pay['place']:
                    returns += pay['place'][h]
                    hits += 1
        
        elif bet_type == 'place_top3':
            # Top3 place bets (all 3)
            for i in range(min(3, len(top_horses))):
                if scores[i] >= th and odds[i] >= min_odds:
                    h = top_horses[i]
                    cost += 100
                    bets += 1
                    if h in pay['place']:
                        returns += pay['place'][h]
                        hits += 1
        
        elif bet_type == 'wide_box':
            # Wide box (Top3 combinations)
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
            # Top2 quinella
            if len(top_horses) >= 2 and scores[0] >= th:
                h1, h2 = top_horses[0], top_horses[1]
                cost += 100
                bets += 1
                key = (min(h1,h2), max(h1,h2))
                if key in pay['quinella']:
                    returns += pay['quinella'][key]
                    hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    hit_rate = (hits / bets * 100) if bets > 0 else 0
    return {'type': bet_type, 'bets': bets, 'hits': hits, 'hit_rate': hit_rate, 'cost': cost, 'returns': returns, 'roi': roi}

def main():
    parser = argparse.ArgumentParser(description="Evaluate multitype ROI from saved predictions")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predictions parquet")
    parser.add_argument("--year", type=int, default=2023, help="Year to evaluate")
    args = parser.parse_args()
    
    logger.info(f"📊 Multitype ROI Evaluation")
    logger.info(f"Predictions: {args.pred_path}")
    
    # Load predictions
    df = pd.read_parquet(args.pred_path)
    df['race_id'] = df['race_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year == args.year]
    logger.info(f"Loaded {len(df)} predictions for {args.year}")
    
    # Load payouts
    engine = get_db_engine()
    df_pay = load_payout_data(engine, [args.year])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Loaded {len(payout_dict)} payout records")
    
    # Simulate all bet types
    bet_types = ['win', 'place', 'place_top3', 'wide_box', 'quinella']
    
    print("\n" + "="*70)
    print(f" Bet Type ROI Summary - {args.pred_path.split('/')[-1]}")
    print("="*70)
    print(f"{'Type':<15} {'Bets':>8} {'Hits':>8} {'Hit%':>8} {'ROI':>10}")
    print("-"*70)
    
    for bt in bet_types:
        result = simulate_bets(df, payout_dict, bt)
        print(f"{result['type']:<15} {result['bets']:>8} {result['hits']:>8} {result['hit_rate']:>7.1f}% {result['roi']:>9.1f}%")
    
    print("="*70)

if __name__ == "__main__":
    main()

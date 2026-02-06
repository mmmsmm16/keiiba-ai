"""
Grid Search for EV + Odds Filter to find optimal betting rules.
Usage: python scripts/run_ev_gridsearch.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from itertools import product

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
        haraimodoshi_fukusho_3a as place_3_horse, haraimodoshi_fukusho_3b as place_3_pay
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
        pay = {'win': {}, 'place': {}}
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
        payout_dict[rid] = pay
    return payout_dict

def simulate_ev_filter(df_data, payout_dict, bet_type, prob_th, ev_th, min_odds, max_odds):
    """Simulate with Probability, EV and Odds filter."""
    cost = 0
    returns = 0
    hits = 0
    bets = 0
    
    for rid, group in df_data.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        
        # For each horse, check prob, EV and odds conditions
        for _, row in group.iterrows():
            prob = row['pred_prob']
            odds = row['odds_final']
            horse = row['horse_number']
            
            if odds <= 0: continue
            
            ev = prob * odds  # Expected Value
            
            # Check conditions: prob >= prob_th AND EV >= ev_th AND odds in range
            if prob >= prob_th and ev >= ev_th and min_odds <= odds <= max_odds:
                cost += 100
                bets += 1
                
                if bet_type == 'win':
                    if horse in pay['win']:
                        returns += pay['win'][horse]
                        hits += 1
                elif bet_type == 'place':
                    if horse in pay['place']:
                        returns += pay['place'][horse]
                        hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    hit_rate = (hits / bets * 100) if bets > 0 else 0
    return {'bets': bets, 'hits': hits, 'hit_rate': hit_rate, 'roi': roi}

def main():
    logger.info("🔍 EV + Probability + Odds Grid Search")
    
    # Load predictions
    pred_dir = "models/experiments/exp_t2_refined_v3/cv_results"
    models = {
        'Binary': f"{pred_dir}/preds_binary.parquet",
        'Top3': f"{pred_dir}/preds_top3.parquet"
    }
    
    # Load payouts
    engine = get_db_engine()
    df_pay = load_payout_data(engine, [2023])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Loaded {len(payout_dict)} payout records")
    
    # Grid Search Parameters
    prob_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]  # 5%-25%
    ev_thresholds = [1.0, 1.2, 1.5, 2.0]
    min_odds_list = [2.0, 3.0, 5.0]
    max_odds_list = [20, 50]
    bet_types = ['win', 'place']
    
    best_results = {}
    
    for model_name, pred_path in models.items():
        df = pd.read_parquet(pred_path)
        df['race_id'] = df['race_id'].astype(str)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year == 2023]
        logger.info(f"\n{'='*80}")
        logger.info(f" {model_name} Model - Grid Search Results")
        logger.info(f"{'='*80}")
        
        for bet_type in bet_types:
            logger.info(f"\n📊 {bet_type.upper()} Bet Type")
            logger.info(f"{'Prob%':<8} {'EV_th':<8} {'MinOdds':<8} {'MaxOdds':<8} {'Bets':>6} {'Hit%':>8} {'ROI':>10}")
            logger.info("-"*70)
            
            best_roi = 0
            best_config = None
            
            for prob_th, ev_th, min_odds, max_odds in product(prob_thresholds, ev_thresholds, min_odds_list, max_odds_list):
                result = simulate_ev_filter(df, payout_dict, bet_type, prob_th, ev_th, min_odds, max_odds)
                
                # Only print if we have some bets
                if result['bets'] >= 50:
                    logger.info(f"{prob_th*100:<8.0f} {ev_th:<8.1f} {min_odds:<8.1f} {max_odds:<8} {result['bets']:>6} {result['hit_rate']:>7.1f}% {result['roi']:>9.1f}%")
                    
                    if result['roi'] > best_roi and result['bets'] >= 100:
                        best_roi = result['roi']
                        best_config = {
                            'prob_th': prob_th, 'ev_th': ev_th, 'min_odds': min_odds, 'max_odds': max_odds,
                            **result
                        }
            
            if best_config:
                key = f"{model_name}_{bet_type}"
                best_results[key] = best_config
                logger.info(f"\n🏆 Best: Prob>={best_config['prob_th']*100:.0f}%, EV>={best_config['ev_th']}, Odds[{best_config['min_odds']}-{best_config['max_odds']}] → ROI={best_config['roi']:.1f}%, Bets={best_config['bets']}")
    
    # Summary
    print("\n" + "="*80)
    print(" Best Configurations Summary")
    print("="*80)
    for key, cfg in best_results.items():
        print(f"{key:<20}: Prob>={cfg['prob_th']*100:.0f}%, EV>={cfg['ev_th']:.1f}, Odds[{cfg['min_odds']:.0f}-{cfg['max_odds']}] → ROI={cfg['roi']:.1f}% ({cfg['bets']} bets, {cfg['hit_rate']:.1f}% hit)")

if __name__ == "__main__":
    main()

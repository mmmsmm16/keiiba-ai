"""
Extended Grid Search for Prob + EV + Odds Filter.
Covers: Binary, Top2, Top3, Blend models and Win, Place, Wide, Quinella bet types.
Usage: python scripts/run_ev_gridsearch_extended.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from itertools import product, combinations

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
        haraimodoshi_umaren_1a as quinella_horses, haraimodoshi_umaren_1b as quinella_pay,
        haraimodoshi_umatan_1a as exacta_1st, haraimodoshi_umatan_1b as exacta_2nd, haraimodoshi_umatan_1c as exacta_pay
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
        pay = {'win': {}, 'place': {}, 'wide': {}, 'quinella': {}, 'exacta': {}, 'trio': {}, 'trifecta': {}}
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
                if p > 0: pay['wide'][(min(h1,h2), max(h1,h2))] = p
            except: pass
        try:
            horses_str = str(row['quinella_horses']).zfill(4)
            h1, h2 = int(horses_str[:2]), int(horses_str[2:])
            p = int(row['quinella_pay'])
            if p > 0: pay['quinella'][(min(h1,h2), max(h1,h2))] = p
        except: pass
        # Exacta (馬単): ordered 1st-2nd
        try:
            h1 = int(row['exacta_1st'])
            h2 = int(row['exacta_2nd'])
            p = int(row['exacta_pay'])
            if p > 0: pay['exacta'][(h1, h2)] = p
        except: pass
        # Trio (三連複): unordered top 3
        try:
            horses_str = str(row['trio_horses']).zfill(6)
            h1, h2, h3 = int(horses_str[:2]), int(horses_str[2:4]), int(horses_str[4:6])
            p = int(row['trio_pay'])
            if p > 0: pay['trio'][tuple(sorted([h1,h2,h3]))] = p
        except: pass
        # Trifecta (三連単): ordered 1st-2nd-3rd
        try:
            h1 = int(row['trifecta_1st'])
            h2 = int(row['trifecta_2nd'])
            h3 = int(row['trifecta_3rd'])
            p = int(row['trifecta_pay'])
            if p > 0: pay['trifecta'][(h1, h2, h3)] = p
        except: pass
        payout_dict[rid] = pay
    return payout_dict

def simulate(df_data, payout_dict, bet_type, prob_th, ev_th, min_odds, max_odds):
    """Simulate with Prob + EV + Odds filter for any bet type."""
    cost = 0
    returns = 0
    hits = 0
    bets = 0
    
    from itertools import permutations
    
    for rid, group in df_data.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        group = group.sort_values('pred_prob', ascending=False)
        
        if bet_type in ['win', 'place']:
            for _, row in group.iterrows():
                prob = row['pred_prob']
                odds = row['odds_final']
                horse = row['horse_number']
                if odds <= 0: continue
                ev = prob * odds
                if prob >= prob_th and ev >= ev_th and min_odds <= odds <= max_odds:
                    cost += 100
                    bets += 1
                    if bet_type == 'win' and horse in pay['win']:
                        returns += pay['win'][horse]
                        hits += 1
                    elif bet_type == 'place' and horse in pay['place']:
                        returns += pay['place'][horse]
                        hits += 1
        
        elif bet_type == 'wide_box':
            top_horses = group[group['pred_prob'] >= prob_th].head(3)
            if len(top_horses) >= 2:
                top_prob = top_horses['pred_prob'].mean()
                avg_odds = top_horses['odds_final'].mean()
                ev = top_prob * avg_odds
                if ev >= ev_th / 2 and min_odds <= avg_odds <= max_odds:
                    horses = top_horses['horse_number'].tolist()
                    for h1, h2 in combinations(horses, 2):
                        cost += 100
                        bets += 1
                        key = (min(h1, h2), max(h1, h2))
                        if key in pay['wide']:
                            returns += pay['wide'][key]
                            hits += 1
        
        elif bet_type == 'quinella':
            top_horses = group[group['pred_prob'] >= prob_th].head(2)
            if len(top_horses) >= 2:
                top_prob = top_horses['pred_prob'].mean()
                avg_odds = top_horses['odds_final'].mean()
                ev = top_prob * avg_odds
                if ev >= ev_th / 2 and min_odds <= avg_odds <= max_odds:
                    horses = top_horses['horse_number'].tolist()
                    h1, h2 = horses[0], horses[1]
                    cost += 100
                    bets += 1
                    key = (min(h1, h2), max(h1, h2))
                    if key in pay['quinella']:
                        returns += pay['quinella'][key]
                        hits += 1
        
        elif bet_type == 'exacta':
            # Top 2 horses ordered (1st-2nd)
            top_horses = group[group['pred_prob'] >= prob_th].head(2)
            if len(top_horses) >= 2:
                top_prob = top_horses['pred_prob'].mean()
                avg_odds = top_horses['odds_final'].mean()
                ev = top_prob * avg_odds
                if ev >= ev_th / 2 and min_odds <= avg_odds <= max_odds:
                    horses = top_horses['horse_number'].tolist()
                    cost += 100
                    bets += 1
                    key = (horses[0], horses[1])
                    if key in pay['exacta']:
                        returns += pay['exacta'][key]
                        hits += 1
        
        elif bet_type == 'trio':
            # Top 3 horses unordered
            top_horses = group[group['pred_prob'] >= prob_th].head(3)
            if len(top_horses) >= 3:
                top_prob = top_horses['pred_prob'].mean()
                avg_odds = top_horses['odds_final'].mean()
                ev = top_prob * avg_odds
                if ev >= ev_th / 3 and min_odds <= avg_odds <= max_odds:
                    horses = top_horses['horse_number'].tolist()
                    cost += 100
                    bets += 1
                    key = tuple(sorted(horses))
                    if key in pay['trio']:
                        returns += pay['trio'][key]
                        hits += 1
        
        elif bet_type == 'trifecta':
            # Top 3 horses ordered (1st-2nd-3rd)
            top_horses = group[group['pred_prob'] >= prob_th].head(3)
            if len(top_horses) >= 3:
                top_prob = top_horses['pred_prob'].mean()
                avg_odds = top_horses['odds_final'].mean()
                ev = top_prob * avg_odds
                if ev >= ev_th / 3 and min_odds <= avg_odds <= max_odds:
                    horses = top_horses['horse_number'].tolist()
                    cost += 100
                    bets += 1
                    key = (horses[0], horses[1], horses[2])
                    if key in pay['trifecta']:
                        returns += pay['trifecta'][key]
                        hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    hit_rate = (hits / bets * 100) if bets > 0 else 0
    return {'bets': bets, 'hits': hits, 'hit_rate': hit_rate, 'roi': roi}

def main():
    logger.info("🔍 Extended EV + Probability + Odds Grid Search (All Bet Types)")
    
    pred_dir = "models/experiments/exp_t2_refined_v3/cv_results"
    
    df_binary = pd.read_parquet(f"{pred_dir}/preds_binary.parquet")
    df_top2 = pd.read_parquet(f"{pred_dir}/preds_top2.parquet")
    df_top3 = pd.read_parquet(f"{pred_dir}/preds_top3.parquet")
    
    for df in [df_binary, df_top2, df_top3]:
        df['race_id'] = df['race_id'].astype(str)
        df['date'] = pd.to_datetime(df['date'])
    
    df_blend = df_binary[['race_id', 'horse_number', 'date', 'rank', 'odds_final']].copy()
    df_blend['pred_prob'] = 0.4 * df_binary['pred_prob'] + 0.6 * df_top3['pred_prob']
    
    models = {
        'Binary': df_binary[df_binary['date'].dt.year == 2023],
        'Top2': df_top2[df_top2['date'].dt.year == 2023],
        'Top3': df_top3[df_top3['date'].dt.year == 2023],
        'Blend': df_blend[df_blend['date'].dt.year == 2023]
    }
    
    engine = get_db_engine()
    df_pay = load_payout_data(engine, [2023])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Loaded {len(payout_dict)} payout records")
    
    # Grid Search Parameters
    prob_thresholds = [0.10, 0.15, 0.20, 0.25]
    ev_thresholds = [1.0, 1.5, 2.0]
    min_odds_list = [2.0, 5.0]
    max_odds_list = [20, 50]
    bet_types = ['win', 'place', 'wide_box', 'quinella', 'exacta']
    
    all_results = []
    
    for model_name, df in models.items():
        logger.info(f"\n{'='*80}")
        logger.info(f" {model_name} Model")
        logger.info(f"{'='*80}")
        
        for bet_type in bet_types:
            best_roi = 0
            best_config = None
            
            for prob_th, ev_th, min_odds, max_odds in product(prob_thresholds, ev_thresholds, min_odds_list, max_odds_list):
                result = simulate(df, payout_dict, bet_type, prob_th, ev_th, min_odds, max_odds)
                
                if result['bets'] >= 50 and result['roi'] > best_roi:
                    best_roi = result['roi']
                    best_config = {
                        'model': model_name, 'bet_type': bet_type,
                        'prob_th': prob_th, 'ev_th': ev_th, 
                        'min_odds': min_odds, 'max_odds': max_odds,
                        **result
                    }
            
            if best_config:
                all_results.append(best_config)
                logger.info(f"  {bet_type:<12}: Prob>={best_config['prob_th']*100:.0f}%, EV>={best_config['ev_th']:.1f}, Odds[{best_config['min_odds']:.0f}-{best_config['max_odds']}] → ROI={best_config['roi']:.1f}% ({best_config['bets']} bets)")
    
    # Final Summary
    print("\n" + "="*95)
    print(" FINAL SUMMARY - Best ROI per Model + Bet Type")
    print("="*95)
    print(f"{'Model':<10} {'BetType':<12} {'Prob%':>6} {'EV':>5} {'Odds':>10} {'Bets':>6} {'Hit%':>7} {'ROI':>8}")
    print("-"*95)
    for r in sorted(all_results, key=lambda x: -x['roi']):
        print(f"{r['model']:<10} {r['bet_type']:<12} {r['prob_th']*100:>5.0f}% {r['ev_th']:>5.1f} {r['min_odds']:.0f}-{r['max_odds']:<5} {r['bets']:>6} {r['hit_rate']:>6.1f}% {r['roi']:>7.1f}%")

if __name__ == "__main__":
    main()


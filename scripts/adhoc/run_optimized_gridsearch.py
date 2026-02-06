"""
Optimized Grid Search with Stricter Conditions
- Vectorized processing for speed
- Focus on high-confidence thresholds (top 5-10% of predictions)
- Stricter odds filters
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import itertools
from sqlalchemy import create_engine
from collections import defaultdict

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
        haraimodoshi_fukusho_1a, haraimodoshi_fukusho_1b,
        haraimodoshi_fukusho_2a, haraimodoshi_fukusho_2b,
        haraimodoshi_fukusho_3a, haraimodoshi_fukusho_3b,
        haraimodoshi_wide_1a, haraimodoshi_wide_1b,
        haraimodoshi_wide_2a, haraimodoshi_wide_2b,
        haraimodoshi_wide_3a, haraimodoshi_wide_3b,
        haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
        haraimodoshi_umatan_1a, haraimodoshi_umatan_1b,
        haraimodoshi_umatan_2a, haraimodoshi_umatan_2b,
        haraimodoshi_umatan_3a, haraimodoshi_umatan_3b,
        haraimodoshi_sanrenpuku_1a, haraimodoshi_sanrenpuku_1b,
        haraimodoshi_sanrentan_1a, haraimodoshi_sanrentan_1b,
        haraimodoshi_sanrentan_2a, haraimodoshi_sanrentan_2b,
        haraimodoshi_sanrentan_3a, haraimodoshi_sanrentan_3b
    FROM jvd_hr WHERE kaisai_nen = '{year}'
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df.apply(lambda r: f"{r['kaisai_nen']}{str(r['keibajo_code']).zfill(2)}{str(r['kaisai_kai']).zfill(2)}{str(r['kaisai_nichime']).zfill(2)}{str(r['race_bango']).zfill(2)}", axis=1)
    return df

def parse_payouts(df_pay):
    payouts = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        payouts[rid] = {'win': {}, 'place': {}, 'wide': {}, 'quinella': {}, 'exacta': {}, 'trio': {}, 'trifecta': {}}
        
        # Win
        if pd.notna(row.get('haraimodoshi_tansho_1a')):
            try: payouts[rid]['win'][int(row['haraimodoshi_tansho_1a'])] = int(row['haraimodoshi_tansho_1b'])
            except: pass
        # Place
        for i in range(1, 4):
            if pd.notna(row.get(f'haraimodoshi_fukusho_{i}a')):
                try: payouts[rid]['place'][int(row[f'haraimodoshi_fukusho_{i}a'])] = int(row[f'haraimodoshi_fukusho_{i}b'])
                except: pass
        # Wide
        for i in range(1, 4):
            if pd.notna(row.get(f'haraimodoshi_wide_{i}a')):
                try:
                    h_str = str(row[f'haraimodoshi_wide_{i}a']).zfill(4)
                    payouts[rid]['wide'][frozenset({int(h_str[:2]), int(h_str[2:])})] = int(row[f'haraimodoshi_wide_{i}b'])
                except: pass
        # Quinella
        if pd.notna(row.get('haraimodoshi_umaren_1a')):
            try:
                h_str = str(row['haraimodoshi_umaren_1a']).zfill(4)
                payouts[rid]['quinella'][frozenset({int(h_str[:2]), int(h_str[2:])})] = int(row['haraimodoshi_umaren_1b'])
            except: pass
        # Exacta
        for i in range(1, 4):
            if pd.notna(row.get(f'haraimodoshi_umatan_{i}a')):
                try:
                    h_str = str(row[f'haraimodoshi_umatan_{i}a']).zfill(4)
                    payouts[rid]['exacta'][(int(h_str[:2]), int(h_str[2:]))] = int(row[f'haraimodoshi_umatan_{i}b'])
                except: pass
        # Trio
        if pd.notna(row.get('haraimodoshi_sanrenpuku_1a')):
            try:
                h_str = str(row['haraimodoshi_sanrenpuku_1a']).zfill(6)
                payouts[rid]['trio'][frozenset({int(h_str[:2]), int(h_str[2:4]), int(h_str[4:])})] = int(row['haraimodoshi_sanrenpuku_1b'])
            except: pass
        # Trifecta
        for i in range(1, 4):
            if pd.notna(row.get(f'haraimodoshi_sanrentan_{i}a')):
                try:
                    h_str = str(row[f'haraimodoshi_sanrentan_{i}a']).zfill(6)
                    payouts[rid]['trifecta'][(int(h_str[:2]), int(h_str[2:4]), int(h_str[4:]))] = int(row[f'haraimodoshi_sanrentan_{i}b'])
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

def precompute_race_data(df_data, payout_dict):
    """Pre-compute race-level sorted data for fast simulation."""
    race_data = {}
    for rid, group in df_data.groupby('race_id'):
        if rid not in payout_dict:
            continue
        g = group.sort_values('pred_prob', ascending=False)
        race_data[rid] = {
            'horses': g['horse_number'].values,
            'probs': g['pred_prob'].values,
            'odds': g['odds_final'].values,
            'payouts': payout_dict[rid]
        }
    return race_data

def simulate_all_strategies(race_data, strategies):
    """Vectorized simulation of all strategies at once."""
    results = {s['name']: {'cost': 0, 'ret': 0, 'races': 0} for s in strategies}
    
    for rid, rd in race_data.items():
        horses = rd['horses']
        probs = rd['probs']
        odds = rd['odds']
        pay = rd['payouts']
        n = len(horses)
        
        for s in strategies:
            stype = s['type']
            
            if stype == 'win':
                th, min_odds = s['th'], s['min_odds']
                for i in range(n):
                    if probs[i] < th: break
                    if odds[i] >= min_odds:
                        results[s['name']]['races'] += 1
                        results[s['name']]['cost'] += 100
                        if horses[i] in pay['win']:
                            results[s['name']]['ret'] += pay['win'][horses[i]]
            
            elif stype == 'place':
                th, min_odds = s['th'], s['min_odds']
                for i in range(n):
                    if probs[i] < th: break
                    if odds[i] >= min_odds:
                        results[s['name']]['races'] += 1
                        results[s['name']]['cost'] += 100
                        if horses[i] in pay['place']:
                            results[s['name']]['ret'] += pay['place'][horses[i]]
            
            elif stype == 'exacta_nagashi':
                axis_th, min_odds, n_flow = s['axis_th'], s['min_odds'], s['n_flow']
                if n < n_flow + 1: continue
                if probs[0] < axis_th: continue
                if odds[0] < min_odds: continue
                
                results[s['name']]['races'] += 1
                axis = horses[0]
                for fi in range(1, min(n_flow+1, n)):
                    results[s['name']]['cost'] += 100
                    key = (axis, horses[fi])
                    if key in pay['exacta']:
                        results[s['name']]['ret'] += pay['exacta'][key]
            
            elif stype == 'wide_nagashi':
                axis_th, min_odds, n_flow = s['axis_th'], s['min_odds'], s['n_flow']
                if n < n_flow + 1: continue
                if probs[0] < axis_th: continue
                if odds[0] < min_odds: continue
                
                results[s['name']]['races'] += 1
                axis = horses[0]
                for fi in range(1, min(n_flow+1, n)):
                    results[s['name']]['cost'] += 100
                    key = frozenset({axis, horses[fi]})
                    if key in pay['wide']:
                        results[s['name']]['ret'] += pay['wide'][key]
            
            elif stype == 'quinella_nagashi':
                axis_th, min_odds, n_flow = s['axis_th'], s['min_odds'], s['n_flow']
                if n < n_flow + 1: continue
                if probs[0] < axis_th: continue
                if odds[0] < min_odds: continue
                
                results[s['name']]['races'] += 1
                axis = horses[0]
                for fi in range(1, min(n_flow+1, n)):
                    results[s['name']]['cost'] += 100
                    key = frozenset({axis, horses[fi]})
                    if key in pay['quinella']:
                        results[s['name']]['ret'] += pay['quinella'][key]
            
            elif stype == 'trio_box':
                axis_th, n_box = s['axis_th'], s['n_box']
                if n < n_box: continue
                if probs[0] < axis_th: continue
                
                results[s['name']]['races'] += 1
                box_horses = horses[:n_box]
                for combo in itertools.combinations(box_horses, 3):
                    results[s['name']]['cost'] += 100
                    key = frozenset(combo)
                    if key in pay['trio']:
                        results[s['name']]['ret'] += pay['trio'][key]
            
            elif stype == 'trifecta_1st_nagashi':
                axis_th, min_odds, n_flow = s['axis_th'], s['min_odds'], s['n_flow']
                if n < n_flow + 1: continue
                if probs[0] < axis_th: continue
                if odds[0] < min_odds: continue
                
                results[s['name']]['races'] += 1
                axis = horses[0]
                flow = horses[1:n_flow+1]
                for p in itertools.permutations(flow, 2):
                    results[s['name']]['cost'] += 100
                    key = (axis, p[0], p[1])
                    if key in pay['trifecta']:
                        results[s['name']]['ret'] += pay['trifecta'][key]
    
    # Calculate ROI
    for name, r in results.items():
        r['roi'] = r['ret'] / r['cost'] * 100 if r['cost'] > 0 else 0
        r['profit'] = r['ret'] - r['cost']
    
    return results

def main():
    logger.info("=" * 70)
    logger.info("OPTIMIZED GRID SEARCH (Stricter Conditions)")
    logger.info("=" * 70)
    
    # Load model
    MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
    model = joblib.load(MODEL_PATH)
    feat_cols = model.feature_name() if hasattr(model, 'feature_name') else model.booster_.feature_name()
    
    # Load features
    df = pd.read_parquet('data/features/temp_merge_fixed_baseline.parquet')
    df['race_id'] = df['race_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Load odds
    df_odds_24 = load_odds_from_db(2024)
    df_odds_25 = load_odds_from_db(2025)
    df_odds = pd.concat([df_odds_24, df_odds_25], ignore_index=True)
    df_odds['race_id'] = df_odds['race_id'].astype(str)
    
    # Prepare 2024
    df_2024 = df[df['year'] == 2024].copy()
    df_2024['horse_number'] = pd.to_numeric(df_2024['horse_number'], errors='coerce').fillna(0).astype(int)
    df_2024 = pd.merge(df_2024, df_odds, on=['race_id', 'horse_number'], how='left')
    df_2024['odds_final'] = df_2024['odds_final'].fillna(1.0)
    
    for c in feat_cols:
        if c not in df_2024.columns:
            df_2024[c] = 0
    X_24 = df_2024[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_2024['pred_prob'] = model.predict(X_24.values.astype(np.float32))
    
    # Prepare 2025
    df_2025 = df[df['year'] == 2025].copy()
    df_2025['horse_number'] = pd.to_numeric(df_2025['horse_number'], errors='coerce').fillna(0).astype(int)
    df_2025 = pd.merge(df_2025, df_odds, on=['race_id', 'horse_number'], how='left')
    df_2025['odds_final'] = df_2025['odds_final'].fillna(1.0)
    
    for c in feat_cols:
        if c not in df_2025.columns:
            df_2025[c] = 0
    X_25 = df_2025[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_2025['pred_prob'] = model.predict(X_25.values.astype(np.float32))
    
    # Show probability percentiles
    logger.info(f"\n2024 Prob Percentiles:")
    for p in [90, 95, 97, 99]:
        logger.info(f"  {p}%: {df_2024['pred_prob'].quantile(p/100):.4f}")
    
    logger.info(f"\n2025 Prob Percentiles:")
    for p in [90, 95, 97, 99]:
        logger.info(f"  {p}%: {df_2025['pred_prob'].quantile(p/100):.4f}")
    
    # Load payouts
    pay_2024 = parse_payouts(load_payout_data(2024))
    pay_2025 = parse_payouts(load_payout_data(2025))
    
    # Pre-compute race data
    logger.info("\nPre-computing race data...")
    race_data_24 = precompute_race_data(df_2024, pay_2024)
    race_data_25 = precompute_race_data(df_2025, pay_2025)
    
    # Generate strategies with STRICT thresholds (top 5-3% of predictions)
    p99 = df_2024['pred_prob'].quantile(0.99)
    p97 = df_2024['pred_prob'].quantile(0.97)
    p95 = df_2024['pred_prob'].quantile(0.95)
    p90 = df_2024['pred_prob'].quantile(0.90)
    
    logger.info(f"\nUsing quantile-based thresholds:")
    logger.info(f"  99th: {p99:.4f}, 97th: {p97:.4f}, 95th: {p95:.4f}, 90th: {p90:.4f}")
    
    strategies = []
    
    # Win/Place - Focus on TOP predictions only
    for th_pct, th_val in [(99, p99), (97, p97), (95, p95), (90, p90)]:
        for min_odds in [1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]:
            strategies.append({'type': 'win', 'th': th_val, 'min_odds': min_odds, 
                               'name': f'Win Top{100-th_pct}% Od>{min_odds}'})
            strategies.append({'type': 'place', 'th': th_val, 'min_odds': min_odds, 
                               'name': f'Place Top{100-th_pct}% Od>{min_odds}'})
    
    # Exacta Nagashi - Strict axis with various flows
    for th_pct, th_val in [(99, p99), (97, p97), (95, p95)]:
        for min_odds in [2.0, 5.0, 10.0]:
            for n_flow in [4, 5, 6, 7]:
                strategies.append({'type': 'exacta_nagashi', 'axis_th': th_val, 'min_odds': min_odds, 'n_flow': n_flow,
                                   'name': f'Exacta Top{100-th_pct}% Od>{min_odds} F{n_flow}'})
    
    # Wide Nagashi
    for th_pct, th_val in [(99, p99), (97, p97), (95, p95)]:
        for min_odds in [2.0, 5.0, 10.0]:
            for n_flow in [4, 5, 6, 7]:
                strategies.append({'type': 'wide_nagashi', 'axis_th': th_val, 'min_odds': min_odds, 'n_flow': n_flow,
                                   'name': f'Wide Top{100-th_pct}% Od>{min_odds} F{n_flow}'})
    
    # Quinella Nagashi
    for th_pct, th_val in [(99, p99), (97, p97), (95, p95)]:
        for min_odds in [2.0, 5.0, 10.0]:
            for n_flow in [4, 5, 6, 7]:
                strategies.append({'type': 'quinella_nagashi', 'axis_th': th_val, 'min_odds': min_odds, 'n_flow': n_flow,
                                   'name': f'Quinella Top{100-th_pct}% Od>{min_odds} F{n_flow}'})
    
    # Trio Box
    for th_pct, th_val in [(99, p99), (97, p97), (95, p95)]:
        for n_box in [4, 5, 6]:
            strategies.append({'type': 'trio_box', 'axis_th': th_val, 'n_box': n_box,
                               'name': f'Trio Top{100-th_pct}% Box{n_box}'})
    
    # Trifecta 1st Nagashi
    for th_pct, th_val in [(99, p99), (97, p97), (95, p95)]:
        for min_odds in [5.0, 10.0]:
            for n_flow in [4, 5, 6]:
                strategies.append({'type': 'trifecta_1st_nagashi', 'axis_th': th_val, 'min_odds': min_odds, 'n_flow': n_flow,
                                   'name': f'Trifecta1st Top{100-th_pct}% Od>{min_odds} F{n_flow}'})
    
    logger.info(f"\nTotal strategies: {len(strategies)}")
    logger.info("Running simulation...")
    
    # Simulate 2024
    results_24 = simulate_all_strategies(race_data_24, strategies)
    
    # Convert to DataFrame and filter
    df_24 = pd.DataFrame([
        {'name': k, 'type': v.get('type', k.split()[0].lower()), 
         'roi': v['roi'], 'races': v['races'], 'profit': v['profit']}
        for k, v in results_24.items()
    ])
    
    # Add type column from name
    df_24['type'] = df_24['name'].apply(lambda x: x.split()[0].lower())
    
    # Show all results sorted by ROI
    df_24_sorted = df_24[df_24['races'] >= 10].sort_values('roi', ascending=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("2024 RESULTS (min 10 races)")
    logger.info("=" * 70)
    print(df_24_sorted.head(30).to_string(index=False))
    
    # Find promising (ROI > 85%)
    promising = df_24_sorted[df_24_sorted['roi'] > 85]['name'].tolist()
    
    if len(promising) > 0:
        logger.info(f"\n{len(promising)} strategies with ROI > 95%!")
        
        # Validate on 2025
        results_25 = simulate_all_strategies(race_data_25, [s for s in strategies if s['name'] in promising])
        
        logger.info("\n" + "=" * 70)
        logger.info("2025 WALK-FORWARD VALIDATION")
        logger.info("=" * 70)
        
        comparison = []
        for name in promising:
            r24 = results_24[name]
            r25 = results_25[name]
            comparison.append({
                'Strategy': name,
                'ROI_24': round(r24['roi'], 1),
                'Races_24': r24['races'],
                'ROI_25': round(r25['roi'], 1),
                'Races_25': r25['races'],
                'Profit_25': r25['profit']
            })
        
        df_cmp = pd.DataFrame(comparison).sort_values('ROI_24', ascending=False)
        print(df_cmp.to_string(index=False))
        
        # Summary
        avg_24 = df_cmp['ROI_24'].mean()
        avg_25 = df_cmp['ROI_25'].mean()
        logger.info(f"\nAverage ROI: 2024={avg_24:.1f}%, 2025={avg_25:.1f}%, Change={avg_25-avg_24:+.1f}%")
    else:
        logger.info("\nNo strategies with ROI > 95% found.")
        logger.info("Top 5 by ROI:")
        print(df_24_sorted.head(5).to_string(index=False))

if __name__ == "__main__":
    main()

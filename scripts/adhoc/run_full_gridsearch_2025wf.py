"""
Full Bet Type Grid Search + 2025 Walk-Forward Validation
All bet types: Win, Place, Wide, Quinella, Exacta, Trifecta
With Fixed Baseline Features
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
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

def simulate(df_data, payout_dict, strategy):
    """Unified simulation function for all bet types."""
    cost, ret, races_bet = 0, 0, 0
    stype = strategy['type']
    
    for rid, group in df_data.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        
        group = group.sort_values('pred_prob', ascending=False)
        scores = group['pred_prob'].values
        horses = group['horse_number'].values
        odds = group['odds_final'].values
        
        n_horses = len(horses)
        
        # === SINGLE BETS ===
        if stype == 'win':
            th = strategy['th']
            min_odds = strategy['min_odds']
            for i in range(n_horses):
                if scores[i] < th: break
                if odds[i] >= min_odds:
                    races_bet += 1
                    cost += 100
                    if horses[i] in pay['win']:
                        ret += pay['win'][horses[i]]
        
        elif stype == 'place':
            th = strategy['th']
            min_odds = strategy['min_odds']
            for i in range(n_horses):
                if scores[i] < th: break
                if odds[i] >= min_odds:
                    races_bet += 1
                    cost += 100
                    if horses[i] in pay['place']:
                        ret += pay['place'][horses[i]]
        
        # === NAGASHI (WHEEL) BETS ===
        elif stype in ['wide_nagashi', 'quinella_nagashi', 'exacta_nagashi']:
            axis_th = strategy['axis_th']
            min_odds = strategy['min_odds']
            n_flow = strategy['n_flow']
            
            if scores[0] < axis_th: continue
            if odds[0] < min_odds: continue
            if n_horses < n_flow + 1: continue
            
            races_bet += 1
            axis = horses[0]
            flow = horses[1:n_flow+1]
            
            bet_type = stype.replace('_nagashi', '')
            for f in flow:
                cost += 100
                if bet_type == 'wide':
                    key = frozenset({axis, f})
                    if key in pay['wide']:
                        ret += pay['wide'][key]
                elif bet_type == 'quinella':
                    key = frozenset({axis, f})
                    if key in pay['quinella']:
                        ret += pay['quinella'][key]
                elif bet_type == 'exacta':
                    key = (axis, f)
                    if key in pay['exacta']:
                        ret += pay['exacta'][key]
        
        # === TRIFECTA 1st NAGASHI ===
        elif stype == 'trifecta_1st_nagashi':
            axis_th = strategy['axis_th']
            min_odds = strategy['min_odds']
            n_flow = strategy['n_flow']
            
            if scores[0] < axis_th: continue
            if odds[0] < min_odds: continue
            if n_horses < n_flow + 1: continue
            
            races_bet += 1
            axis = horses[0]
            flow = horses[1:n_flow+1]
            
            for p in itertools.permutations(flow, 2):
                cost += 100
                key = (axis, p[0], p[1])
                if key in pay['trifecta']:
                    ret += pay['trifecta'][key]
        
        # === TRIO BOX ===
        elif stype == 'trio_box':
            n_box = strategy['n_box']
            axis_th = strategy['axis_th']
            
            if scores[0] < axis_th: continue
            if n_horses < n_box: continue
            
            races_bet += 1
            box_horses = horses[:n_box]
            
            for combo in itertools.combinations(box_horses, 3):
                cost += 100
                key = frozenset(combo)
                if key in pay['trio']:
                    ret += pay['trio'][key]
    
    roi = ret / cost * 100 if cost > 0 else 0
    return {'cost': cost, 'return': ret, 'roi': roi, 'races': races_bet, 'profit': ret - cost}

def main():
    logger.info("=" * 70)
    logger.info("FULL BET TYPE GRID SEARCH + 2025 WALK-FORWARD VALIDATION")
    logger.info("=" * 70)
    
    # Load 2024 model
    MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
    model = joblib.load(MODEL_PATH)
    
    # Get feature names
    feat_cols = model.feature_name() if hasattr(model, 'feature_name') else model.booster_.feature_name()
    
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
    
    # Prepare 2025 data
    df_2025 = df[df['year'] == 2025].copy()
    df_2025['horse_number'] = pd.to_numeric(df_2025['horse_number'], errors='coerce').fillna(0).astype(int)
    df_2025 = pd.merge(df_2025, df_odds, on=['race_id', 'horse_number'], how='left')
    df_2025['odds_final'] = df_2025['odds_final'].fillna(1.0)
    
    # Predict 2025
    for c in feat_cols:
        if c not in df_2025.columns:
            df_2025[c] = 0
    X_25 = df_2025[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_2025['pred_prob'] = model.predict(X_25.values.astype(np.float32))
    
    logger.info(f"2024: {len(df_2024)} rows, Prob 95%: {df_2024['pred_prob'].quantile(0.95):.4f}")
    logger.info(f"2025: {len(df_2025)} rows, Prob 95%: {df_2025['pred_prob'].quantile(0.95):.4f}")
    
    # Load payouts
    pay_2024 = parse_payouts(load_payout_data(2024))
    pay_2025 = parse_payouts(load_payout_data(2025))
    
    # ==== GRID SEARCH ====
    logger.info("\n" + "=" * 70)
    logger.info("GRID SEARCH: All Bet Types (2024)")
    logger.info("=" * 70)
    
    strategies = []
    
    # Win/Place - Extended range with higher thresholds
    for th in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
        for min_odds in [1.5, 2.0, 3.0, 5.0, 8.0, 10.0]:
            strategies.append({'type': 'win', 'th': th, 'min_odds': min_odds, 'name': f'Win C>{th} Od>{min_odds}'})
            strategies.append({'type': 'place', 'th': th, 'min_odds': min_odds, 'name': f'Place C>{th} Od>{min_odds}'})
    
    # Nagashi (Wide, Quinella, Exacta) - With more flow variations
    for axis_th in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
        for min_odds in [1.5, 2.0, 3.0, 5.0, 8.0]:
            for n_flow in [3, 4, 5, 6, 7, 8]:
                strategies.append({'type': 'wide_nagashi', 'axis_th': axis_th, 'min_odds': min_odds, 'n_flow': n_flow, 
                                   'name': f'Wide C>{axis_th} Od>{min_odds} F{n_flow}'})
                strategies.append({'type': 'quinella_nagashi', 'axis_th': axis_th, 'min_odds': min_odds, 'n_flow': n_flow, 
                                   'name': f'Quinella C>{axis_th} Od>{min_odds} F{n_flow}'})
                strategies.append({'type': 'exacta_nagashi', 'axis_th': axis_th, 'min_odds': min_odds, 'n_flow': n_flow, 
                                   'name': f'Exacta C>{axis_th} Od>{min_odds} F{n_flow}'})
    
    # Trifecta 1st Nagashi (high payout)
    for axis_th in [0.12, 0.14, 0.16, 0.18, 0.20]:
        for min_odds in [2.0, 3.0, 5.0, 8.0]:
            for n_flow in [3, 4, 5, 6]:
                strategies.append({'type': 'trifecta_1st_nagashi', 'axis_th': axis_th, 'min_odds': min_odds, 'n_flow': n_flow, 
                                   'name': f'Trifecta1st C>{axis_th} Od>{min_odds} F{n_flow}'})
    
    # Trio Box
    for axis_th in [0.10, 0.12, 0.14, 0.16, 0.18]:
        for n_box in [3, 4, 5, 6]:
            strategies.append({'type': 'trio_box', 'axis_th': axis_th, 'n_box': n_box, 
                               'name': f'TrioBox C>{axis_th} N{n_box}'})
    
    logger.info(f"Total strategies to test: {len(strategies)}")
    
    # Run 2024 simulation
    results_2024 = []
    for s in strategies:
        res = simulate(df_2024, pay_2024, s)
        if res['races'] > 50:  # Min bet threshold
            results_2024.append({
                'name': s['name'],
                'type': s['type'],
                'roi': res['roi'],
                'races': res['races'],
                'profit': res['profit']
            })
    
    df_24 = pd.DataFrame(results_2024)
    
    # Filter ROI > 90%
    promising = df_24[df_24['roi'] > 90].copy()
    logger.info(f"\nStrategies with ROI > 90% in 2024: {len(promising)}")
    
    if len(promising) == 0:
        # Show top 30 strategies anyway
        logger.info("\nTop 30 strategies by ROI (all):")
        print(df_24.sort_values('roi', ascending=False).head(30).to_string(index=False))
        return
    
    # ==== 2025 VALIDATION ====
    logger.info("\n" + "=" * 70)
    logger.info("2025 WALK-FORWARD VALIDATION (ROI > 95% strategies)")
    logger.info("=" * 70)
    
    promising_names = set(promising['name'].tolist())
    promising_strategies = [s for s in strategies if s['name'] in promising_names]
    
    results_25 = []
    for s in promising_strategies:
        res = simulate(df_2025, pay_2025, s)
        res_24 = simulate(df_2024, pay_2024, s)
        results_25.append({
            'Strategy': s['name'],
            'Type': s['type'],
            'ROI_24': round(res_24['roi'], 1),
            'Races_24': res_24['races'],
            'ROI_25': round(res['roi'], 1),
            'Races_25': res['races'],
            'Profit_25': res['profit']
        })
    
    df_res = pd.DataFrame(results_25).sort_values('ROI_24', ascending=False)
    
    print("\n" + "=" * 70)
    print("RESULTS: 2024 ROI > 95% Strategies → 2025 Validation")
    print("=" * 70)
    print(df_res.to_string(index=False))
    
    # Summary by type
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY BY BET TYPE")
    logger.info("=" * 70)
    
    for bet_type in df_res['Type'].unique():
        subset = df_res[df_res['Type'] == bet_type]
        avg_24 = subset['ROI_24'].mean()
        avg_25 = subset['ROI_25'].mean()
        count = len(subset)
        logger.info(f"{bet_type}: Count={count}, Avg ROI 2024={avg_24:.1f}%, Avg ROI 2025={avg_25:.1f}%, Change={avg_25-avg_24:+.1f}%")
    
    # Save results
    df_res.to_csv('reports/walkforward_validation_2025.csv', index=False)
    logger.info(f"\nResults saved to: reports/walkforward_validation_2025.csv")

if __name__ == "__main__":
    main()


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
    logger.info(f"Loading Payout Data (jvd_hr) for {year}...")
    engine = get_db_engine()
    
    query = f"""
    SELECT 
        kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
        -- Tansho
        haraimodoshi_tansho_1a, haraimodoshi_tansho_1b,
        -- Fukusho 
        haraimodoshi_fukusho_1a, haraimodoshi_fukusho_1b,
        haraimodoshi_fukusho_2a, haraimodoshi_fukusho_2b,
        haraimodoshi_fukusho_3a, haraimodoshi_fukusho_3b,
        haraimodoshi_fukusho_4a, haraimodoshi_fukusho_4b,
        haraimodoshi_fukusho_5a, haraimodoshi_fukusho_5b,
        -- Wide
        haraimodoshi_wide_1a, haraimodoshi_wide_1b,
        haraimodoshi_wide_2a, haraimodoshi_wide_2b,
        haraimodoshi_wide_3a, haraimodoshi_wide_3b,
        haraimodoshi_wide_4a, haraimodoshi_wide_4b,
        haraimodoshi_wide_5a, haraimodoshi_wide_5b,
        haraimodoshi_wide_6a, haraimodoshi_wide_6b,
        haraimodoshi_wide_7a, haraimodoshi_wide_7b,
        -- Umaren
        haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
        haraimodoshi_umaren_2a, haraimodoshi_umaren_2b,
        haraimodoshi_umaren_3a, haraimodoshi_umaren_3b,
        -- Umatan (Exacta)
        haraimodoshi_umatan_1a, haraimodoshi_umatan_1b,
        haraimodoshi_umatan_2a, haraimodoshi_umatan_2b,
        haraimodoshi_umatan_3a, haraimodoshi_umatan_3b,
        haraimodoshi_umatan_4a, haraimodoshi_umatan_4b,
        haraimodoshi_umatan_5a, haraimodoshi_umatan_5b,
        haraimodoshi_umatan_6a, haraimodoshi_umatan_6b,
        -- Trio
        haraimodoshi_sanrenpuku_1a, haraimodoshi_sanrenpuku_1b,
        haraimodoshi_sanrenpuku_2a, haraimodoshi_sanrenpuku_2b,
        haraimodoshi_sanrenpuku_3a, haraimodoshi_sanrenpuku_3b,
        -- Trifecta
        haraimodoshi_sanrentan_1a, haraimodoshi_sanrentan_1b,
        haraimodoshi_sanrentan_2a, haraimodoshi_sanrentan_2b,
        haraimodoshi_sanrentan_3a, haraimodoshi_sanrentan_3b,
        haraimodoshi_sanrentan_4a, haraimodoshi_sanrentan_4b,
        haraimodoshi_sanrentan_5a, haraimodoshi_sanrentan_5b,
        haraimodoshi_sanrentan_6a, haraimodoshi_sanrentan_6b
    FROM jvd_hr
    WHERE kaisai_nen = '{year}'
    """
    df = pd.read_sql(query, engine)
    
    def make_id(row):
        return f"{row['kaisai_nen']}{str(row['keibajo_code']).zfill(2)}{str(row['kaisai_kai']).zfill(2)}{str(row['kaisai_nichime']).zfill(2)}{str(row['race_bango']).zfill(2)}"
    
    df['race_id'] = df.apply(make_id, axis=1)
    return df

def parse_payouts(df_pay):
    payouts = {}
    for idx, row in df_pay.iterrows():
        rid = row['race_id']
        payouts[rid] = {'win': {}, 'place': {}, 'wide': {}, 'quinella': {}, 'exacta': {}, 'trio': {}, 'trifecta': {}}
        
        # Win
        if pd.notna(row.get('haraimodoshi_tansho_1a')):
             try:
                 h = int(row['haraimodoshi_tansho_1a'])
                 p = int(row['haraimodoshi_tansho_1b'])
                 payouts[rid]['win'][h] = p
             except: pass

        # Place
        for i in range(1, 6):
            if pd.notna(row.get(f'haraimodoshi_fukusho_{i}a')):
                try:
                    h = int(row[f'haraimodoshi_fukusho_{i}a'])
                    p = int(row[f'haraimodoshi_fukusho_{i}b'])
                    if h > 0: payouts[rid]['place'][h] = p
                except: pass

        # Wide
        for i in range(1, 8):
            if pd.notna(row.get(f'haraimodoshi_wide_{i}a')):
                try:
                    h_str = str(row[f'haraimodoshi_wide_{i}a']).zfill(4)
                    h1, h2 = int(h_str[:2]), int(h_str[2:])
                    p = int(row[f'haraimodoshi_wide_{i}b'])
                    payouts[rid]['wide'][frozenset({h1, h2})] = p
                except: pass

        # Quinella
        for i in range(1, 4):
            if pd.notna(row.get(f'haraimodoshi_umaren_{i}a')):
                try:
                    h_str = str(row[f'haraimodoshi_umaren_{i}a']).zfill(4)
                    h1, h2 = int(h_str[:2]), int(h_str[2:])
                    p = int(row[f'haraimodoshi_umaren_{i}b'])
                    payouts[rid]['quinella'][frozenset({h1, h2})] = p
                except: pass

        # Exacta
        for i in range(1, 7):
            if pd.notna(row.get(f'haraimodoshi_umatan_{i}a')):
                try:
                    h_str = str(row[f'haraimodoshi_umatan_{i}a']).zfill(4)
                    h1, h2 = int(h_str[:2]), int(h_str[2:])
                    p = int(row[f'haraimodoshi_umatan_{i}b'])
                    payouts[rid]['exacta'][(h1, h2)] = p
                except: pass
        
        # Trifecta
        for i in range(1, 7):
            if pd.notna(row.get(f'haraimodoshi_sanrentan_{i}a')):
                try:
                    h_str = str(row[f'haraimodoshi_sanrentan_{i}a']).zfill(6)
                    h1, h2, h3 = int(h_str[:2]), int(h_str[2:4]), int(h_str[4:])
                    p = int(row[f'haraimodoshi_sanrentan_{i}b'])
                    payouts[rid]['trifecta'][(h1, h2, h3)] = p
                except: pass

    return payouts

def simulate(year, df_data, payout_dict, strategies):
    logger.info(f"Scanning {len(strategies)} strategies on {year} data ({len(df_data.groupby('race_id'))} races)...")
    
    results = [{'name': s['name'], 'cost': 0, 'return': 0, 'hits': 0, 'bets': 0, 'races_bet': 0} for s in strategies]
    
    race_groups = df_data.groupby('race_id')
    
    for rid, group in race_groups:
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        
        group = group.sort_values('pred_prob', ascending=False)
        top_horses = group['horse_number'].to_numpy()
        scores = group['pred_prob'].to_numpy()
        odds = group['odds_final'].to_numpy()
        
        top1_horse = top_horses[0]
        top1_score = scores[0]
        top1_odds = odds[0]
        
        for idx, s in enumerate(strategies):
            stype = s['type']
            stat = results[idx]
            bets_in_race = 0
            
            if stype == 'win':
                 for i in range(len(top_horses)):
                    if scores[i] < s['th']: break
                    if s['min_odds'] <= odds[i]:
                        h = top_horses[i]
                        bets_in_race += 1
                        stat['cost'] += 100
                        stat['bets'] += 1
                        if h in pay['win']:
                            stat['return'] += pay['win'][h]
                            stat['hits'] += 1
            
            elif stype == 'place':
                 for i in range(len(top_horses)):
                    if scores[i] < s['th']: break
                    if odds[i] >= s['min_odds']:
                        h = top_horses[i]
                        bets_in_race += 1
                        stat['cost'] += 100
                        stat['bets'] += 1
                        if h in pay['place']:
                            stat['return'] += pay['place'][h]
                            stat['hits'] += 1
            else:
                 # Axis based
                 if top1_score < s['axis_th']: continue
                 if top1_odds < s['axis_min_odds']: continue
                 n_flow = s['n_flow']
                 if len(top_horses) < n_flow + 1: continue
                 
                 axis = top1_horse
                 flow = top_horses[1:n_flow+1]
                 
                 if stype == 'wide_nagashi':
                    for f in flow:
                        bets_in_race += 1
                        stat['cost'] += 100
                        stat['bets'] += 1
                        key = frozenset({axis, f})
                        if key in pay['wide']:
                            stat['return'] += pay['wide'][key]
                            stat['hits'] += 1

                 elif stype == 'quinella_nagashi':
                    for f in flow:
                        bets_in_race += 1
                        stat['cost'] += 100
                        stat['bets'] += 1
                        key = frozenset({axis, f})
                        if key in pay['quinella']:
                            stat['return'] += pay['quinella'][key]
                            stat['hits'] += 1

                 elif stype == 'exacta_nagashi':
                    for f in flow:
                        bets_in_race += 1
                        stat['cost'] += 100
                        stat['bets'] += 1
                        key = (axis, f)
                        if key in pay['exacta']:
                            stat['return'] += pay['exacta'][key]
                            stat['hits'] += 1

                 elif stype == 'trifecta_1st_nagashi':
                    perms = itertools.permutations(flow, 2)
                    for p in perms:
                        bets_in_race += 1
                        stat['cost'] += 100
                        stat['bets'] += 1
                        key = (axis, p[0], p[1])
                        if key in pay['trifecta']:
                            stat['return'] += pay['trifecta'][key]
                            stat['hits'] += 1

            if bets_in_race > 0:
                stat['races_bet'] += 1
                
    final_res = []
    for stat in results:
        roi = 0
        if stat['cost'] > 0:
            roi = stat['return'] / stat['cost'] * 100
        hit_rate = 0
        if stat['bets'] > 0:
             hit_rate = stat['hits'] / stat['bets'] * 100
             
        final_res.append({
            'name': stat['name'],
            'roi': roi,
            'hit_rate': hit_rate,
            'races': stat['races_bet'],
            'tickets': stat['bets'],
            'cost': stat['cost'],
            'profit': stat['return'] - stat['cost']
        })
    return pd.DataFrame(final_res)

def load_odds_from_db(year):
    logger.info(f"Loading Odds Data (jvd_se) for {year}...")
    engine = get_db_engine()
    # jvd_se has tansho_odds
    query = f"""
    SELECT 
        kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
        umaban, tansho_odds
    FROM jvd_se
    WHERE kaisai_nen = '{year}'
    """
    df = pd.read_sql(query, engine)
    def make_id(row):
        return f"{row['kaisai_nen']}{str(row['keibajo_code']).zfill(2)}{str(row['kaisai_kai']).zfill(2)}{str(row['kaisai_nichime']).zfill(2)}{str(row['race_bango']).zfill(2)}"
    df['race_id'] = df.apply(make_id, axis=1)
    df['horse_number'] = df['umaban'].astype(int)
    # Check for NaNs or 0s
    df['odds_final'] = pd.to_numeric(df['tansho_odds'], errors='coerce').fillna(1.0)
    return df[['race_id', 'horse_number', 'odds_final']]

def main():
    logger.info("🚀 Forward Validation (2024 -> 2025)...")
    
    # Load Data
    pred_path_24 = "data/temp_t2/T2_predictions_2024_2025.parquet"
    pred_path_25 = "data/temp_t2/T2_predictions_2025_only.parquet"
    
    # Load Odds from DB (Reliable source)
    df_odds_24 = load_odds_from_db(2024)
    df_odds_25 = load_odds_from_db(2025)
    
    logger.info(f"Odds 2025 rows: {len(df_odds_25)}")
    if not df_odds_25.empty:
        logger.info(f"Sample Odds ID: {df_odds_25['race_id'].iloc[0]}")
    
    df_odds = pd.concat([df_odds_24, df_odds_25], ignore_index=True)
    df_odds['race_id'] = df_odds['race_id'].astype(str)
    
    # Load 2024
    if not os.path.exists(pred_path_24): 
        logger.error("2024 Prediction file not found")
        return
    df_pred_24 = pd.read_parquet(pred_path_24)
    df_pred_24['race_id'] = df_pred_24['race_id'].astype(str)
    df_pred_24['date'] = pd.to_datetime(df_pred_24['date'])
    df_2024 = df_pred_24[df_pred_24['date'].dt.year == 2024].copy()
    
    if not df_odds.empty:
        df_2024 = pd.merge(df_2024, df_odds, on=['race_id', 'horse_number'], how='left')
        df_2024['odds_final'] = df_2024['odds_final'].fillna(1.0)
    else:
        df_2024['odds_final'] = 1.0

    # Load 2025
    if os.path.exists(pred_path_25):
        logger.info("Loading 2025 predictions...")
        df_2025 = pd.read_parquet(pred_path_25)
        df_2025['race_id'] = df_2025['race_id'].astype(str)
        df_2025['date'] = pd.to_datetime(df_2025['date']) # Should be 2025
        
        logger.info(f"Predictions 2025 rows: {len(df_2025)}")
        if not df_2025.empty:
            logger.info(f"Sample Pred ID: {df_2025['race_id'].iloc[0]}")
        
        if not df_odds.empty:
            # Drop odds_final from pred if exists to avoid _x _y collision
            if 'odds_final' in df_2025.columns:
                df_2025 = df_2025.drop(columns=['odds_final'])
            
            df_2025 = pd.merge(df_2025, df_odds, on=['race_id', 'horse_number'], how='left')
            logger.info(f"Merged 2025 rows: {len(df_2025)}")
            # Check odds dist
            logger.info(f"Odds > 1.0 count: {len(df_2025[df_2025['odds_final'] > 1.0])}")
            logger.info(f"Max Pred Prob 2025: {df_2025['pred_prob'].max()}")
            logger.info(f"Mean Pred Prob 2025: {df_2025['pred_prob'].mean()}")
            logger.info(f"Max Odds 2025: {df_2025['odds_final'].max()}")
            df_2025['odds_final'] = df_2025['odds_final'].fillna(1.0)
        else:
            df_2025['odds_final'] = 1.0
    else:
        logger.error("2025 Prediction file not found (Run inference first)")
        return
    
    # Generate Strategies Grid
    strategies = []
    confs = [0.45, 0.50, 0.55]
    min_odds_list = [2.0, 3.0, 3.5]
    flows = [6, 7, 8]
    
    strategies.append({'type': 'win', 'name': 'Win C>0.50 Od>2.0', 'th': 0.50, 'min_odds': 2.0})
    strategies.append({'type': 'place', 'name': 'Place C>0.50 Od>1.5', 'th': 0.50, 'min_odds': 1.5})
    
    for c in confs:
        for m_odd in min_odds_list:
             for f in flows:
                strategies.append({'type': 'wide_nagashi', 'name': f'Wide Ax(C>{c},Od>{m_odd})-Flow{f}', 'axis_th': c, 'axis_min_odds': m_odd, 'n_flow': f})
                strategies.append({'type': 'quinella_nagashi', 'name': f'Quinella Ax(C>{c},Od>{m_odd})-Flow{f}', 'axis_th': c, 'axis_min_odds': m_odd, 'n_flow': f})
                strategies.append({'type': 'exacta_nagashi', 'name': f'Exacta Ax(C>{c},Od>{m_odd})-Flow{f}', 'axis_th': c, 'axis_min_odds': m_odd, 'n_flow': f})
                if c >= 0.55 and m_odd >= 2.0 and f==6:
                     strategies.append({'type': 'trifecta_1st_nagashi', 'name': f'Tri1st Ax(C>{c},Od>{m_odd})-Flow{f}', 'axis_th': c, 'axis_min_odds': m_odd, 'n_flow': f})

    # Step 1: Run 2024
    pay_2024 = parse_payouts(load_payout_data(2024))
    res_2024 = simulate(2024, df_2024, pay_2024, strategies)
    
    promising = res_2024[res_2024['roi'] > 95].copy()
    logger.info(f"Found {len(promising)} strategies with ROI > 95% in 2024.")
    
    if promising.empty:
        print("No promising strategies found.")
        return

    # Extract strictly the successful strategy configs
    # We can just filter the names
    promising_names = set(promising['name'].tolist())
    promising_configs = [s for s in strategies if s['name'] in promising_names]
    
    # Step 2: Run 2025
    pay_2025 = parse_payouts(load_payout_data(2025))
    
    logger.info(f"Payout 2025 count: {len(pay_2025)}")
    if pay_2025:
        logger.info(f"Sample Payout ID: {list(pay_2025.keys())[0]}")
        
    uniq_pred = df_2025['race_id'].unique()
    matches = sum(1 for r in uniq_pred if r in pay_2025)
    logger.info(f"Prediction Races: {len(uniq_pred)}")
    logger.info(f"Matches with Payouts: {matches}")
    
    res_2025 = simulate(2025, df_2025, pay_2025, promising_configs)
    
    # Merge
    merged = pd.merge(promising[['name', 'roi', 'races', 'profit']], 
                      res_2025[['name', 'roi', 'races', 'profit']], 
                      on='name', suffixes=('_24', '_25'))
    
    # Sort by 2024 ROI
    merged = merged.sort_values('roi_24', ascending=False)
    
    print("\n========= Forward Validation Results (2024 -> 2025) =========")
    print(merged.to_string(index=False))

if __name__ == "__main__":
    main()

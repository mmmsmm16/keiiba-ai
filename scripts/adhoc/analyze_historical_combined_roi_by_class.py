"""
Analyze Historical Combined (V13+V14) Performance by Race Class (2024-2025)
"""
import pandas as pd
import numpy as np
import os
import joblib
import sys
from datetime import datetime

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader

V13_OOF_2024 = 'data/predictions/v13_oof_2024_clean.parquet'
V13_OOF_2025 = 'data/predictions/v13_oof_2025_clean.parquet'
V14_MODEL_PATH = 'models/experiments/exp_gap_v14_production/model_v14.pkl'
V14_FEATS_PATH = 'models/experiments/exp_gap_v14_production/features.csv'
DATA_PATH = 'data/processed/preprocessed_data_v13_active.parquet'

def get_class_name(grade, joken):
    g = str(grade) if pd.notna(grade) else ' '
    if g in ['1', '2', '3']: return "重賞 (G1-G3)"
    if g in ['4', '5']: return "オープン (L/OP)"
    j = str(joken)
    if '000' in j or j == '0': return "新馬・未勝利"
    if '701' in j: return "1勝クラス"
    if '702' in j: return "2勝クラス"
    if '703' in j: return "3勝クラス"
    if j == '999': return "オープン"
    return "その他"

def main():
    print("Loading OOF and Feature Data...")
    oof24 = pd.read_parquet(V13_OOF_2024)
    oof25 = pd.read_parquet(V13_OOF_2025)
    df_oof = pd.concat([oof24, oof25], ignore_index=True)
    df_oof['race_id'] = df_oof['race_id'].astype(str)
    
    df_all = pd.read_parquet(DATA_PATH)
    df_all['date'] = pd.to_datetime(df_all['date'])
    # Focus on 2024-2025
    df_hist = df_all[df_all['date'].dt.year.isin([2024, 2025])].copy()
    df_hist['race_id'] = df_hist['race_id'].astype(str)
    
    print("Merging and Predicting V14...")
    df = pd.merge(df_hist, df_oof[['race_id', 'horse_number', 'pred_prob', 'odds']].rename(columns={'odds': 'oof_odds'}), 
                  on=['race_id', 'horse_number'])
    
    # Prep V14
    model_v14 = joblib.load(V14_MODEL_PATH)
    feats_v14 = pd.read_csv(V14_FEATS_PATH)['feature'].tolist()
    
    # Handle V14 features
    if 'odds_10min' not in df.columns: df['odds_10min'] = df['odds']
    df['odds_rank_10min'] = df.groupby('race_id')['odds_10min'].rank(method='min')
    pop_col = 'popularity' if 'popularity' in df.columns else 'tansho_ninki'
    df['rank_diff_10min'] = df[pop_col] - df['odds_rank_10min']
    df['odds_log_ratio_10min'] = np.log(df['odds'] + 1e-9) - np.log(df['odds_10min'] + 1e-9)
    df['odds_ratio_60_10'] = 1.0; df['odds_60min'] = df['odds_10min']; df['odds_final'] = df['odds']
    
    X_v14 = df.reindex(columns=feats_v14, fill_value=0.0).fillna(0.0)
    df['gap_v14'] = model_v14.predict(X_v14)
    
    # Ranks
    df['v13_rank'] = df.groupby('race_id')['pred_prob'].rank(ascending=False, method='first')
    df['v14_rank'] = df.groupby('race_id')['gap_v14'].rank(ascending=False, method='first')

    # Fetch Payouts (Slow but thorough)
    print("Fetching Payouts from DB (This may take a minute)...")
    loader = JraVanDataLoader()
    from sqlalchemy import text
    # Fetch for both years. 
    payout_map = {}
    for year in [2024, 2025]:
        q = text(f"SELECT * FROM jvd_hr WHERE kaisai_nen = '{year}'")
        with loader.engine.connect() as conn:
            df_hr = pd.read_sql(q, conn)
            for _, row in df_hr.iterrows():
                try:
                    rid = f"{int(row['kaisai_nen'])}{int(row['keibajo_code']):02}{int(row['kaisai_kai']):02}{int(row['kaisai_nichime']):02}{int(row['race_bango']):02}"
                    uma, wide = {}, {}
                    if pd.notna(row['haraimodoshi_umaren_1a']):
                        raw = str(int(row['haraimodoshi_umaren_1a'])).zfill(4)
                        uma[tuple(sorted((int(raw[:2]), int(raw[2:4]))))] = int(float(str(row['haraimodoshi_umaren_1b']).replace(',', '')))
                    for i in range(1, 4):
                        k_a, k_b = f'haraimodoshi_wide_{i}a', f'haraimodoshi_wide_{i}b'
                        if pd.notna(row.get(k_a)):
                            raw = str(int(row[k_a])).zfill(4)
                            wide[tuple(sorted((int(raw[:2]), int(raw[2:4]))))] = int(float(str(row[k_b]).replace(',', '')))
                    payout_map[rid] = {'uma': uma, 'wide': wide}
                except: continue

    # Simulation
    print("Simulating Betting...")
    results = []
    for rid in df['race_id'].unique():
        rdf = df[df['race_id'] == rid]
        pout = payout_map.get(rid)
        if not pout: continue
        
        r_class = get_class_name(rdf['grade_code'].iloc[0], rdf['kyoso_joken_code'].iloc[0])
        axis = rdf[rdf['v13_rank'] == 1].iloc[0]['horse_number']
        partners = rdf[rdf['v14_rank'] <= 5]['horse_number'].tolist()
        partners = [p for p in partners if p != axis]
        
        if not partners: continue
        inv = len(partners) * 200
        ret = 0
        for p in partners:
            pair = tuple(sorted((int(axis), int(p))))
            if pair in pout['uma']: ret += pout['uma'][pair]
            if pair in pout['wide']: ret += pout['wide'][pair]
        results.append({'race_class': r_class, 'inv': inv, 'ret': ret})

    df_res = pd.DataFrame(results)
    stats = df_res.groupby('race_class').agg({'inv':'sum', 'ret':'sum', 'race_class':'count'}).rename(columns={'race_class':'Races'})
    stats['ROI'] = (stats['ret'] / stats['inv']) * 100
    
    print("\n=== Combined (V13+V14) Historical Performance by Class (2024-2025) ===")
    print(stats.sort_values('ROI', ascending=False).to_string(formatters={'ROI':'{:.1f}%'.format}))

if __name__ == "__main__":
    main()

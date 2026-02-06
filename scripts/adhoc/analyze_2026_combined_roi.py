"""
Analyze 2026 Combined (V13+V14) Performance by Race Class (Robust Fix)
"""
import pandas as pd
import numpy as np
import os
import joblib
import sys
from datetime import datetime, timedelta

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

V13_MODEL_PATH = 'models/experiments/exp_lambdarank_hard_weighted/model.pkl'
V13_FEATS_PATH = 'models/experiments/exp_lambdarank_hard_weighted/features.csv'
V14_MODEL_PATH = 'models/experiments/exp_gap_v14_production/model_v14.pkl'
V14_FEATS_PATH = 'models/experiments/exp_gap_v14_production/features.csv'

def get_class_name(grade, joken):
    g = str(grade).strip()
    j = str(joken).strip()
    
    # Graded
    if g == 'A': return "G1"
    if g == 'B': return "G2"
    if g == 'C': return "G3"
    if g == 'L': return "Listed"
    
    # Open
    if g == 'E' or j == '999' or g in ['D', 'F', 'G', 'H']: return "OP"
    
    # Condition Mapping (Corrected)
    if j == '016': return "3勝クラス"
    if j == '010': return "2勝クラス"
    if j == '005': return "1勝クラス"
    if j in ['000', '701', '703']: return "未勝利・新馬"
    
    return "その他"

def main():
    print("Loading 2026 Data (Jan 1 - Jan 24) from DB...")
    loader = JraVanDataLoader()
    df_raw = loader.load(history_start_date='2026-01-01', end_date='2026-01-25', skip_odds=True)
    
    if df_raw.empty:
        print("No 2026 data found."); return

    # Preserve RAW meta
    meta = df_raw[['race_id', 'horse_number', 'odds', 'rank', 'grade_code', 'kyoso_joken_code', 'popularity']].copy()
    meta['race_id'] = meta['race_id'].astype(str); meta['horse_number'] = meta['horse_number'].astype(int)

    pipeline = FeaturePipeline(cache_dir='data/features_v14/prod_cache')
    df_feat = pipeline.load_features(df_raw, list(pipeline.registry.keys()))
    df_feat['race_id'] = df_feat['race_id'].astype(str); df_feat['horse_number'] = df_feat['horse_number'].astype(int)
    
    print("Predicting V13 & V14...")
    m_v13 = joblib.load(V13_MODEL_PATH)
    f_v13 = pd.read_csv(V13_FEATS_PATH, header=None).iloc[:, 0].tolist()
    if f_v13[0] in ['0', 'feature']: f_v13 = f_v13[1:]
    
    m_v14 = joblib.load(V14_MODEL_PATH)
    f_v14 = pd.read_csv(V14_FEATS_PATH)['feature'].tolist()

    X_v13 = df_feat.reindex(columns=f_v13, fill_value=-999.0).fillna(-999.0)
    df_feat['score_v13'] = m_v13.predict_proba(X_v13)[:, -1] if hasattr(m_v13, 'predict_proba') else m_v13.predict(X_v13)
    
    # Simple V14 Mock features
    if 'rank_diff_10min' not in df_feat.columns:
        df_feat['odds_rank_10min'] = df_feat.groupby('race_id')['odds'].rank(method='min')
        df_feat['rank_diff_10min'] = meta['popularity'].values - df_feat['odds_rank_10min'].values
        df_feat['odds_log_ratio_10min'] = 0.0
        df_feat['odds_ratio_60_10'] = 1.0; df_feat['odds_10min'] = df_feat['odds']; df_feat['odds_final'] = df_feat['odds']; df_feat['odds_60min'] = df_feat['odds']

    X_v14 = df_feat.reindex(columns=f_v14, fill_value=0.0).fillna(0.0)
    df_feat['gap_v14'] = m_v14.predict(X_v14)
    
    # 2. Payouts
    print("Fetching Payouts...")
    from sqlalchemy import text
    q_hr = text("SELECT * FROM jvd_hr WHERE kaisai_nen = '2026' AND kaisai_tsukihi BETWEEN '0101' AND '0125'")
    with loader.engine.connect() as conn:
        df_hr = pd.read_sql(q_hr, conn)
    
    payout_map = {}
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

    # 3. Simulation
    df_sim = pd.merge(df_feat[['race_id', 'horse_number', 'score_v13', 'gap_v14']], meta, on=['race_id', 'horse_number'], how='inner')
    df_sim['v13_rank'] = df_sim.groupby('race_id')['score_v13'].rank(ascending=False, method='first')
    df_sim['v14_rank'] = df_sim.groupby('race_id')['gap_v14'].rank(ascending=False, method='first')
    
    results = []
    for rid in df_sim['race_id'].unique():
        rdf = df_sim[df_sim['race_id'] == rid]
        pout = payout_map.get(rid)
        if not pout: continue
        
        row0 = rdf.iloc[0]
        r_class = get_class_name(row0['grade_code'], row0['kyoso_joken_code'])
        
        axis = rdf[rdf['v13_rank'] == 1].iloc[0]['horse_number']
        partners = rdf[rdf['v14_rank'] <= 5]['horse_number'].tolist()
        partners = [p for p in partners if p != axis]
        
        if not partners: continue
        
        inv = len(partners) * 200
        ret = 0
        hits = 0
        for p in partners:
            pair = tuple(sorted((int(axis), int(p))))
            if pair in pout['uma']: ret += pout['uma'][pair]; hits += 1
            if pair in pout['wide']: ret += pout['wide'][pair]; hits += 1
        
        results.append({'race_class': r_class, 'inv': inv, 'ret': ret, 'hits': hits})

    df_res = pd.DataFrame(results)
    if df_res.empty: print("No simulation data produced."); return
    
    stats = df_res.groupby('race_class').agg({'inv':'sum', 'ret':'sum', 'race_class':'count'}).rename(columns={'race_class': 'Races'})
    stats['ROI'] = (stats['ret'] / stats['inv']) * 100
    
    print("\n=== Combined (V13 Axis + V14 Partners) ROI by Class (2026 Jan 1-24) ===")
    print(stats.sort_values('ROI', ascending=False).to_string(formatters={'ROI':'{:.1f}%'.format}))

if __name__ == "__main__":
    main()

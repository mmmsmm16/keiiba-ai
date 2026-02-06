"""
Detailed Historical Combined (V13+V14) Performance by Race Class (2024-2025)
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

def get_detailed_class_name(grade, joken):
    g = str(grade).strip()
    j = str(joken).strip()
    
    # Graded
    if g == 'A': return "G1"
    if g == 'B': return "G2"
    if g == 'C': return "G3"
    if g == 'L': return "Listed"
    
    # Open
    if g == 'E' or j == '999' or g in ['D', 'F', 'G', 'H']: return "OP"
    
    # Condition Mapping (All inclusive)
    if j == '016': return "3勝クラス"
    if j == '010': return "2勝クラス"
    if j == '005': return "1勝クラス"
    if j in ['000', '701', '703']: return "未勝利・新馬"
    
    return "その他"

def main():
    print("Loading Data...")
    oof24 = pd.read_parquet(V13_OOF_2024)
    oof25 = pd.read_parquet(V13_OOF_2025)
    df_oof = pd.concat([oof24, oof25], ignore_index=True)
    df_oof['race_id'] = df_oof['race_id'].astype(str)
    
    df_all = pd.read_parquet(DATA_PATH)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_hist = df_all[df_all['date'].dt.year.isin([2024, 2025])].copy()
    df_hist['race_id'] = df_hist['race_id'].astype(str)
    
    print("Predicting V14...")
    df = pd.merge(df_hist, df_oof[['race_id', 'horse_number', 'pred_prob', 'odds']].rename(columns={'odds': 'oof_odds'}), 
                  on=['race_id', 'horse_number'])
    
    # Check classes in merged df
    df['race_class'] = [get_detailed_class_name(g, j) for g, j in zip(df['grade_code'], df['kyoso_joken_code'])]
    print("Class distribution in merged data:")
    print(df.drop_duplicates('race_id')['race_class'].value_counts())
    
    model_v14 = joblib.load(V14_MODEL_PATH)
    feats_v14 = pd.read_csv(V14_FEATS_PATH)['feature'].tolist()
    
    # Handle V14
    if 'odds_10min' not in df.columns: df['odds_10min'] = df['odds']
    df['odds_rank_10min'] = df.groupby('race_id')['odds_10min'].rank(method='min')
    pop_col = 'popularity' if 'popularity' in df.columns else 'tansho_ninki'
    df['rank_diff_10min'] = df[pop_col] - df['odds_rank_10min']
    df['odds_log_ratio_10min'] = 0.0 # historical proxy
    
    X_v14 = df.reindex(columns=feats_v14, fill_value=0.0).fillna(0.0)
    df['gap_v14'] = model_v14.predict(X_v14)
    
    df['v13_rank'] = df.groupby('race_id')['pred_prob'].rank(ascending=False, method='first')
    df['v14_rank'] = df.groupby('race_id')['gap_v14'].rank(ascending=False, method='first')

    print("Fetching Payouts from DB (Horse-based for robust alignment)...")
    loader = JraVanDataLoader()
    from sqlalchemy import text
    
    payout_map = {}
    for year in ['2024', '2025']:
        print(f"  Retrieving {year} payouts and horse mapping...")
        # Join HR (payouts) with SE (results) to get payout metadata PER HORSE
        q = text(f"""
            SELECT 
                TO_DATE(se.kaisai_nen || se.kaisai_tsukihi, 'YYYYMMDD') as date,
                se.keibajo_code as venue,
                se.ketto_toroku_bango as horse_id,
                hr.haraimodoshi_umaren_1a, hr.haraimodoshi_umaren_1b,
                hr.haraimodoshi_wide_1a, hr.haraimodoshi_wide_1b,
                hr.haraimodoshi_wide_2a, hr.haraimodoshi_wide_2b,
                hr.haraimodoshi_wide_3a, hr.haraimodoshi_wide_3b
            FROM jvd_se se
            JOIN jvd_hr hr ON se.kaisai_nen = hr.kaisai_nen 
                          AND se.kaisai_tsukihi = hr.kaisai_tsukihi
                          AND se.keibajo_code = hr.keibajo_code
                          AND se.race_bango = hr.race_bango
            WHERE se.kaisai_nen = '{year}'
        """)
        try:
            with loader.engine.connect() as conn:
                df_hr = pd.read_sql(q, conn)
                print(f"    Fetched {len(df_hr)} horse-payout records for {year}")
                for _, row in df_hr.iterrows():
                    try:
                        # Standardized key: (YYYYMMDD, 'JJ', 'GID')
                        dt_str = row['date'].strftime('%Y%m%d')
                        v_str = str(row['venue']).zfill(2)
                        h_str = str(row['horse_id']).strip()
                        key = (dt_str, v_str, h_str)
                        
                        uma, wide = {}, {}
                        if pd.notna(row['haraimodoshi_umaren_1a']):
                            raw = str(int(row['haraimodoshi_umaren_1a'])).zfill(4)
                            uma[tuple(sorted((int(raw[:2]), int(raw[2:4]))))] = int(float(str(row['haraimodoshi_umaren_1b']).replace(',', '')))
                        for i in range(1, 4):
                            k_a, k_b = f'haraimodoshi_wide_{i}a', f'haraimodoshi_wide_{i}b'
                            if pd.notna(row.get(k_a)):
                                raw = str(int(row[k_a])).zfill(4)
                                wide[tuple(sorted((int(raw[:2]), int(raw[2:4]))))] = int(float(str(row[k_b]).replace(',', '')))
                        payout_map[key] = {'uma': uma, 'wide': wide}
                    except: continue
        except Exception as e: print(f"    Error: {e}")

    print("Simulating...")
    if payout_map:
        print(f"  Payout Map contains {len(payout_map)} unique horse-race entries.")
    else:
        print("  WARNING: Payout map is still empty!")

    results = []
    rids = df['race_id'].unique()
    skip_counts = {"No Payout": 0, "No Axis/Partners": 0, "Other": 0}
    maiden_skip_counts = {"No Payout": 0, "No Axis/Partners": 0}
    
    print(f"  Looping through {len(rids)} races in data...")
    for rid in rids:
        rdf = df[df['race_id'] == rid]
        
        # Build key from DF (using the Axis horse as anchor)
        r0 = rdf.iloc[0]
        # v13_rank is already calculated in Predicting section
        axis_rdf = rdf[rdf['v13_rank'] == 1]
        if axis_rdf.empty:
            skip_counts["No Axis/Partners"] += 1
            if r0['race_class'] == "未勝利・新馬": maiden_skip_counts["No Axis/Partners"] += 1
            continue
        axis_row = axis_rdf.iloc[0]
        
        # Build key from DF: (YYYYMMDD, venue_zfill, horse_id_strip)
        dt_df = axis_row['date'].strftime('%Y%m%d')
        v_df = str(axis_row['venue']).zfill(2)
        h_df = str(axis_row['horse_id']).strip()
        df_lookup_key = (dt_df, v_df, h_df)
        
        pout = payout_map.get(df_lookup_key)
        if not pout:
            skip_counts["No Payout"] += 1
            if axis_row['race_class'] == "未勝利・新馬": 
                maiden_skip_counts["No Payout"] += 1
                # Debug one failure
                if maiden_skip_counts["No Payout"] == 1:
                    print(f"DEBUG FIRST FAILURE: Key={df_lookup_key}, RID={rid}")
            continue
        
        r_class = axis_row['race_class']
        axis_hnum = axis_row['horse_number']
        partners = rdf[rdf['v14_rank'] <= 5]['horse_number'].tolist()
        partners = [p for p in partners if p != axis_hnum]
        
        if not partners:
            skip_counts["No Axis/Partners"] += 1
            if r_class == "未勝利・新馬": maiden_skip_counts["No Axis/Partners"] += 1
            continue
        inv = len(partners) * 200
        ret = 0
        for p in partners:
            pair = tuple(sorted((int(axis_hnum), int(p))))
            if pair in pout['uma']: ret += pout['uma'][pair]
            if pair in pout['wide']: ret += pout['wide'][pair]
        results.append({'race_class': r_class, 'inv': inv, 'ret': ret})

    print(f"  Skip Summary: {skip_counts}")
    print(f"  Maiden Skip Summary: {maiden_skip_counts}")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No simulation results."); return

    stats = df_res.groupby('race_class').agg({'inv':'sum', 'ret':'sum', 'race_class':'count'}).rename(columns={'race_class':'Races'})
    stats['ROI'] = (stats['ret'] / stats['inv']) * 100
    
    print("\n=== Granular Combined (V13+V14) ROI (2024-2025) ===")
    print(stats.sort_values('ROI', ascending=False).to_string(formatters={'ROI':'{:.1f}%'.format}))

if __name__ == "__main__":
    main()

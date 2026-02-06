"""
Backtest Daily Report Script (Universal)
========================================
- 2024: Uses Golden OOF scores for simulation consistency.
- 2025/2026+: Uses Live models for real-world backtesting.
- Performance: Uses preprocessed parquet when available (Fast Path).
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
import argparse
import logging
from datetime import datetime, timedelta

# Force UTF-8 Output
sys.stdout.reconfigure(encoding='utf-8')

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

# --- Config ---
V13_OOF_PATH = 'data/predictions/v13_oof_2024_clean.parquet'
V13_MODEL_PATH = 'models/experiments/exp_lambdarank_hard_weighted/model.pkl'
V13_FEATS_PATH = 'models/experiments/exp_lambdarank_hard_weighted/features.csv'
V14_MODEL_PATH = 'models/experiments/exp_gap_v14_production/model_v14.pkl'
V14_FEATS_PATH = 'models/experiments/exp_gap_v14_production/features.csv'
PREPROCESSED_DATA_PATH = 'data/processed/preprocessed_data_v13_active.parquet'
CACHE_DIR = 'data/features_v14/prod_cache'

def load_v13_model_info(logger):
    try:
        model = joblib.load(V13_MODEL_PATH)
        feats = pd.read_csv(V13_FEATS_PATH)
        feat_list = feats['feature'].tolist() if 'feature' in feats.columns else feats.iloc[:, 0].tolist()
        return model, feat_list
    except Exception as e:
        logger.error(f"Failed to load V13 model: {e}")
        return None, []

def add_v13_odds_features(df, logger):
    # [Fix] Leak-free: Backtest must only use pre-race information.
    # We prioritize odds_10min (synced with JIT).
    if 'odds_10min' in df.columns:
        # Fill zero or invalid with NaN to ensure it falls to neutral fill_value later
        odds_base = df['odds_10min'].replace(0, np.nan)
    else:
        logger.warning("V13 odds features: odds_10min column missing. Leak-free fallback to neutral.")
        odds_base = pd.Series(np.nan, index=df.index)

    df['odds_calc'] = odds_base.fillna(10.0)
    df['odds_rank'] = df.groupby('race_id')['odds_calc'].rank(ascending=True, method='min')

    if 'relative_horse_elo_z' in df.columns:
        df['elo_rank'] = df.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False, method='min')
        df['odds_rank_vs_elo'] = df['odds_rank'] - df['elo_rank']
    else:
        df['odds_rank_vs_elo'] = 0

    df['is_high_odds'] = (df['odds_calc'] >= 10).astype(int)
    df['is_mid_odds'] = ((df['odds_calc'] >= 5) & (df['odds_calc'] < 10)).astype(int)
    return df

def softmax_per_race(scores, race_ids):
    df = pd.DataFrame({'race_id': race_ids, 'score': scores})
    df['score_shift'] = df.groupby('race_id')['score'].transform(lambda x: x - x.max())
    df['exp'] = np.exp(df['score_shift'])
    return df.groupby('race_id')['exp'].transform(lambda x: x / x.sum()).values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='YYYYMMDD or YYYYMMDD-YYYYMMDD')
    parser.add_argument('--summary', action='store_true', help='Minimal output')
    parser.add_argument('--live', action='store_true', help='Force full database pipeline')
    parser.add_argument('--with-odds', action='store_true', help='Include slow time-series odds for accuracy')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 1. Date Range
    if '-' in args.date: s_str, e_str = args.date.split('-')
    else: s_str = e_str = args.date
    start_dt, end_dt = pd.to_datetime(s_str), pd.to_datetime(e_str)
    
    # 2. Data Loading (Fast Path vs DB Path)
    df_raw_meta = None
    if not args.live and os.path.exists(PREPROCESSED_DATA_PATH):
        logger.info(f"Loading data range {s_str} to {e_str} (Fast Path)...")
        df_all = pd.read_parquet(PREPROCESSED_DATA_PATH)
        df_all['date'] = pd.to_datetime(df_all['date'])
        mask = (df_all['date'] >= start_dt) & (df_all['date'] <= end_dt)
        df_target = df_all[mask].copy()
    else:
        df_target = pd.DataFrame()

    if df_target.empty:
        logger.info(f"Target not in parquet or --live requested. Running DB pipeline (skip_odds={not args.with_odds})...")
        loader = JraVanDataLoader()
        
        # Optimized Load Strategy:
        # 1. First find horses participating on target dates
        logger.info(f"Identifying horses running between {s_str} and {e_str}...")
        q_horses = f"SELECT DISTINCT ketto_toroku_bango FROM jvd_se WHERE CONCAT(kaisai_nen, kaisai_tsukihi) BETWEEN '{s_str.replace('-','')}' AND '{e_str.replace('-','')}'"
        with loader.engine.connect() as conn:
            horse_ids = pd.read_sql(q_horses, conn)['ketto_toroku_bango'].tolist()
        
        logger.info(f"Found {len(horse_ids)} horses. Loading optimized context...")
        
        # Handle multiple dates by using the first date as target for context, 
        # but load_for_horses logic is usually enough if we broaden its where clause.
        # Actually, let's just loop if it's multiple days, or use a slightly broader load.
        # For simplicity and robustness, if it's many days, we might still hit memory limits, 
        # but for daily reports (1-2 days) it's perfect.
        
        # Use load_for_horses for each date if multiple, or just একদিন
        if start_dt == end_dt:
            # [Fix] skip_training must be False to avoid KeyError in training-related features
            df_db = loader.load_for_horses(horse_ids, start_dt.strftime('%Y-%m-%d'), skip_training=False) 
        else:
            # For ranges, we'll use a manually optimized load similar to load_for_horses but for a range
            logger.info("Range detected. Using optimized range load.")
            df_db = loader.load(history_start_date="2020-01-01", end_date=end_dt.strftime("%Y-%m-%d"), horse_ids=horse_ids, skip_training=False)
            # Re-fetch the full target dates to ensure context
            df_target_days = loader.load(history_start_date=start_dt.strftime("%Y-%m-%d"), end_date=end_dt.strftime("%Y-%m-%d"), skip_training=False)
            df_db = pd.concat([df_db, df_target_days]).drop_duplicates(['race_id', 'horse_number'])

        if df_db.empty:
            logger.error("No data found in DB."); return
            
        pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
        # [Fix] Force recalculation for JIT runs to ensure feature consistency and avoid stale cache
        df_features = pipeline.load_features(df_db, list(pipeline.registry.keys()), force=True)
        
        # Meta merge
        meta_cols = ['race_id', 'horse_number', 'date', 'odds', 'popularity', 'horse_name', 'rank']
        base_map = df_db[meta_cols].drop_duplicates(['race_id', 'horse_number'])
        if 'race_id' not in df_features.columns: df_features = df_features.reset_index()
        
        # Cleanup types
        df_features['race_id'] = df_features['race_id'].astype(str)
        base_map['race_id'] = base_map['race_id'].astype(str)
        df_features['horse_number'] = pd.to_numeric(df_features['horse_number'], errors='coerce').fillna(0).astype(int)
        base_map['horse_number'] = pd.to_numeric(base_map['horse_number'], errors='coerce').fillna(0).astype(int)
        
        # Drop colliding
        df_features = df_features.drop(columns=[c for c in ['date','odds','popularity','horse_name'] if c in df_features.columns], errors='ignore')
        df_target = pd.merge(df_features, base_map, on=['race_id', 'horse_number'])
        df_target['date'] = pd.to_datetime(df_target['date'])
        df_target = df_target[(df_target['date'] >= start_dt) & (df_target['date'] <= end_dt)].copy()

        # --- Detailed Odds Fetching (Synced with predict_combined_formation.py) ---
        if 'odds_10min' not in df_target.columns:
            df_target['odds_10min'] = np.nan
        
        # Fallback from jvd_o1 if odds_10min is missing
        invalid_mask = df_target['odds_10min'].isna() | (df_target['odds_10min'] <= 0)
        if invalid_mask.any():
            try:
                date_str = start_dt.strftime("%Y%m%d")
                year_str = f"'{date_str[:4]}'"
                q_o1 = f"SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen = {year_str} AND kaisai_tsukihi = '{date_str[4:]}'"
                with loader.engine.connect() as conn:
                    df_o1_raw = pd.read_sql(q_o1, conn)
                if not df_o1_raw.empty:
                    parsed = []
                    for _, row in df_o1_raw.iterrows():
                        rid = f"{int(float(row['kaisai_nen']))}{int(float(row['keibajo_code'])):02}{int(float(row['kaisai_kai'])):02}{int(float(row['kaisai_nichime'])):02}{int(float(row['race_bango'])):02}"
                        s = row['odds_tansho']
                        if not isinstance(s, str): continue
                        for i in range(0, len(s), 8):
                            chunk = s[i:i+8]
                            if len(chunk) < 8: break
                            try:
                                parsed.append({
                                    'race_id': rid, 'horse_number': int(chunk[0:2]),
                                    'odds_10min_o1': int(chunk[2:6]) / 10.0,
                                    'popularity_10min_o1': int(chunk[6:8])
                                })
                            except: continue
                    if parsed:
                        df_o1 = pd.DataFrame(parsed)
                        df_target = pd.merge(df_target, df_o1, on=['race_id', 'horse_number'], how='left')
                        df_target.loc[invalid_mask, 'odds_10min'] = df_target.loc[invalid_mask, 'odds_10min_o1']
                        # Keep popularity_10min for rank_diff_10min
                        if 'popularity_10min' not in df_target.columns:
                             df_target['popularity_10min'] = df_target['popularity_10min_o1']
                        else:
                             df_target['popularity_10min'] = df_target['popularity_10min'].fillna(df_target['popularity_10min_o1'])
                        df_target = df_target.drop(columns=['odds_10min_o1', 'popularity_10min_o1'], errors='ignore')
            except Exception as e:
                logger.warning(f"Failed to fetch jvd_o1 odds: {e}")

        # [Fix] Strictly Leak-free Final fallback. Do NOT use 'odds' here as it's closing odds in backtest.
        df_target['odds_10min'] = df_target['odds_10min'].fillna(10.0)
        df_target.loc[df_target['odds_10min'] <= 0, 'odds_10min'] = 10.0

    if df_target.empty:
        logger.error(f"No races found for {args.date}.Verify DB/Date."); return

    # 3. V13 Scores
    is_in_oof_range = (start_dt.year in [2024, 2025] and end_dt.year in [2024, 2025])
    missing_oof = True
    
    if is_in_oof_range:
        logger.info(f"Merging Golden V13 OOF scores for {start_dt.year}-{end_dt.year} consistency...")
        
        oof_files = []
        if 2024 in range(start_dt.year, end_dt.year + 1):
            oof_files.append(pd.read_parquet(V13_OOF_PATH))
        if 2025 in range(start_dt.year, end_dt.year + 1):
            oof_25_path = 'data/predictions/v13_oof_2025_clean.parquet'
            if os.path.exists(oof_25_path):
                oof_files.append(pd.read_parquet(oof_25_path))
        
        if oof_files:
            df_oof = pd.concat(oof_files, ignore_index=True).rename(columns={'pred_prob': 'prob_v13'})
            df_oof['race_id'] = df_oof['race_id'].astype(str)
            df_target['race_id'] = df_target['race_id'].astype(str)
            df_target = df_target.drop(columns=['prob_v13'], errors='ignore')
            # Use left join to keep all target rows even if OOF is incomplete
            df_target = pd.merge(df_target, df_oof[['race_id', 'horse_number', 'prob_v13']], on=['race_id', 'horse_number'], how='left')
            
            missing_rows = df_target['prob_v13'].isna().sum()
            if missing_rows > 0:
                logger.info(f"{missing_rows} rows missing from OOF. Falling back to Live for these rows.")
                missing_oof = True
            else:
                missing_oof = False
        else:
            logger.warning("OOF files not found. Falling back to Live.")
            missing_oof = True

    if missing_oof:
        logger.info("Predicting V13 Scores (Live Model) for missing/all target rows...")
        m_v13, f_v13 = load_v13_model_info(logger)
        
        # Identify rows that need prediction
        mask = df_target['prob_v13'].isna() if 'prob_v13' in df_target.columns else pd.Series(True, index=df_target.index)
        
        if mask.any():
            # [Sync] Apply odds features to the sliced subset explicitly, then build features
            df_v13_subset = add_v13_odds_features(df_target[mask].copy(), logger)
            
            # [Match build_v13_features logic precisely]
            X_v13 = df_v13_subset.reindex(columns=f_v13, fill_value=0.0)
            for col in X_v13.columns:
                if X_v13[col].dtype.name == 'category':
                    X_v13[col] = X_v13[col].cat.codes
                elif X_v13[col].dtype == 'object':
                    X_v13[col] = pd.to_numeric(X_v13[col], errors='coerce')
            X_v13 = X_v13.fillna(0.0).astype(float)
            
            if not X_v13.empty:
                # [Sync] Match predict calls
                if hasattr(m_v13, 'predict_proba'):
                    v13_scores = m_v13.predict_proba(X_v13)
                    if isinstance(v13_scores, np.ndarray):
                        if v13_scores.ndim == 2:
                            # Use second column if available (binary), else first
                            v13_scores = v13_scores[:, 1] if v13_scores.shape[1] > 1 else v13_scores[:, 0]
                        else:
                            v13_scores = v13_scores.flatten()
                else:
                    v13_scores = m_v13.predict(X_v13)
                
                # [Sync] Softmax application
                df_target.loc[mask, 'prob_v13'] = softmax_per_race(v13_scores, df_v13_subset['race_id'].values)

    # 4. V14 Gaps
    logger.info("Predicting V14 Gaps (Live Model)...")
    m_v14 = joblib.load(V14_MODEL_PATH)
    f_v14 = pd.read_csv(V14_FEATS_PATH)['feature'].tolist()
    
    # [Fix] Leak-free Derived Features (Backtest simulation must NOT use final closing odds)
    # odds_final for features should be our latest pre-race estimate (odds_10min).
    # We keep the real 'odds' for reporting/payouts, but 'X_v14' must be clean.
    df_target['odds_final'] = df_target['odds_10min'].copy()
    
    # Ratios using pre-race data only
    df_target['odds_ratio_10min'] = df_target['odds_final'] / df_target['odds_10min'].replace(0, np.nan)
    df_target['odds_ratio_10min'] = df_target['odds_ratio_10min'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    if 'odds_60min' not in df_target.columns:
        df_target['odds_60min'] = df_target['odds_10min']
    df_target['odds_60min'] = df_target['odds_60min'].fillna(df_target['odds_10min']) # Sync

    df_target['odds_ratio_60_10'] = df_target['odds_10min'] / df_target['odds_60min'].replace(0, np.nan)
    df_target['odds_ratio_60_10'] = df_target['odds_ratio_60_10'].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df_target['odds_log_ratio_10min'] = np.log(df_target['odds_final'] + 1e-9) - np.log(df_target['odds_10min'] + 1e-9)
    df_target['odds_log_ratio_10min'] = df_target['odds_log_ratio_10min'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df_target['odds_rank_10min'] = df_target.groupby('race_id')['odds_10min'].rank(method='min')
    
    # [Fix] Use popularity_10min to avoid leakage from final popularity
    if 'popularity_10min' in df_target.columns:
        pop_base = df_target['popularity_10min']
    else:
        pop_base = pd.Series(np.nan, index=df_target.index)
        
    pop_base = pop_base.fillna(df_target['odds_rank_10min'])
    df_target['rank_diff_10min'] = pop_base - df_target['odds_rank_10min']
    
    if 'field_size' not in df_target.columns:
        df_target['field_size'] = df_target.groupby('race_id')['horse_number'].transform('count')

    # Pred V14
    X_v14 = df_target.reindex(columns=f_v14, fill_value=0.0).fillna(0.0)
    df_target['gap_v14'] = m_v14.predict(X_v14)
    
    # Ranks
    df_target['rank_v13'] = df_target.groupby('race_id')['prob_v13'].rank(ascending=False, method='first')
    df_target['rank_v14'] = df_target.groupby('race_id')['gap_v14'].rank(ascending=False, method='first')

    # 5. Payouts
    logger.info("Fetching Actual Payouts...")
    loader = JraVanDataLoader()
    from sqlalchemy import text
    q_hr = text("SELECT * FROM jvd_hr WHERE kaisai_nen = :y AND kaisai_tsukihi BETWEEN :s AND :e")
    with loader.engine.connect() as conn:
        df_hr = pd.read_sql(q_hr, conn, params={'y': str(start_dt.year), 's': s_str[4:], 'e': e_str[4:]})
    
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

    # 6. 評価 (統合版)
    report_data = []
    
    # 軸精度集計用
    axis1_stats = {'win': 0, 'ren': 0, 'fuku': 0, 'total': 0}
    axis2_stats = {'win': 0, 'ren': 0, 'fuku': 0, 'total': 0}

    place_map = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
    for rid in sorted(df_target['race_id'].unique()):
        rdf = df_target[df_target['race_id'] == rid]
        pout = payout_map.get(rid)
        
        # --- 軸馬の特定 (V13 Rank 1 & 2) ---
        a1_row = rdf[rdf['rank_v13'] == 1].iloc[0]
        a2_rows = rdf[rdf['rank_v13'] == 2]
        a2_row = a2_rows.iloc[0] if not a2_rows.empty else None
        
        # 軸の精度集計 (有効な着順データ 1以上 があれば集計)
        if pd.notna(a1_row['rank']) and a1_row['rank'] > 0:
            axis1_stats['total'] += 1
            if a1_row['rank'] == 1: axis1_stats['win'] += 1
            if a1_row['rank'] <= 2: axis1_stats['ren'] += 1
            if a1_row['rank'] <= 3: axis1_stats['fuku'] += 1
        if a2_row is not None and pd.notna(a2_row['rank']) and a2_row['rank'] > 0:
            axis2_stats['total'] += 1
            if a2_row['rank'] == 1: axis2_stats['win'] += 1
            if a2_row['rank'] <= 2: axis2_stats['ren'] += 1
            if a2_row['rank'] <= 3: axis2_stats['fuku'] += 1

        # 表示用マーカー生成ヘルパー
        def get_marker_str(row):
            if row is None: return "-"
            # 有効な着順は 1以上。0やNaNは未確定扱い
            rank_val = int(row['rank']) if (pd.notna(row['rank']) and row['rank'] > 0) else "?"
            res = f"{row['horse_number']}({rank_val})"
            if rank_val == 1: res += "◎"
            elif rank_val != "?" and rank_val <= 3: res += "○"
            return res

        # --- 1軸戦略 (A1 - P1~5) ---
        p1_rows = rdf[rdf['rank_v14'] <= 5].sort_values('rank_v14')
        partners_list = []
        ret1 = 0
        for _, prow in p1_rows.iterrows():
            if prow['horse_number'] == a1_row['horse_number']: continue
            
            if pout:
                pair = tuple(sorted((int(a1_row['horse_number']), int(prow['horse_number']))))
                if pair in pout.get('uma', {}): ret1 += pout['uma'][pair]
                if pair in pout.get('wide', {}): ret1 += pout['wide'][pair]
            
            rank_val = int(prow['rank']) if (pd.notna(prow['rank']) and prow['rank'] > 0) else "?"
            p_str = f"{prow['horse_number']}({rank_val})"
            if rank_val != "?":
                if prow['popularity'] >= 7 and rank_val <= 3: p_str += "★"
                elif rank_val <= 3: p_str += "○"
            partners_list.append(p_str)
        inv1 = len(partners_list) * 200

        # --- 軸同士戦略 (A1 - A2) ---
        ret_ax = 0; inv_ax = 0
        if a2_row is not None:
            inv_ax = 200
            if pout:
                pair_ax = tuple(sorted((int(a1_row['horse_number']), int(a2_row['horse_number']))))
                if pair_ax in pout.get('uma', {}): ret_ax += pout['uma'][pair_ax]
                if pair_ax in pout.get('wide', {}): ret_ax += pout['wide'][pair_ax]

        # --- 2軸戦略 (A1&2 - P1~5) ---
        ret2 = 0; inv2 = 0
        if a2_row is not None:
            # A1-A2 の結果は既に ret_ax にある
            ret2 = ret_ax
            # 相手馬（軸2頭を除くV14上位5頭）
            p2_nums = [n for n in p1_rows['horse_number'].tolist() if n not in [a1_row['horse_number'], a2_row['horse_number']]]
            for p_num in p2_nums:
                for ax in [a1_row, a2_row]:
                    if pout:
                        pair = tuple(sorted((int(ax['horse_number']), int(p_num))))
                        if pair in pout.get('uma', {}): ret2 += pout['uma'][pair]
                        if pair in pout.get('wide', {}): ret2 += pout['wide'][pair]
            inv2 = (1 + 2 * len(p2_nums)) * 200

        report_data.append({
            'date': rdf['date'].iloc[0], 'rid': rid, 'place': place_map.get(rid[4:6], rid[4:6]), 'r': rid[10:12],
            'axis1': get_marker_str(a1_row), 'axis2': get_marker_str(a2_row),
            'partners': ",".join(partners_list),
            'ret1': ret1, 'ret_ax': ret_ax, 'ret2': ret2,
            'inv1': inv1, 'inv_ax': inv_ax, 'inv2': inv2
        })

    df_rep = pd.DataFrame(report_data)
    if df_rep.empty: print("対象日付のデータが見つかりませんでした。"); return

    # --- Print 統合レポート ---
    suffix = "(OOFスコア使用)" if is_in_oof_range else "(Liveモデル使用)"
    print(f"\nバックテスト結果: {args.date} {suffix}")
    print("マーク説明: ◎:軸1着, ○:3着以内, ★:穴馬(7人気以下)3着以内")
    if not args.summary:
        header = f"{'場所':<4} {'R':<2} {'軸1':<8} {'軸2':<8} {'穴(相手)':<32} {'1軸払':<6} {'軸同払':<6} {'2軸払'}"
        print(header)
        print("-" * 110)
        for _, r in df_rep.iterrows():
            print(f"{r['place']:<4} {r['r']:<2} {r['axis1']:<8} {r['axis2']:<8} {r['partners']:<32} {r['ret1']:<6} {r['ret_ax']:<6} {r['ret2']}")

    # --- Summary ---
    print("\n[戦略別トータル集計]")
    inv1 = df_rep['inv1'].sum(); ret1 = df_rep['ret1'].sum()
    inv_ax = df_rep['inv_ax'].sum(); ret_ax = df_rep['ret_ax'].sum()
    inv2 = df_rep['inv2'].sum(); ret2 = df_rep['ret2'].sum()
    
    print(f"1軸設定 (軸1-相手1~5):  投資={inv1:,} 回収={ret1:,} 回収率={ret1/inv1*100:>6.1f}%")
    if inv_ax > 0:
        print(f"軸同士 (軸1-軸2):       投資={inv_ax:,} 回収={ret_ax:,} 回収率={ret_ax/inv_ax*100:>6.1f}%")
    if inv2 > 0:
        print(f"2軸設定 (軸1,2-相手1~5): 投資={inv2:,} 回収={ret2:,} 回収率={ret2/inv2*100:>6.1f}%")

    print("\n[軸馬モデル精度 (V13)]")
    if axis1_stats['total'] > 0:
        a1_win = axis1_stats['win'] / axis1_stats['total'] * 100
        a1_ren = axis1_stats['ren'] / axis1_stats['total'] * 100
        a1_fuku = axis1_stats['fuku'] / axis1_stats['total'] * 100
        print(f"軸1 (1位): 勝率:{a1_win:>5.1f}%  連対率:{a1_ren:>5.1f}%  複勝率:{a1_fuku:>5.1f}%")
    if axis2_stats['total'] > 0:
        a2_win = axis2_stats['win'] / axis2_stats['total'] * 100
        a2_ren = axis2_stats['ren'] / axis2_stats['total'] * 100
        a2_fuku = axis2_stats['fuku'] / axis2_stats['total'] * 100
        print(f"軸2 (2位): 勝率:{a2_win:>5.1f}%  連対率:{a2_ren:>5.1f}%  複勝率:{a2_fuku:>5.1f}%")

if __name__ == "__main__":
    main()

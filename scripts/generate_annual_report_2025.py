"""
Comprehensive Annual Backtest Report Generator (2025)
=====================================================
- Generates detailed ROI reports for JRA 2025.
- Summaries: Yearly, Monthly, Daily, Venue, Class.
- Format: Full detailed table in Markdown.
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
CACHE_DIR = 'data/features_v14/prod_cache'
OUTPUT_PATH = 'reports/backtest_2025_full_report.md'

def load_v13_model_info(logger):
    try:
        model = joblib.load(V13_MODEL_PATH)
        feats = pd.read_csv(V13_FEATS_PATH)
        feat_list = feats['feature'].tolist() if 'feature' in feats.columns else feats.iloc[:, 0].tolist()
        return model, feat_list
    except Exception as e:
        logger.error(f"Failed to load V13 model: {e}")
        return None, []

def add_v13_odds_features(df):
    base = df['odds_10min'] if 'odds_10min' in df.columns else df['odds'] if 'odds' in df.columns else pd.Series(10.0, index=df.index)
    df['odds_calc'] = base.fillna(10.0)
    df['odds_rank'] = df.groupby('race_id')['odds_calc'].rank(ascending=True, method='min')
    df['is_high_odds'] = (df['odds_calc'] >= 10).astype(int)
    df['is_mid_odds'] = ((df['odds_calc'] >= 5) & (df['odds_calc'] < 10)).astype(int)
    return df

def softmax_per_race(scores, race_ids):
    df = pd.DataFrame({'race_id': race_ids, 'score': scores})
    df['score_shift'] = df.groupby('race_id')['score'].transform(lambda x: x - x.max())
    df['exp'] = np.exp(df['score_shift'])
    return df.groupby('race_id')['exp'].transform(lambda x: x / x.sum()).values

def categorize_race_class(grade_code, cond_code):
    try:
        g = int(float(grade_code))
        c = int(float(cond_code))
        if g == 1: return "G1"
        if g == 2: return "G2"
        if g == 3: return "G3"
        if g >= 5: return "OP/L"
        if c == 16: return "3勝クラス"
        if c == 10: return "2勝クラス"
        if c == 5: return "1勝クラス"
        if c == 0: return "未勝利/新馬"
        return "その他"
    except: return "不明"

# --- Helper for LightGBM ---
def clean_for_lightgbm(df, feature_cols):
    """
    Ensure all features are numeric or label-encoded for LightGBM sklearn API.
    Handles object/string/category columns by converting to codes.
    """
    df_clean = df.copy()
    existing = [c for c in feature_cols if c in df_clean.columns]
    
    # 1. Select non-numeric columns
    # include=['object', 'category', 'string'] and excludes number
    cat_cols = df_clean[existing].select_dtypes(exclude=[np.number]).columns.tolist()
    
    for c in cat_cols:
        # Try converting to numeric first (e.g. "12.3")
        try:
            s_num = pd.to_numeric(df_clean[c], errors='coerce')
            # If mostly successful, keep as numeric
            if s_num.notna().mean() > 0.5:
                df_clean[c] = s_num
            else:
                # Force category codes
                df_clean[c] = df_clean[c].astype(str).astype('category').cat.codes
        except:
             df_clean[c] = df_clean[c].astype(str).astype('category').cat.codes

    return df_clean

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    start_dt = datetime(2025, 1, 1)
    end_dt = datetime(2025, 12, 31)
    s_str = start_dt.strftime("%Y%m%d")
    e_str = end_dt.strftime("%Y%m%d")

    # 1. Load Data
    logger.info(f"Loading data for full year 2025 (2016-start history window)...")
    loader = JraVanDataLoader()
    # Deep History for consistent features
    df_db = loader.load(history_start_date="2016-01-01", end_date=e_str, skip_odds=True)
    
    if df_db.empty:
        logger.error("No data found in DB."); return
    
    # Meta merge preparation
    # Keep all columns from loader for reliable access later
    base_map = df_db.drop_duplicates(['race_id', 'horse_number'])
    
    # 2. Features
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    df_features = pipeline.load_features(df_db, list(pipeline.registry.keys()))
    if 'race_id' not in df_features.columns: df_features = df_features.reset_index()
    
    # Types cleanup
    df_features['race_id'] = df_features['race_id'].astype(str)
    base_map['race_id'] = base_map['race_id'].astype(str)
    df_features['horse_number'] = pd.to_numeric(df_features['horse_number'], errors='coerce').fillna(0).astype(int)
    base_map['horse_number'] = pd.to_numeric(base_map['horse_number'], errors='coerce').fillna(0).astype(int)
    
    # Filter only 2025 for report
    # Drop columns from features that exist in base_map to avoid suffixes
    cols_to_drop = [c for c in base_map.columns if c in df_features.columns and c not in ['race_id', 'horse_number']]
    df_features = df_features.drop(columns=cols_to_drop, errors='ignore')
    
    logger.info(f"Merging features ({df_features.shape}) with meta ({base_map.shape})...")
    df_full = pd.merge(df_features, base_map, on=['race_id', 'horse_number'], how='inner')
    df_full['date'] = pd.to_datetime(df_full['date'])
    df_target = df_full[(df_full['date'] >= start_dt) & (df_full['date'] <= end_dt)].copy()
    
    if df_target.empty:
        logger.error("No races found for 2025 target range."); return

    # 3. V13 Scores (Fixed OOF + Fallback logic)
    logger.info("Merging V13 Scores (OOF + Live Fallback)...")
    df_target = df_target.drop(columns=['prob_v13'], errors='ignore')
    
    # Load 2025 OOF
    oof_25_path = 'data/predictions/v13_oof_2025_clean.parquet'
    if os.path.exists(oof_25_path):
        df_oof = pd.read_parquet(oof_25_path).rename(columns={'pred_prob': 'prob_v13'})
        df_oof['race_id'] = df_oof['race_id'].astype(str)
        df_target = pd.merge(df_target, df_oof[['race_id', 'horse_number', 'prob_v13']], on=['race_id', 'horse_number'], how='left')
    
    # Live Fallback for missing rows
    missing_mask = df_target['prob_v13'].isna() if 'prob_v13' in df_target.columns else pd.Series(True, index=df_target.index)
    if missing_mask.any():
        logger.info(f"Predicting V13 Scores for {missing_mask.sum()} missing rows...")
        m_v13, f_v13 = load_v13_model_info(logger)
        df_v13_in = add_v13_odds_features(df_target[missing_mask].copy())
        
        # CLEANUP
        df_v13_in = clean_for_lightgbm(df_v13_in, f_v13)
        
        X_v13 = df_v13_in.reindex(columns=f_v13, fill_value=-999.0).fillna(-999.0)
        v13_scores = m_v13.predict_proba(X_v13)[:, -1] if hasattr(m_v13, 'predict_proba') else m_v13.predict(X_v13)
        df_target.loc[missing_mask, 'prob_v13'] = softmax_per_race(v13_scores, df_v13_in['race_id'].values)

    # 4. V14 Gaps
    logger.info("Predicting V14 Gaps...")
    m_v14 = joblib.load(V14_MODEL_PATH)
    f_v14 = pd.read_csv(V14_FEATS_PATH)['feature'].tolist()
    
    if 'odds_10min' not in df_target.columns: df_target['odds_10min'] = df_target['odds']
    df_target['odds_rank_10min'] = df_target.groupby('race_id')['odds_10min'].rank(method='min')
    pop_col = 'popularity' if 'popularity' in df_target.columns else 'tansho_ninki'
    df_target['rank_diff_10min'] = df_target.get(pop_col, 0) - df_target['odds_rank_10min']
    df_target['odds_log_ratio_10min'] = np.log(df_target['odds'] + 1e-9) - np.log(df_target['odds_10min'] + 1e-9)
    df_target['odds_ratio_60_10'] = 1.0; df_target['odds_60min'] = df_target['odds_10min']
    if 'odds_final' not in df_target.columns: df_target['odds_final'] = df_target['odds']
    
    # CLEANUP V14
    df_target = clean_for_lightgbm(df_target, f_v14)
    
    X_v14 = df_target.reindex(columns=f_v14, fill_value=0.0).fillna(0.0)
    df_target['gap_v14'] = m_v14.predict(X_v14)
    
    # Ranks
    df_target['rank_v13'] = df_target.groupby('race_id')['prob_v13'].rank(ascending=False, method='first')
    df_target['rank_v14'] = df_target.groupby('race_id')['gap_v14'].rank(ascending=False, method='first')

    # 5. Payouts (Year-scale)
    logger.info("Fetching Payouts for the whole year 2025...")
    from sqlalchemy import text
    q_hr = text("SELECT * FROM jvd_hr WHERE kaisai_nen = '2025'")
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

    # 6. Evaluation Logic
    report_data = []
    place_map = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
    
    race_ids = sorted(df_target['race_id'].unique())
    logger.info(f"Processing {len(race_ids)} races...")
    
    for rid in race_ids:
        rdf = df_target[df_target['race_id'] == rid]
        pout = payout_map.get(rid)
        if not pout: continue 
        
        # Combined Strategy: V13 Top 1 (Axis) + V14 Top 5 partners
        axis_row = rdf[rdf['rank_v13'] == 1].iloc[0]
        partners_raw = rdf[rdf['rank_v14'] <= 5].sort_values('rank_v14')
        partners = [int(p) for p in partners_raw['horse_number'].tolist() if p != axis_row['horse_number']]
        
        if not partners: continue
        uma_ret = 0; wide_ret = 0; hits = []
        for p in partners:
            pair = tuple(sorted((int(axis_row['horse_number']), int(p))))
            if pair in pout['uma']: 
                uma_ret += pout['uma'][pair]; hits.append(f"馬連{p}")
            if pair in pout['wide']: 
                wide_ret += pout['wide'][pair]; hits.append(f"ワイド{p}")
        
        report_data.append({
            'date': rdf['date'].iloc[0],
            'rid': rid,
            'month': rdf['date'].iloc[0].strftime("%Y-%m"),
            'place': place_map.get(rid[4:6], rid[4:6]),
            'race_class': categorize_race_class(rdf['grade_code'].iloc[0], rdf['kyoso_joken_code'].iloc[0]),
            'axis': axis_row['horse_number'],
            'partners': ",".join([str(p) for p in partners]),
            'inv': len(partners) * 200, # 100 for Umaren, 100 for Wide per partner
            'ret': uma_ret + wide_ret,
            'hits': ",".join(hits)
        })

    df_rep = pd.DataFrame(report_data)
    if df_rep.empty:
        logger.error("No valid results found."); return

    # 7. Generate MD Report
    logger.info(f"Generating Markdown report at {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write("# Annual Backtest Report: 2025\n\n")
        f.write("Generated using: **V13 Axis (Hard Weighted) x V14 Gap (Undervalued)**\n")
        f.write("- **Strategy**: Axis (V13 Rank 1) x Partners (V14 Rank Top 5)\n")
        f.write("- **Window**: Umaren (100 JPY) + Wide (100 JPY) per partner\n")
        f.write("- **History**: Deep History (2016-start) with cache consistency.\n\n")
        
        total_inv = df_rep['inv'].sum()
        total_ret = df_rep['ret'].sum()
        total_roi = (total_ret / total_inv * 100) if total_inv > 0 else 0
        
        f.write(f"## Overall Summary\n")
        f.write(f"| Races | Total Invest | Total Return | ROI |\n")
        f.write(f"|-------|--------------|--------------|-----|\n")
        f.write(f"| {len(df_rep):,} | ¥{total_inv:,} | ¥{total_ret:,} | **{total_roi:.1f}%** |\n\n")
        
        # Monthly
        f.write("## Monthly Summary\n")
        f.write("| Month | Races | Invest | Return | ROI |\n")
        f.write("|-------|-------|--------|--------|-----|\n")
        monthly = df_rep.groupby('month').agg({'rid':'count', 'inv':'sum', 'ret':'sum'})
        for m, r in monthly.iterrows():
            roi = (r['ret'] / r['inv'] * 100) if r['inv'] > 0 else 0
            f.write(f"| {m} | {r['rid']} | ¥{r['inv']:,} | ¥{r['ret']:,} | {roi:.1f}% |\n")
        f.write("\n")
        
        # Venue
        f.write("## Venue Summary\n")
        f.write("| Venue | Races | Invest | Return | ROI |\n")
        f.write("|-------|-------|--------|--------|-----|\n")
        venue = df_rep.groupby('place').agg({'rid':'count', 'inv':'sum', 'ret':'sum'})
        for v, r in venue.sort_values('ret', ascending=False).iterrows():
            roi = (r['ret'] / r['inv'] * 100) if r['inv'] > 0 else 0
            f.write(f"| {v} | {r['rid']} | ¥{r['inv']:,} | ¥{r['ret']:,} | {roi:.1f}% |\n")
        f.write("\n")
        
        # Class
        f.write("## Race Class Summary\n")
        f.write("| Class | Races | Invest | Return | ROI |\n")
        f.write("|-------|-------|--------|--------|-----|\n")
        rclass = df_rep.groupby('race_class').agg({'rid':'count', 'inv':'sum', 'ret':'sum'})
        for c, r in rclass.sort_values('ret', ascending=False).iterrows():
            roi = (r['ret'] / r['inv'] * 100) if r['inv'] > 0 else 0
            f.write(f"| {c} | {r['rid']} | ¥{r['inv']:,} | ¥{r['ret']:,} | {roi:.1f}% |\n")
        f.write("\n")
        
        # Detailed Results (Expandable)
        f.write("## Detailed Prediction Log\n")
        f.write("<details>\n<summary>Click to expand all race results</summary>\n\n")
        f.write("| Date | Place | R | Class | Axis | Partners | Invest | Return | Hits |\n")
        f.write("|------|-------|---|-------|------|----------|--------|--------|------|\n")
        for _, r in df_rep.iterrows():
            f.write(f"| {r['date'].strftime('%Y-%m-%d')} | {r['place']} | {r['rid'][10:12]} | {r['race_class']} | {r['axis']} | {r['partners']} | {r['inv']} | {r['ret']} | {r['hits']} |\n")
        f.write("\n</details>\n")
        
    logger.info("Done.")

if __name__ == "__main__":
    main()

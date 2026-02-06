
import sys
import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(message)s') # Simplified format for report
logger = logging.getLogger(__name__)

def load_data():
    pred_path_2025 = "models/experiments/exp_r3_ensemble/predictions_2025.parquet"
    t1_path = "data/temp_t1/T1_features_2024_2025.parquet"
    raw_2025_path = "data/temp_q1/year_2025.parquet"
    
    if not os.path.exists(pred_path_2025) or not os.path.exists(t1_path):
        raise FileNotFoundError("Data files missing.")
        
    df_preds = pd.read_parquet(pred_path_2025)
    df_preds['race_id'] = df_preds['race_id'].astype(str)
    
    df_features = pd.read_parquet(t1_path)
    df_features['race_id'] = df_features['race_id'].astype(str)
    
    df = pd.merge(df_preds, df_features, on=['race_id', 'horse_number'], how='inner')

    # Load Race Info for Segmentation
    if os.path.exists(raw_2025_path):
        try:
            df_info = pd.read_parquet(raw_2025_path).drop_duplicates(subset=['race_id'])
            df_info['race_id'] = df_info['race_id'].astype(str)
            df = pd.merge(df, df_info, on='race_id', how='left')
        except Exception as e:
            logger.warning(f"Could not load race info: {e}")

    # Ensure rank exists
    if 'rank' not in df.columns:
        if 'rank_x' in df.columns:
            df['rank'] = df['rank_x']
        elif 'rank_y' in df.columns:
            df['rank'] = df['rank_y']
        else:
            # Fallback based on prediction path logic? Usually rank is in preds
            logger.error("Rank column missing from merged df.")
            raise KeyError("rank")

    # Load Meta Model
    model_path = "models/experiments/exp_t1_meta/meta_lgbm.txt"
    if os.path.exists(model_path):
        model = lgb.Booster(model_file=model_path)
        df['prob_market_10min'] = 0.8 / (df['odds_10min'] + 1e-9)
        df['prob_diff_10min'] = df['pred_prob'] - df['prob_market_10min']
        # Handle NA
        df['odds_10min'] = df['odds_10min'].fillna(100) 
        df['prob_market_10min'] = df['prob_market_10min'].fillna(0)
        df['prob_diff_10min'] = df['prob_diff_10min'].fillna(0)
        
        feature_cols = ['pred_prob', 'odds_10min', 'prob_market_10min', 'prob_diff_10min']
        df['meta_prob'] = model.predict(df[feature_cols])
    else:
        logger.warning("Meta model missing! Using base prob.")
        df['meta_prob'] = df['pred_prob']

    # Final Odds
    if 'odds_final' in df.columns:
        df['odds_calc'] = df['odds_final']
    elif 'odds' in df.columns:
        df['odds_calc'] = df['odds']
    else:
        df['odds_calc'] = 0
        
    return df

def analyze_quadrants(df):
    logger.info("\n=== 1. AI vs Market Quadrants Analysis ===")
    logger.info("Segmentation by AI Confidence (Meta Prob) vs Market Confidence (10min Odds)")
    
    def get_ai_label(p):
        if p < 0.1: return "AI:Low"
        elif p < 0.3: return "AI:Mid"
        else: return "AI:High"
        
    def get_market_label(o):
        if pd.isna(o): return "Mkt:Unknown"
        if o < 5.0: return "Mkt:Fav"
        elif o < 20.0: return "Mkt:Mid"
        else: return "Mkt:Long"
        
    df['ai_seg'] = df['meta_prob'].apply(get_ai_label)
    df['mkt_seg'] = df['odds_10min'].apply(get_market_label)
    
    grouped = df.groupby(['ai_seg', 'mkt_seg'], observed=False)
    
    results = []
    for (ai, mkt), group in grouped:
        n = len(group)
        hits = group[group['rank'] == 1]
        ret = (hits['odds_calc'] * 100).sum()
        cost = n * 100
        roi = ret / cost if cost > 0 else 0
        hit_rate = len(hits) / n if n > 0 else 0
        results.append({
            'Segment': f"{ai} x {mkt}",
            'Bets': n,
            'Hit%': hit_rate * 100,
            'ROI%': roi * 100,
            'Profit': ret - cost
        })
        
    res_df = pd.DataFrame(results).sort_values('ROI%', ascending=False)
    print(res_df.to_string(index=False, float_format="{:.1f}".format))

def analyze_segments(df):
    logger.info("\n=== 2. Race Segment Analysis (Class/Distance/Surface) ===")
    
    candidates = df[df['meta_prob'] >= 0.15].copy()
    logger.info(f"Analyzing candidates (Prob >= 0.15): {len(candidates)} bets")
    
    segments = ['grade_code', 'race_type_code', 'course_id_x', 'distance_x'] 
    # Note: merge might produce _x/_y suffix if join keys overlap or common cols exist
    # Check columns
    cols = df.columns.tolist()
    
    # Adjust column names dynamically
    target_segs = []
    for seg in ['grade_code', 'race_type_code', 'distance', 'course_id']:
        if seg in cols: target_segs.append(seg)
        elif f"{seg}_y" in cols: target_segs.append(f"{seg}_y") # Prefer info from race_info join
        elif f"{seg}_x" in cols: target_segs.append(f"{seg}_x")
        
    for seg in target_segs:
        logger.info(f"\n--- Segment: {seg} ---")
        grouped = candidates.groupby(seg, observed=False)
        
        results = []
        for name, group in grouped:
            n = len(group)
            if n < 30: continue 
            
            hits = group[group['rank'] == 1]
            ret = (hits['odds_calc'] * 100).sum()
            cost = n * 100
            roi = ret / cost
            hit_rate = len(hits) / n
            
            results.append({
                'Value': str(name),
                'Bets': n,
                'Hit%': hit_rate * 100,
                'ROI%': roi * 100,
                'Profit': ret - cost
            })
            
        print(pd.DataFrame(results).sort_values('ROI%', ascending=False).to_string(index=False, float_format="{:.1f}".format))

def analyze_value_gap(df):
    logger.info("\n=== 3. Value Gap Analysis (EV based on Meta Prob) ===")
    
    df['EV_est'] = df['meta_prob'] * df['odds_10min']
    
    bins = [0.8, 1.0, 1.2, 1.5, 2.0, 5.0]
    df['ev_bin'] = pd.cut(df['EV_est'], bins=bins)
    
    grouped = df.groupby('ev_bin', observed=False)
    
    results = []
    for name, group in grouped:
        n = len(group)
        hits = group[group['rank'] == 1]
        ret = (hits['odds_calc'] * 100).sum() 
        cost = n * 100
        roi = ret / cost if cost > 0 else 0
        hit_rate = len(hits) / n if n > 0 else 0
        
        results.append({
            'Est EV Range': str(name),
            'Bets': n,
            'Act Hit%': hit_rate * 100,
            'Act ROI%': roi * 100,
            'Profit': ret - cost
        })
        
    print(pd.DataFrame(results).to_string(index=False, float_format="{:.1f}".format))

def main():
    try:
        df = load_data()
    except KeyError as e:
        logger.error(f"Data Load Error: {e}")
        return

    with open("strategy_deep_dive_2025.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        
        try:
            print(f"Data Loaded: {len(df)} rows")
            analyze_quadrants(df)
            analyze_segments(df)
            analyze_value_gap(df)
        finally:
            sys.stdout = original_stdout
            
    print("Analysis complete. Saved to strategy_deep_dive_2025.txt")

if __name__ == "__main__":
    main()

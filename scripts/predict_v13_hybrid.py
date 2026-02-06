
"""
Hybrid Prediction Script (Deep Value + Gap Reg)
===============================================
Deploys the Hybrid Strategy:
1. Load Deep Value Model (Safe) -> Predicts Rank
2. Load Gap Reg Model (Aggressive) -> Predicts Gap (Undervaluation)
3. Combine signals to output "Gap Picks" with "Safety Check".

Usage:
  python scripts/predict_v13_hybrid.py --date 20240101 --discord
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import argparse
import datetime
import logging
from src.preprocessing.feature_pipeline import FeaturePipeline
from src.utils.logger import get_logger
from src.utils.database import get_db_engine

# Config
MODEL_DIR_SAFE = 'models/experiments/exp_lambdarank_hard_weighted'
MODEL_DIR_GAP = 'models/experiments/exp_gap_prediction_reg'
OUTPUT_DIR = 'predictions/v13_hybrid'

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='YYYYMMDD')
    parser.add_argument('--discord', action='store_true', help='Send to Discord')
    args = parser.parse_args()
    
    date_str = args.date
    target_date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    
    logger.info(f"🚀 Starting Hybrid Prediction for: {date_str}")
    
    # 1. Load Models & Features
    logger.info("Loading models...")
    model_safe = joblib.load(f'{MODEL_DIR_SAFE}/model.pkl')
    model_gap = joblib.load(f'{MODEL_DIR_GAP}/model.pkl')
    
    # Load feature lists
    try:
        feats_safe = pd.read_csv(f'{MODEL_DIR_SAFE}/features.csv')['0'].tolist()
        feats_gap = pd.read_csv(f'{MODEL_DIR_GAP}/features.csv')['0'].tolist()
    except Exception as e:
        logger.error(f"Failed to load feature lists: {e}")
        return

    # 2. Pipeline: Load Data & Features
    pipeline = FeaturePipeline()
    # Use standard JIT pipeline
    # Note: feature_blocks must match training.
    # Training of both models used v12/v13 set.
    # We load standard Blocks.
    # Crucially, we must ensure 'yoso_juni' is available if used (Deep Value uses it? No, Deep Value removed it).
    # Gap model removed yoso too.
    # So we just need standard features.
    
    # Pipeline execution
    # 1. Load Raw (JRA-VAN)
    df_raw = pipeline.load_raw_data(target_date)
    if df_raw.empty:
        logger.warning("No races found.")
        return
        
    # 2. Live Odds (for popularity/odds features)
    df_live_odds = pipeline.load_live_odds(target_date) # Optional: usage logic
    
    feature_blocks = [
        'base_attributes',
        'bloodline_detail',
        'weight_patterns',
        'jockey_profile',
        'trainer_profile',
        'rotation_stats',
        'field_context',
        'race_context',
        'weather_cond',
        'course_bias',
        'pace_stats',
        'habit_stats',
        'corner_stats',
        'bms_stats',
        'head_to_head',
        'aptitude_smoothing',
        'burden_stats',
        'changes_stats',
        'aptitude_stats',
        'frame_bias',
        'horse_gear',
        'mining_features' # Loaded even if not used by model, to support pipeline integrity
    ]
    
    df_features = pipeline.load_features(df_raw, feature_blocks)
    
    # 3. Pre-process for Model (Ad-hoc Features)
    # Both models expect: odds_rank_vs_elo, is_high_odds, is_mid_odds
    # And NO yoso features.
    
    df_today = df_features.copy()
    
    # Merge Live Odds if available, else usage previous odds (from raw?)
    # df_raw usually has zero odds until race over, unless 'apd_sokuho_o1' is merged.
    # Pipeline's `load_features` usually merges odds if available?
    # Actually `MiningFeatureGenerator` might need odds if we used it.
    # Let's manually merge odds from df_live_odds if present.
    if not df_live_odds.empty:
         # Merge on race_id, horse_number
         # Rename 'odds_live' to 'odds'
         odds_data = df_live_odds[['race_id', 'horse_number', 'odds_live']].rename(columns={'odds_live': 'odds'})
         df_today = df_today.merge(odds_data, on=['race_id', 'horse_number'], how='left')
    else:
         logger.warning("Live odds not available. using default odds=10.0 for prediction feature generation.")
         df_today['odds'] = 10.0 # Placeholder
         
    # Fill nan odds
    df_today['odds'] = df_today['odds'].fillna(10.0)
    
    # Feature Engineering (Ad-hoc)
    df_today['odds_rank'] = df_today.groupby('race_id')['odds'].rank(ascending=True)
    
    if 'relative_horse_elo_z' in df_today.columns:
        df_today['elo_rank'] = df_today.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
        df_today['odds_rank_vs_elo'] = df_today['odds_rank'] - df_today['elo_rank']
    else:
        df_today['odds_rank_vs_elo'] = 0
        
    df_today['is_high_odds'] = (df_today['odds'] >= 10).astype(int)
    df_today['is_mid_odds'] = ((df_today['odds'] >= 5) & (df_today['odds'] < 10)).astype(int)
    
    # 4. Predict
    # Make sure we select columns matching model
    X_safe = df_today[feats_safe].fillna(0) # Simple imputation
    X_gap = df_today[feats_gap].fillna(0)
    
    pred_safe = model_safe.predict(X_safe)
    pred_gap = model_gap.predict(X_gap)
    
    df_today['score_safe'] = pred_safe
    df_today['score_gap'] = pred_gap
    
    # Calculate Ranks per race
    df_today['rank_safe'] = df_today.groupby('race_id')['score_safe'].rank(ascending=False)
    df_today['rank_gap'] = df_today.groupby('race_id')['score_gap'].rank(ascending=False)
    
    # 5. Output Generation
    race_ids = df_today['race_id'].unique()
    
    output_lines = []
    output_lines.append(f"Hybrid Prediction (V13) for {date_str}")
    output_lines.append(f"Model Safe: {MODEL_DIR_SAFE}")
    output_lines.append(f"Model Gap:  {MODEL_DIR_GAP}")
    output_lines.append("-" * 40)
    
    for rid in sorted(race_ids):
        race_df = df_today[df_today['race_id'] == rid].sort_values('rank_gap') # Sort by GAP (Aggressive)
        
        # Meta info
        race_meta = df_raw[df_raw['race_id'] == rid].iloc[0]
        venue = race_meta.get('venue', 'Unknown')
        rno = race_meta.get('race_number', '??')
        name = race_meta.get('race_name', 'Race')
        
        output_lines.append(f"\n[{venue} {rno}R] {name}")
        output_lines.append(f"{'No':<4} {'Name':<12} {'Odds':<6} {'GapRank':<8} {'SafeRank':<8} {'Action'}")
        output_lines.append("-" * 60)
        
        for _, row in race_df.iterrows():
            hno = int(row['horse_number'])
            hname = str(row.get('horse_name', f"H{hno}"))[:12]
            odds = row['odds']
            r_gap = int(row['rank_gap'])
            r_safe = int(row['rank_safe'])
            
            action = ""
            # Signal Logic
            if r_gap == 1:
                action += "★ GAP TOP"
                if r_safe <= 5:
                    action += " (Safe Ok)"
                else:
                    action += " (High Risk)"
            elif r_gap <= 3:
                action += "Gap Top 3"
                if r_safe == 1:
                    action += " + Safe Top 1 (Strong)"
                    
            if r_safe == 1 and r_gap > 3:
                action += "Safe Top 1 (Low Value)"
                
            output_lines.append(f"{hno:02d}   {hname:<12} {odds:>5.1f}  {r_gap:<8d} {r_safe:<8d} {action}")
            
    final_output = "\n".join(output_lines)
    print(final_output)
    
    # Save text
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f'{OUTPUT_DIR}/pred_{date_str}.txt', 'w', encoding='utf-8') as f:
        f.write(final_output)
        
    logger.info("Done.")
    
    if args.discord:
        # TODO: Implement Discord webhook
        pass

if __name__ == '__main__':
    main()

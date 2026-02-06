#!/usr/bin/env python
"""
Compare predictions between ID-based and No-IDs models for today's races.
Usage: python scripts/adhoc/compare_today_predictions.py --date YYYYMMDD
"""
import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model paths
MODEL_WITH_IDS = "models/experiments/exp_t2_track_bias/model.pkl"
MODEL_NO_IDS = "models/experiments/exp_t2_no_ids/model.pkl"

TEMP_FEATURE_DIR = "data/jit_compare_temp"

JIT_BLOCKS = [
    'base_attributes',
    'history_stats',
    'jockey_stats',
    'pace_stats',
    'pace_pressure_stats',
    'relative_stats',
    'jockey_trainer_stats',
    'bloodline_stats',
    'training_stats',
    'burden_stats',
    'changes_stats',
    'aptitude_stats',
    'speed_index_stats',
    'class_stats',
    'runstyle_fit',
    'risk_stats'
]

def load_today_data(loader, target_date):
    """Load today's races and horse history."""
    logger.info(f"Loading races for {target_date}")
    
    # Format date
    target_date_formatted = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    
    # Load today's races
    df_today = loader.load(
        history_start_date=target_date_formatted,
        end_date=target_date_formatted,
        jra_only=True
    )
    
    if df_today.empty:
        logger.warning("No races found for today")
        return None
        
    logger.info(f"Found {len(df_today)} horses in {df_today['race_id'].nunique()} races")
    
    # Get horse IDs
    horse_ids = df_today['horse_id'].unique().tolist()
    
    # Load horse history
    history_start = (pd.to_datetime(target_date) - timedelta(days=365*5)).strftime("%Y-%m-%d")
    history_end = (pd.to_datetime(target_date) - timedelta(days=1)).strftime("%Y-%m-%d")
    
    df_history = loader.load(
        history_start_date=history_start,
        end_date=history_end,
        horse_ids=horse_ids,
        jra_only=True
    )
    
    logger.info(f"Loaded {len(df_history)} history records")
    
    # Combine
    df_combined = pd.concat([df_history, df_today], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['race_id', 'horse_number'])
    df_combined = df_combined.sort_values(['horse_id', 'date'])
    
    return df_combined, df_today

def generate_features(df_combined, df_today, target_date):
    """Generate features using JIT blocks."""
    os.makedirs(TEMP_FEATURE_DIR, exist_ok=True)
    
    pipeline = FeaturePipeline(cache_dir=TEMP_FEATURE_DIR)
    df_features = pipeline.load_features(df_combined, JIT_BLOCKS)
    
    # Merge back date from original data
    df_with_date = df_combined[['race_id', 'horse_number', 'date']].drop_duplicates()
    df_features = df_features.merge(df_with_date, on=['race_id', 'horse_number'], how='left')
    
    # Filter to today's races only
    df_features['date'] = pd.to_datetime(df_features['date']).dt.strftime('%Y-%m-%d')
    target_date_str = pd.to_datetime(target_date).strftime('%Y-%m-%d')
    df_today_feats = df_features[df_features['date'] == target_date_str].copy()
    
    # Cleanup
    shutil.rmtree(TEMP_FEATURE_DIR, ignore_errors=True)
    
    return df_today_feats

def predict_with_model(df_features, model_path, exclude_ids=False):
    """Run predictions with a model."""
    model = joblib.load(model_path)
    feats = model.feature_name()
    
    # Fill missing features
    for f in feats:
        if f not in df_features.columns:
            df_features[f] = 0
    
    X = df_features[feats].copy()
    
    # Convert types
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            X[col] = pd.to_numeric(X[col].astype(str), errors='coerce').fillna(0)
        X[col] = X[col].fillna(0)
    
    preds = model.predict(X.values)
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y%m%d"),
                        help="Target date YYYYMMDD")
    args = parser.parse_args()
    
    target_date = args.date
    logger.info(f"=== Comparing Models for {target_date} ===")
    
    # Load data
    loader = JraVanDataLoader()
    result = load_today_data(loader, target_date)
    
    if result is None:
        logger.error("No data available")
        return
    
    df_combined, df_today = result
    
    # Generate features
    logger.info("Generating features...")
    df_features = generate_features(df_combined, df_today, target_date)
    logger.info(f"Generated features for {len(df_features)} horses")
    
    # Predictions
    logger.info("Running predictions...")
    df_features['pred_with_ids'] = predict_with_model(df_features.copy(), MODEL_WITH_IDS)
    df_features['pred_no_ids'] = predict_with_model(df_features.copy(), MODEL_NO_IDS)
    
    # Normalize per race
    df_features['prob_with_ids'] = df_features.groupby('race_id')['pred_with_ids'].transform(
        lambda x: x / x.sum()
    )
    df_features['prob_no_ids'] = df_features.groupby('race_id')['pred_no_ids'].transform(
        lambda x: x / x.sum()
    )
    
    # Rank within race
    df_features['rank_with_ids'] = df_features.groupby('race_id')['pred_with_ids'].rank(ascending=False)
    df_features['rank_no_ids'] = df_features.groupby('race_id')['pred_no_ids'].rank(ascending=False)
    
    # Print comparison per race
    print("\n" + "="*80)
    print(f"MODEL COMPARISON FOR {target_date}")
    print("="*80)
    
    races = df_features['race_id'].unique()
    for race_id in sorted(races):
        race = df_features[df_features['race_id'] == race_id].copy()
        race = race.sort_values('rank_no_ids')
        
        print(f"\n--- Race: {race_id} ({len(race)} horses) ---")
        print(f"{'馬番':<6} {'With IDs':<12} {'No IDs':<12} {'Rank差':<8}")
        print("-" * 40)
        
        for _, row in race.head(5).iterrows():
            horse_num = int(row['horse_number'])
            rank_with = int(row['rank_with_ids'])
            rank_no = int(row['rank_no_ids'])
            prob_with = row['prob_with_ids'] * 100
            prob_no = row['prob_no_ids'] * 100
            rank_diff = rank_with - rank_no
            diff_str = f"+{rank_diff}" if rank_diff > 0 else str(rank_diff)
            
            print(f"{horse_num:<6} {prob_with:>5.1f}% ({rank_with}位) {prob_no:>5.1f}% ({rank_no}位)  {diff_str}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Count rank changes
    df_features['rank_diff'] = abs(df_features['rank_with_ids'] - df_features['rank_no_ids'])
    
    corr = df_features['pred_with_ids'].corr(df_features['pred_no_ids'])
    avg_rank_diff = df_features['rank_diff'].mean()
    same_top1 = ((df_features['rank_with_ids'] == 1) & (df_features['rank_no_ids'] == 1)).sum()
    total_races = len(races)
    
    print(f"Total Races: {total_races}")
    print(f"Prediction Correlation: {corr:.4f}")
    print(f"Avg Rank Difference: {avg_rank_diff:.2f}")
    print(f"Same Top-1 Pick: {same_top1}/{total_races} ({100*same_top1/total_races:.1f}%)")
    
    # Save results
    output_path = f"reports/compare_models_{target_date}.csv"
    os.makedirs("reports", exist_ok=True)
    df_features[['race_id', 'horse_number', 'pred_with_ids', 'pred_no_ids', 
                 'prob_with_ids', 'prob_no_ids', 'rank_with_ids', 'rank_no_ids']].to_csv(
        output_path, index=False
    )
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

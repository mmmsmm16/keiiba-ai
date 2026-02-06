
import sys
import os
import logging
import argparse
import pandas as pd
import yaml
import traceback

# Ensure src in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate Features for T2")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    start_date = cfg['dataset']['train_start_date']
    end_date = cfg['dataset']['test_end_date']
    feature_blocks = cfg['features']
    
    logger.info(f"🚀 Generating Features for {cfg['experiment_name']}")
    logger.info(f"   Period: {start_date} ~ {end_date}")
    logger.info(f"   Features: {feature_blocks}")
    
    loader = JraVanDataLoader()
    pipeline = FeaturePipeline(cache_dir="data/features") # Use shared cache
    
    # Load Base Data
    logger.info("Loading Base Data (JraVan)...")
    
    # Check if odds features are needed
    needs_odds = any('odds' in block.lower() for block in feature_blocks)
    skip_odds = not needs_odds
    if skip_odds:
        logger.info("   Skipping odds time series (no odds features requested)")
    
    try:
        df = loader.load(
            history_start_date=start_date, 
            end_date=end_date, 
            skip_training=False,
            skip_odds=skip_odds  # Skip slow odds loading if not needed
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        traceback.print_exc()
        return

    if df.empty:
        logger.error("Loaded DF is empty!")
        return
        
    logger.info(f"Loaded {len(df)} rows.")
    
    # Generate/Load Features
    logger.info("Pipeline: Loading/Computing Features...")
    try:
        df_features = pipeline.load_features(df, feature_blocks)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        return
        
    # Save Combined Features
    os.makedirs("data/temp_t2", exist_ok=True)
    feat_path = "data/temp_t2/T2_features.parquet"
    df_features.to_parquet(feat_path)
    logger.info(f"Saved Features to {feat_path}")
    
    # Save Targets (Rank/Win) for Training Script convenience
    # Usually Training script re-merges, but let's save a target file compatible with Q-series scripts
    tgt_path = "data/temp_t2/T2_targets.parquet"
    
    # Target columns usually: race_id, horse_number, rank, payoff, time, etc.
    # Keep metadata for analysis
    tgt_cols = ['race_id', 'horse_number', 'rank', 'date', 'odds_10min', 'odds_final'] 
    # Add other useful columns if present like 'is_win', 'is_top3' if precalculated
    
    # Ensure columns exist
    avail_tgt_cols = [c for c in tgt_cols if c in df.columns]
    df_targets = df[avail_tgt_cols].copy()
    
    df_targets.to_parquet(tgt_path)
    logger.info(f"Saved Targets to {tgt_path}")

if __name__ == "__main__":
    main()

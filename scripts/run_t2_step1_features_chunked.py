
import sys
import os
import logging
import argparse
import pandas as pd
import yaml
import traceback
import gc

sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate Features for T2 (Chunked)")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    start_date = pd.to_datetime(cfg['dataset']['train_start_date'])
    end_date = pd.to_datetime(cfg['dataset']['test_end_date'])
    feature_blocks = cfg['features']
    
    start_year = start_date.year
    end_year = end_date.year
    
    logger.info(f"🚀 Generating Features (Chunked) for {cfg['experiment_name']}")
    logger.info(f"   Period: {start_year} ~ {end_year}")
    
    loader = JraVanDataLoader()
    pipeline = FeaturePipeline(cache_dir="data/features")
    
    all_dfs = []
    all_targets = [] # Collect targets from raw loader output
    
    for year in range(start_year, end_year + 1):
        logger.info(f"Processing Year: {year}")
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31"
        
        # Clip to config dates
        if year == start_year:
            y_start = str(start_date.date())
        if year == end_year:
            y_end = str(end_date.date())
            
        try:
            # Skip odds because Base Model features don't use 10min odds (odds_features not in list)
            # This significantly speeds up loading.
            df = loader.load(history_start_date=y_start, end_date=y_end, skip_training=False, skip_odds=True)
            if df.empty:
                logger.warning(f"No data for {year}")
                continue
            
            # Save targets from raw loader output BEFORE pipeline
            tgt_cols_avail = [c for c in ['race_id', 'horse_number', 'rank', 'date'] if c in df.columns]
            all_targets.append(df[tgt_cols_avail].copy())
                
            logger.info(f"  Loaded {len(df)} rows. Pipeline processing...")
            df_feat = pipeline.load_features(df, feature_blocks)
            
            all_dfs.append(df_feat)
            
            del df
            del df_feat
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error in {year}: {e}")
            traceback.print_exc()
            
    if not all_dfs:
        logger.error("No data collected!")
        return
        
    logger.info("Concatenating all years...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save Combined Features
    os.makedirs("data/temp_t2", exist_ok=True)
    feat_path = "data/temp_t2/T2_features.parquet"
    full_df.to_parquet(feat_path)
    logger.info(f"Saved Features to {feat_path}")
    
    # Save Targets (collected from raw loader outputs)
    if all_targets:
        full_targets = pd.concat(all_targets, ignore_index=True)
        tgt_path = "data/temp_t2/T2_targets.parquet"
        full_targets.to_parquet(tgt_path)
        logger.info(f"Saved Targets to {tgt_path}")
    else:
        logger.error("No targets collected!")

if __name__ == "__main__":
    main()

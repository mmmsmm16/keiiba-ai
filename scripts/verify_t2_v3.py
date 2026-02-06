
import os
import sys
import yaml
import pandas as pd
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline

def setup_logger():
    logger = logging.getLogger('Verification')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def run_verification():
    logger = setup_logger()
    logger.info("Starting T2 Feature Pipeline Verification (Dry Run)...")
    
    # Load Config
    config_path = "config/experiments/exp_t2_refined_v3.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded config: {config['experiment_name']}")
    
    # Select feature blocks
    selected_features = config.get('features', [])
    logger.info(f"Target Feature Blocks: {len(selected_features)}")
    
    # Load Sample Data (Small)
    loader = JraVanDataLoader()
    # Use recent data
    df = loader.load(limit=2000, history_start_date="2023-01-01", jra_only=True)
    logger.info(f"Loaded sample data: {len(df)} rows")
    
    if df.empty:
        logger.error("No data loaded. Validation failed.")
        return

    # Initialize Pipeline
    pipeline = FeaturePipeline(cache_dir="data/features_test")
    
    # Run Blocks Manual check or Pipeline method? (Pipeline usually has run() or similar, 
    # but FeaturePipeline usually manages cache. Here we simulate 'create_features' logic.)
    
    # We iterate blocks and run them
    merged_df = df.copy()
    
    # Mock registry access if needed, or use public method?
    # FeaturePipeline usually doesn't expose 'compute_all' directly? 
    # Let's inspect feature_pipeline.py usage in `train_model.py` usually?
    # Or just instantiate and call private methods if needed, or use a runner script?
    # Looking at pipeline code (Step 14):
    # It has registry.
    # We can iterate registry and run if in selected_features.
    
    # Also need to run standard blocks first?
    # Base attributes etc.
    
    keys = ['race_id', 'horse_number', 'horse_id']
    
    for block_name in selected_features:
        if block_name in pipeline.registry:
            logger.info(f"Running block: {block_name}")
            try:
                func = pipeline.registry[block_name]
                feat_df = func(merged_df) # Some blocks need raw df? Or updated df? 
                # Most blocks in this pipeline design take the 'base df' (raw data) and return features.
                # However, some might depend on previously calculated columns (e.g. pace_pressure using last_nige_rate).
                # If dependencies exist, we should merge results back to merged_df?
                # The design pattern in this project:
                # usually `df` passed to blocks is the RAW data.
                # BUT if blocks adhere to "Output only keys + new feats", then we must merge to accumulate.
                # AND if a block *needs* a feature from another block, it must be in the input `df`.
                # So we MUST merge results back to `merged_df` progressively.
                
                # Check columns overlap
                new_cols = [c for c in feat_df.columns if c not in keys]
                logger.info(f"  -> Generated {len(new_cols)} features. (e.g. {new_cols[:3]})")
                
                # Merge
                merged_df = pd.merge(merged_df, feat_df.drop(columns=[c for c in feat_df.columns if c in merged_df.columns and c not in keys], errors='ignore'), 
                                     on=keys, how='left')
                                     
            except Exception as e:
                logger.error(f"Block {block_name} failed: {e}")
                # Print output for debug
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f"Block {block_name} not registered!")

    # Final Check
    logger.info(f"Final Column Count: {len(merged_df.columns)}")
    
    # Check specific new features
    check_cols = ['horse_elo', 'field_size', 'impost_change', 'weather_code', 'track_variant', 'speed_index_ewm_5']
    for c in check_cols:
        if c in merged_df.columns:
            nulls = merged_df[c].isnull().mean()
            logger.info(f"Feature {c}: Exists. Null Rate={nulls:.2%}")
        else:
            logger.warning(f"Feature {c}: MISSING")

    logger.info("Verification Done.")

if __name__ == "__main__":
    run_verification()

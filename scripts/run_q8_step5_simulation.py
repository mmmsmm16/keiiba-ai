
import os
import sys
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    EXP_NAME = "exp_q8_freshness"
    YEAR = 2024
    
    logger.info(f"🚀 Starting ROI Simulation for {EXP_NAME} (Year {YEAR})")
    
    # Paths
    MODEL_DIR = f"models/experiments/{EXP_NAME}"
    MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
    CONFIG_PATH = os.path.join(MODEL_DIR, "config_copy.yaml") # Use the config saved with model
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        return
        
    # Load Model
    logger.info("Loading Model...")
    model = joblib.load(MODEL_PATH)
    
    # Load Config to check used features
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load Features for 2024
    # We check if Q8 features exist, otherwise we might need to rely on Q11/Q12 cache mostly?
    # Actually Q8 features are unique to Q8. 
    # Try loading from temp if available
    FEAT_PATH = "data/temp_q8/Q8_features.parquet"
    
    if not os.path.exists(FEAT_PATH):
        logger.warning(f"{FEAT_PATH} not found. Checking if we can load from Q12 but filter cols?")
        # Q8 features are subset of Q12? 
        # Q8: +Interval Aptitude.
        # Q9+: +Physique.
        # So Q12 has Q8 cols. We can load Q12 features and select Q8 columns.
        FEAT_PATH_ALT = "data/temp_q12/Q12_features.parquet"
        if os.path.exists(FEAT_PATH_ALT):
            logger.info(f"Using alternate feature file: {FEAT_PATH_ALT}")
            FEAT_PATH = FEAT_PATH_ALT
        else:
             logger.error("No suitable feature file found. Please re-run feature generation.")
    logger.info(f"Loading features from {FEAT_PATH}")
    df_feat = pd.read_parquet(FEAT_PATH)
    
    # Load Targets for Date/Rank
    TGT_PATH = "data/temp_q8/Q8_targets.parquet"
    if not os.path.exists(TGT_PATH):
        logger.error(f"{TGT_PATH} not found.")
        return
        
    logger.info(f"Loading targets from {TGT_PATH}")
    df_tgt = pd.read_parquet(TGT_PATH)
    
    # Merge Date/Rank from Targets
    # Keys: race_id, horse_number
    # (assuming indexes align or merge)
    if 'date' not in df_feat.columns:
        cols_to_merge = ['race_id', 'horse_number', 'date', 'rank']
        df_feat = pd.merge(df_feat, df_tgt[cols_to_merge], on=['race_id', 'horse_number'], how='left')

    # Filter for Year
    if 'date' in df_feat.columns:
        df_feat['date'] = pd.to_datetime(df_feat['date'])
        df_feat = df_feat[df_feat['date'].dt.year == YEAR].copy()
    else:
        logger.error("Date column missing in features")
        return
        
    logger.info(f"Loaded {len(df_feat)} rows for {YEAR}")
    
    # Identify Model Features
    # LGBM model usually has feature_name()
    model_feats = model.feature_name()
    logger.info(f"Model expects {len(model_feats)} features.")
    
    # Cast Categorical Features (Match Train Logic)
    # Only cast object columns to category. 
    # Numeric categoricals (like ids) should remain numeric as trained.
    for col in df_feat.columns:
        if df_feat[col].dtype == 'object':
             df_feat[col] = df_feat[col].astype('category')
    
    # Also ensure all model features exist
    missing = [c for c in model_feats if c not in df_feat.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")
        for c in missing:
            df_feat[c] = np.nan
            
    X = df_feat[model_feats]
    
    # Predict
    logger.info("Predicting...")
    preds = model.predict(X)
    df_feat['pred_prob'] = preds
    
    # Load Odds / Results
    # We need: race_id, horse_number, rank, odds (tansho)
    # If not in df_feat, load from raw
    
    # Simple check if odds in feat
    if 'odds' not in df_feat.columns or 'rank' not in df_feat.columns:
        logger.info("Loading Raw Data for Odds/Results...")
        # Recalling the raw loader logic roughly? 
        # Or just read from a raw cache if we have one. step1 usually loads it.
        # Let's try loading 'Q1_targets' or something?
        # Actually standard JRA loader is best.
        loader = JraVanDataLoader()
        # raw load is heavy.
        # Try finding a targets file that might have odds? Q8_targets usually only has rank.
        
        # Load specific 2024 raw file
        raw_2024_path = "data/temp_q1/year_2024.parquet"
        if os.path.exists(raw_2024_path):
            df_raw = pd.read_parquet(raw_2024_path)
            # Merge
            cols_to_merge = ['race_id', 'horse_number', 'odds', 'rank']
            # rank might be in both, careful
            if 'rank' in df_feat.columns: cols_to_merge.remove('rank')
            
            # Merge
            df_feat = pd.merge(df_feat, df_raw[cols_to_merge], on=['race_id', 'horse_number'], how='left')
        else:
            logger.error(f"Raw 2024 file not found: {raw_2024_path}")
            return
            
    # Simulation Logic
    # 1. Rank Prediction (Sort by prob)
    df_feat['pred_rank'] = df_feat.groupby('race_id')['pred_prob'].rank(ascending=False, method='first')
    
    # 2. Win Strategy
    # Bet on Top 1
    # ROI = Returns / Cost
    
    # Top 1 ROI
    top1 = df_feat[df_feat['pred_rank'] == 1]
    n_races = len(top1)
    cost = n_races * 100 # 100 yen per race
    
    # Returns
    # Win if rank=1
    # Payout = 100 * odds
    wins = top1[top1['rank'] == 1]
    return_amount = (wins['odds'] * 100).sum()
    
    roi = return_amount / cost if cost > 0 else 0
    hit_rate = len(wins) / n_races if n_races > 0 else 0
    
    logger.info(f"=== ROI Simulation (2024) ===")
    logger.info(f"Strategy: Flat Bet on Top 1 Prediction")
    logger.info(f"Races: {n_races}")
    logger.info(f"Hit Rate: {hit_rate:.4f} ({len(wins)}/{n_races})")
    logger.info(f"Cost: {cost} JPY")
    logger.info(f"Return: {return_amount:.0f} JPY")
    logger.info(f"ROI: {roi*100:.2f}%")
    
    # Strategy: Threshold > 0.3?
    # ... (Can add more)
    
    # Strategy: Box Top 1 (Just checking if odds help)
    # If odds < X, skip?
    
    # Strategy: Bet if Prob > Odds Implied Prob * 1.2 (Value Betting)
    # Implied Prob = 1 / Odds * 0.8 (Take)
    # Let's keep it simple first.
    
if __name__ == "__main__":
    main()

import os
import sys
import gc
import argparse
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from scipy.special import logit, expit
from dateutil.relativedelta import relativedelta

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.scripts.auto_predict_v13 import get_db_engine

# Safe Feature List (Excluding Price)
DROP_COLS = [
    'race_id', 'target', 'date', 'month', 'rank', 'rank_result', 'kakutei_chakujun',
    'odds', 'tansho_odds', 'popularity', 'final_odds', 'final_popularity',
    'payout', 'time', 'agari', 'logit_final_odds', 'logit_t10_odds',
    'p_market_tminus10m', 'prob_residual_softmax', 'odds_tminus10m', 'odds_snapshot',
    'p_market', 'p_market_calibrated', 'own_odds', 'norm_odds',
    'raw_time', 'last_3f', 'estimated_place_rate', 'rank_str'
]

def load_data(year_start=2013, year_end=2025):
    """Load all necessary data for WF"""
    # 1. Load Preprocessed Data (Historical + 2025)
    # Using existing parquet cache if available, else standard path
    # Absolute path based on script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base_dir, '../../data/processed/preprocessed_data.parquet')
    
    if not os.path.exists(parquet_path):
        # Try relative from workspace root
        parquet_path = 'data/processed/preprocessed_data.parquet'
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Data not found: {parquet_path}")
    
    logger.info(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded Parquet. Shape: {df.shape}")
    
    # 2. Add Time Info
    if 'date' not in df.columns:
        logger.error("Column 'date' missing!")
        
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")

    # Filter Years
    df = df[(df['year'] >= year_start) & (df['year'] <= year_end)].copy()
    logger.info(f"Filtered Shape: {df.shape}")

    # 3. Target & Base Margin
    # Target: rank=1
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['target'] = (df['rank'] == 1).astype(int)
    
    # Base Margin (Training): Logit(1/FinalOdds)
    # Note: 'odds' in parquet is usually Final Odds.
    # Clip probabilities to avoid inf
    # Ensure odds is numeric
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(1000.0)
    p_final = (1.0 / df['odds']).clip(1e-4, 1.0 - 1e-4)
    df['logit_final_odds'] = logit(p_final)
    
    # Convert Object info Category for LightGBM
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c] = df[c].astype('category')
    
    logger.info(f"Converted {len(cat_cols)} columns to category.")
    
    logger.info("Data loaded successfully.")
    return df

def align_inference_odds(df_2025, noleak_pq_path):
    """Merge T-10m odds from noleak parquet for inference"""
    logger.info(f"Merging T-10m odds from {noleak_pq_path}...")
    noleak = pd.read_parquet(noleak_pq_path)
    
    # Keys: race_id, horse_number
    noleak['race_id'] = noleak['race_id'].astype(str)
    noleak['horse_number'] = noleak['horse_number'].astype(int)
    
    # Select cols
    noleak = noleak[['race_id', 'horse_number', 'odds_tminus10m']]
    
    # Merge
    df_2025['race_id'] = df_2025['race_id'].astype(str)
    df_2025['horse_number'] = df_2025['horse_number'].astype(int)
    
    merged = pd.merge(df_2025, noleak, on=['race_id', 'horse_number'], how='left')
    
    # Fill missing T-10m with Final (Warning: This is essentially fallback, but for missing races only)
    # Actually, if missing, we should skip or use final with flag?
    # For diagnosis, we want strictly T-10m.
    # We will drop rows where T-10m is missing?
    # Or just keep NaN and filter later.
    return merged

def train_and_predict_wf(df, noleak_pq_path):
    """Walk Forward 2025"""
    
    results = []
    
    # Prepare 2025 Data for Inference (Merge T-10m)
    df_2025 = df[df['year'] == 2025].copy()
    df_2025 = align_inference_odds(df_2025, noleak_pq_path)
    
    # Calc Inference Base Margin: logit(1/T-10m)
    p_t10 = (1.0 / df_2025['odds_tminus10m']).clip(1e-4, 1.0 - 1e-4)
    df_2025['logit_t10_odds'] = logit(p_t10)
    
    # Loop Months 1..4 (Partial WF for Speed)
    for m in range(1, 4):
        logger.info(f"=== Walk Forward: 2025-{m:02d} === ")
        
        # Test Data: Current Month
        test_mask = (df_2025['month'] == m) & (df_2025['logit_t10_odds'].notna())
        test_data = df_2025[test_mask].copy()
        
        if test_data.empty:
            continue
            
        # APPX: Train Data = All History before this month
        # To save time, we can reuse previous model or incremental?
        # LightGBM refit is fast.
        # Train on < 2025-m-01
        
        cutoff_date = datetime(2025, m, 1)
        train_df = df[df['date'] < cutoff_date].copy()
        
        # Features
        # STRICT ALLOWLIST
        def is_allowed(c):
            if c in ['race_id', 'target', 'date', 'month', 'year']: return False
            if c.startswith('lag1_'): return True
            if c.startswith('mean_'): return True
            if '_emb_' in c: return True
            if c in ['horse_id', 'jockey_id', 'trainer_id', 'course_id', 'distance', 'age', 'sex_num', 'weight', 'frame_number', 'horse_number']: return True
            if c.endswith('_n_races') or c.endswith('_win_rate') or c.endswith('_top3_rate'): return True # Expanding stats
            return False

        use_cols = [c for c in train_df.columns if is_allowed(c)]
        
        logger.info(f"Training on {len(train_df)} rows. Features: {len(use_cols)}")
        
        # Dataset
        lgb_train = lgb.Dataset(
            train_df[use_cols], 
            label=train_df['target'],
            init_score=train_df['logit_final_odds'] # Residual Learning
        )
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 15,
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=40
        )
        
        # Predict
        preds_margin = model.predict(test_data[use_cols], raw_score=True)
        # Add Base Margin (T-10m)
        final_logit = preds_margin + test_data['logit_t10_odds'].values
        final_prob = expit(final_logit)
        
        # Store
        test_data['pred_prob'] = final_prob
        test_data['pred_logit'] = final_logit
        test_data['base_logit'] = test_data['logit_t10_odds']
        test_data['model_logit_delta'] = preds_margin
        
        results.append(test_data[['race_id', 'horse_number', 'odds_tminus10m', 'pred_prob', 'pred_logit']])
        
        gc.collect()

    return pd.concat(results, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noleak_pq', required=True)
    parser.add_argument('--out_pq', required=True)
    args = parser.parse_args()
    
    # Load
    df = load_data()
    
    # Run WF
    wf_preds = train_and_predict_wf(df, args.noleak_pq)
    
    # Save OOF
    os.makedirs(os.path.dirname(args.out_pq), exist_ok=True)
    wf_preds.to_parquet(args.out_pq)
    logger.info(f"Saved WF predictions to {args.out_pq}")
    
    # Basic Eval
    # Merge targets
    engine = get_db_engine()
    results = pd.read_sql("SELECT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id, umaban as horse_number, kakutei_chakujun as rank, tansho_odds FROM jvd_se WHERE kaisai_nen = '2025'", engine)
    results['race_id'] = results['race_id'].astype(str)
    results['horse_number'] = pd.to_numeric(results['horse_number'], errors='coerce')
    results['final_odds'] = pd.to_numeric(results['tansho_odds'], errors='coerce') / 10.0
    
    wf_preds['race_id'] = wf_preds['race_id'].astype(str)
    wf_preds['horse_number'] = wf_preds['horse_number'].astype(int)
    
    eval_df = pd.merge(wf_preds, results, on=['race_id', 'horse_number'])
    eval_df['target'] = (eval_df['rank'].astype(float) == 1).astype(int)
    
    # LogLoss
    loss = -np.mean(eval_df['target'] * np.log(eval_df['pred_prob'] + 1e-15) + (1-eval_df['target'])*np.log(1-eval_df['pred_prob'] + 1e-15))
    logger.info(f"WF 2025 LogLoss: {loss:.4f}")
    
    # ROI (Win EV > 1.0)
    eval_df['ev'] = eval_df['pred_prob'] * eval_df['odds_tminus10m']
    bets = eval_df[eval_df['ev'] > 1.0]
    if not bets.empty:
        ret = bets[bets['target']==1]['final_odds'].sum()
        roi = ret / len(bets) * 100
        logger.info(f"WF 2025 ROI (EV>1.0): {roi:.2f}% ({len(bets)} bets)")
    else:
        logger.info("No bets found.")

if __name__ == "__main__":
    main()

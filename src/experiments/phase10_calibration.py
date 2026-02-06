import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from dateutil.relativedelta import relativedelta
from scipy.special import logit, expit

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.scripts.auto_predict_v13 import get_db_engine

def load_predictions(pq_path):
    logger.info(f"Loading WF predictions from {pq_path}...")
    df = pd.read_parquet(pq_path)
    # Ensure date/month if not present
    # WF OOF usually has date or month cols if we added them.
    # If not, merged from race_id??
    # phase10_wf_full_retrain.py adds: race_id, horse_number, odds_tminus10m, pred_prob, pred_logit, base_logit
    # It does NOT explicitly save 'date' or 'month' in the results, but it loops.
    # We need to recover date information.
    
    # Recover date from race_id (Year only? No, we need month for rolling window)
    # race_id: 2025...
    # We should merge with DB to get accurate dates.
    engine = get_db_engine()
    query = "SELECT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id, kaisai_nen, kaisai_tsukihi FROM jvd_se WHERE kaisai_nen = '2025' GROUP BY race_id, kaisai_nen, kaisai_tsukihi"
    dates = pd.read_sql(query, engine)
    
    dates['race_id'] = dates['race_id'].astype(str)
    # kaisai_tsukihi: MMDD
    dates['month'] = dates['kaisai_tsukihi'].astype(str).str.zfill(4).str.slice(0, 2).astype(int)
    
    df['race_id'] = df['race_id'].astype(str)
    df = pd.merge(df, dates[['race_id', 'month']], on='race_id', how='left')
    
    # Target
    # Merge rank if not present (WF script doesn't save rank in output parquet)
    if 'target' not in df.columns:
        query_rank = "SELECT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id, umaban as horse_number, kakutei_chakujun as rank FROM jvd_se WHERE kaisai_nen = '2025'"
        ranks = pd.read_sql(query_rank, engine)
        ranks['race_id'] = ranks['race_id'].astype(str)
        ranks['horse_number'] = pd.to_numeric(ranks['horse_number'], errors='coerce')
        
        df['horse_number'] = df['horse_number'].astype(int)
        df = pd.merge(df, ranks, on=['race_id', 'horse_number'], how='inner')
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
        df['target'] = (df['rank'] == 1).astype(int)
        
    return df

def calibrate_month(df, target_month):
    """
    Calibrate predictions for target_month using previous months (up to 6 months window).
    Methods:
    - Temperature Scaling (Logistic Regression on Logit)
    - Isotonic Regression (Non-parametric)
    """
    # Train Data: (M-6) <= Month < M
    # Min 1 month of history required
    start_month = max(1, target_month - 6)
    train_mask = (df['month'] >= start_month) & (df['month'] < target_month)
    eval_mask = (df['month'] == target_month)
    
    train_df = df[train_mask]
    eval_df = df[eval_mask]
    
    if train_df.empty:
        logger.warning(f"No calibration data for Month {target_month}. Using raw prob.")
        return eval_df[['pred_prob']].rename(columns={'pred_prob': 'calib_prob_iso'}) # fallback
        
    X_train = train_df[['pred_logit']].values
    y_train = train_df['target'].values
    X_eval = eval_df[['pred_logit']].values
    
    # 1. Logistic Calibration (Platt Scaling)
    lr = LogisticRegression(C=1.0, solver='lbfgs')
    lr.fit(X_train, y_train)
    prob_lr = lr.predict_proba(X_eval)[:, 1]
    
    # 2. Isotonic Calibration (Requires monotonic, usually on Prob)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(train_df['pred_prob'].values, y_train)
    prob_iso = iso.predict(eval_df['pred_prob'].values)
    
    # Return
    calib_res = pd.DataFrame(index=eval_df.index)
    calib_res['calib_prob_lr'] = prob_lr
    calib_res['calib_prob_iso'] = prob_iso
    
    return calib_res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pq', required=True)
    parser.add_argument('--out_pq', required=True)
    args = parser.parse_args()
    
    df = load_predictions(args.input_pq)
    
    dfs = []
    
    # Loop Months 1..12
    # Jan (1) has no history in 2025 -> Skip calibration or use raw?
    # Spec says: "Calibrate using future data is forbidden".
    # For Jan, if we don't have 2024 OOF, we can't calibrate properly using OOF method.
    # Fallback for Jan: Raw Prob.
    
    for m in range(1, 13):
        logger.info(f"Calibrating Month {m}...")
        
        mask = df['month'] == m
        month_df = df[mask].copy()
        
        if month_df.empty:
            continue
            
        if m == 1:
            # No prior data in this dataset
            month_df['calib_prob_lr'] = month_df['pred_prob']
            month_df['calib_prob_iso'] = month_df['pred_prob']
        else:
            calib_res = calibrate_month(df, m)
            month_df['calib_prob_lr'] = calib_res['calib_prob_lr']
            month_df['calib_prob_iso'] = calib_res['calib_prob_iso']
            
        dfs.append(month_df)
        
    final_df = pd.concat(dfs)
    final_df.to_parquet(args.out_pq)
    logger.info(f"Saved calibrated predictions to {args.out_pq}")
    
    # Quick Eval
    from sklearn.metrics import log_loss, brier_score_loss
    
    logger.info("Evaluation (Full Year):")
    logger.info(f"Raw LogLoss: {log_loss(final_df['target'], final_df['pred_prob']):.4f}")
    logger.info(f"LR  LogLoss: {log_loss(final_df['target'], final_df['calib_prob_lr']):.4f}")
    logger.info(f"Iso LogLoss: {log_loss(final_df['target'], final_df['calib_prob_iso']):.4f}")

if __name__ == "__main__":
    main()

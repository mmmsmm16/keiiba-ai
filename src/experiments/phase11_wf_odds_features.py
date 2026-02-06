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

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.features.odds_movement_features import calculate_odds_movement_features

def load_data(year_start=2014, year_end=2025):
    """Load Base Data"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base_dir, '../../data/processed/preprocessed_data.parquet')
    
    logger.info(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Filter
    df = df[(df['year'] >= year_start) & (df['year'] <= year_end)].copy()
    
    # Target & Base Margin
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['target'] = (df['rank'] == 1).astype(int)
    
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(1000.0)
    p_final = (1.0 / df['odds']).clip(1e-4, 1.0 - 1e-4)
    df['logit_final_odds'] = logit(p_final)
    
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c] = df[c].astype('category')
        
    return df

def train_and_predict_wf_odds(df, noleak_pq_path):
    """Walk Forward 2025 with Odds Features (Trained on 2014-2024 History)"""
    
    # 1. Calc Odds Features for Full History (2014-2025)
    logger.info("Computing Odds Movement Features (2014-2025)...")
    
    # We don't need to pass a specific DF, just race_ids?
    # Actually calculate_odds_movement_features takes 'df_features' just for signature matching or future?
    # The current implementation ignores the input DF and loads everything from parquet.
    # So we can pass a dummy.
    
    odds_feats = calculate_odds_movement_features(None, start_year=2014, end_year=2025)
    
    if odds_feats.empty:
        logger.error("No odds features computed!")
        return pd.DataFrame()
        
    logger.info(f"Odds Features Computed: {len(odds_feats)} rows. Cols: {odds_feats.columns.tolist()}")
    
    df['race_id'] = df['race_id'].astype(str)
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')
    df = df.dropna(subset=['horse_number'])
    df['horse_number'] = df['horse_number'].astype(int)
    
    # Merge (Left join)
    df = pd.merge(df, odds_feats, on=['race_id', 'horse_number'], how='left')
    
    # Prepare 2025 Data for Inference (Merge T-10m)
    noleak = pd.read_parquet(noleak_pq_path)
    noleak['race_id'] = noleak['race_id'].astype(str)
    df_2025 = df[df['year'] == 2025].copy()
    df_2025 = pd.merge(df_2025, noleak[['race_id', 'horse_number', 'odds_tminus10m']], on=['race_id', 'horse_number'], how='left')
    
    p_t10 = (1.0 / df_2025['odds_tminus10m']).clip(1e-4, 1.0 - 1e-4)
    df_2025['logit_t10_odds'] = logit(p_t10)
    
    results = []
    
    # Odds Features + Base Features
    def is_allowed(c):
        # Base Strict Allowlist
        if c in ['race_id', 'target', 'date', 'month', 'year']: return False
        if c.startswith('lag1_'): return True
        if c.startswith('mean_'): return True
        if '_emb_' in c: return True
        if c in ['horse_id', 'jockey_id', 'trainer_id', 'course_id', 'distance', 'age', 'sex_num', 'weight', 'frame_number', 'horse_number']: return True
        if c.endswith('_n_races') or c.endswith('_win_rate') or c.endswith('_top3_rate'): return True
        
        # New Odds Features
        if c in ['dlog_odds_t60_t10', 'dlog_odds_t30_t10', 'odds_volatility', 'rank_change_t60_t10', 'odds_drop_rate_t60_t10']: return True
        # Note: log_odds_t10 is essentially T-10 odds (logit). 
        # We ALREADY use T-10 odds as Base Margin. 
        # Adding it as feature might help (non-linear), but Base Margin handles linear part.
        # Let's include it.
        if c == 'log_odds_t10': return True
            
        return False

    all_cols = [c for c in df.columns if is_allowed(c)]
    
    logger.info(f"Feature Count: {len(all_cols)}")
    logger.info(f"Odds Features in list: {[c for c in all_cols if 'odds' in c or 'volatility' in c]}")

    for m in range(1, 13):
        logger.info(f"=== Walk Forward: 2025-{m:02d} === ")
        
        test_mask = (df_2025['month'] == m) & (df_2025['logit_t10_odds'].notna())
        test_data = df_2025[test_mask].copy()
        
        if test_data.empty:
            continue
            
        cutoff_date = datetime(2025, m, 1)
        train_df = df[df['date'] < cutoff_date].copy()
        
        lgb_train = lgb.Dataset(train_df[all_cols], label=train_df['target'], init_score=train_df['logit_final_odds'])
        
        # Train
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'learning_rate': 0.05, 'num_leaves': 15, 'verbose': -1
        }
        
        model = lgb.train(params, lgb_train, num_boost_round=40)
        
        preds_margin = model.predict(test_data[all_cols], raw_score=True)
        final_logit = preds_margin + test_data['logit_t10_odds'].values
        final_prob = expit(final_logit)
        
        test_data['pred_prob'] = final_prob
        results.append(test_data)
        
    return pd.concat(results, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noleak_pq', required=True)
    parser.add_argument('--out_report', required=True)
    args = parser.parse_args()
    
    df = load_data()
    preds = train_and_predict_wf_odds(df, args.noleak_pq)
    
    if preds.empty:
        logger.error("No predictions!")
        return

    # Eval
    preds['ev'] = preds['pred_prob'] * preds['odds_tminus10m']
    loss = -np.mean(preds['target'] * np.log(preds['pred_prob'] + 1e-15) + (1-preds['target'])*np.log(1-preds['pred_prob'] + 1e-15))
    
    bets = preds[preds['ev'] > 1.0]
    if len(bets) > 0:
        # Merge Final Odds for ROI
        from src.scripts.auto_predict_v13 import get_db_engine
        engine = get_db_engine()
        odds_res = pd.read_sql("SELECT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id, umaban as horse_number, tansho_odds FROM jvd_se WHERE kaisai_nen = '2025'", engine)
        odds_res['race_id'] = odds_res['race_id'].astype(str)
        odds_res['horse_number'] = pd.to_numeric(odds_res['horse_number'], errors='coerce')
        odds_res['final_odds'] = pd.to_numeric(odds_res['tansho_odds'], errors='coerce') / 10.0
        
        bets = pd.merge(bets, odds_res[['race_id', 'horse_number', 'final_odds']], on=['race_id', 'horse_number'], how='inner')
        ret = bets[bets['target']==1]['final_odds'].sum()
        roi = ret / len(bets) * 100
    else:
        roi = 0
        
    # Write Report
    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    with open(args.out_report, 'w') as f:
        f.write("# Phase 11: Odds Feature Evaluation (2025)\n\n")
        f.write(f"- **LogLoss**: {loss:.4f}\n")
        f.write(f"- **ROI (EV>1.0)**: {roi:.2f}% ({len(bets)} bets)\n")
        f.write("\n## Feature Importance (Last Split)\n")
        # model is local variable... can't print easily without refactoring. 
        # Just result summary first.

if __name__ == "__main__":
    main()

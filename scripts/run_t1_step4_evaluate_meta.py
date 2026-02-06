
import sys
import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    # Load same data as training
    pred_path_2025 = "models/experiments/exp_r3_ensemble/predictions_2025.parquet"
    t1_path = "data/temp_t1/T1_features_2024_2025.parquet"
    
    if not os.path.exists(pred_path_2025) or not os.path.exists(t1_path):
        raise FileNotFoundError("Data files missing.")
        
    df_preds = pd.read_parquet(pred_path_2025)
    df_preds['year'] = 2025
    df_features = pd.read_parquet(t1_path)

    # Merge features
    # Ensure race_id is str in all dfs
    df_preds['race_id'] = df_preds['race_id'].astype(str)
    df_features['race_id'] = df_features['race_id'].astype(str)
    
    df = pd.merge(df_preds, df_features, on=['race_id', 'horse_number'], how='inner')
    
    # Ensure 'date' column exists for monthly analysis
    if 'date' not in df.columns:
        logger.info("Date column missing, loading from year_2025.parquet...")
        raw_2025_path = "data/temp_q1/year_2025.parquet"
        if os.path.exists(raw_2025_path):
            df_dates = pd.read_parquet(raw_2025_path, columns=['race_id', 'date']).drop_duplicates(subset=['race_id'])
            df_dates['race_id'] = df_dates['race_id'].astype(str)
            df = pd.merge(df, df_dates, on='race_id', how='left')
        else:
             logger.warning(f"{raw_2025_path} not found! Monthly analysis might fail.")

    # Feature Eng (Must match training)
    df['prob_market_10min'] = 0.8 / (df['odds_10min'] + 1e-9)
    df['prob_diff_10min'] = df['pred_prob'] - df['prob_market_10min']
    
    # Ensure Final Odds for ROI calc (odds_final from t1 or odds from preds)
    if 'odds_final' in df_features.columns:
        df['odds_calc'] = df['odds_final']
    elif 'odds' in df_preds.columns:
        df['odds_calc'] = df['odds']
    else:
        df['odds_calc'] = 0
        
    return df

def evaluate_calibration(df, target_col='rank', prob_col='meta_prob'):
    logger.info(f"\n--- Calibration ({prob_col}) ---")
    y_true = (df[target_col] == 1).astype(int)
    prob_true, prob_pred = calibration_curve(y_true, df[prob_col], n_bins=10)
    
    print(f"{'Mean Pred':<10} | {'True Prob':<10} | {'Diff':<10}")
    print("-" * 35)
    for p_pred, p_true in zip(prob_pred, prob_true):
        diff = p_pred - p_true
        print(f"{p_pred:.4f}     | {p_true:.4f}     | {diff:+.4f}")
        
    brier = brier_score_loss(y_true, df[prob_col])
    logger.info(f"Brier Score: {brier:.4f}")

def evaluate_monthly_roi(df, prob_col='meta_prob', threshold=0.2):
    logger.info(f"\n--- Monthly ROI (Threshold >= {threshold}) ---")
    
    # Extract Month using date column
    print(f"DEBUG: Columns = {df.columns.tolist()}")
    if 'date' in df.columns:
        print(f"DEBUG: date head: {df['date'].head()}")
        # date might be int (20250105) or str or datetime
        # Convert to str first then datetime
        df['temp_date'] = pd.to_datetime(df['date'].astype(str))
        df['month'] = df['temp_date'].dt.month
        
    elif 'year' in df.columns:
         # Without date, maybe we can't do monthly.
         pass
        
    if 'month' not in df.columns:
        logger.warning("Month column not found, skipping Monthly ROI.")
        return

    df_bet = df[df[prob_col] >= threshold].copy()
    
    grouped = df_bet.groupby('month')
    
    print(f"{'Month':<5} | {'Count':<6} | {'Hit%':<6} | {'ROI%':<6} | {'Profit':<8}")
    print("-" * 50)
    
    total_cost = 0
    total_return = 0
    
    for month, group in grouped:
        n_bets = len(group)
        if n_bets == 0: continue
        
        hits = group[group['rank'] == 1]
        cost = n_bets * 100
        ret = (hits['odds_calc'] * 100).sum()
        
        roi = ret / cost
        hit_rate = len(hits) / n_bets
        profit = ret - cost
        
        total_cost += cost
        total_return += ret
        
        print(f"{str(month):<5} | {n_bets:<6} | {hit_rate*100:>5.1f}% | {roi*100:>5.1f}% | {profit:>8.0f}")
        
    print("-" * 50)
    if total_cost > 0:
        print(f"{'Total':<5} | {int(total_cost/100):<6} | {(total_return/total_cost)*100:>5.1f}% | {total_return - total_cost:>8.0f}")

def evaluate_odds_bands(df, prob_col='meta_prob', threshold=0.2):
    logger.info(f"\n--- Odds Band ROI (Threshold >= {threshold}) ---")
    
    df_bet = df[df[prob_col] >= threshold].copy()
    
    bins = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0]
    labels = ['1.0-2.0', '2.0-5.0', '5.0-10.0', '10.0-20.0', '20.0-50.0', '50.0-100', '100+']
    
    df_bet['odds_bin'] = pd.cut(df_bet['odds_calc'], bins=bins, labels=labels, right=False)
    
    grouped = df_bet.groupby('odds_bin', observed=False)
    
    print(f"{'Odds Band':<12} | {'Count':<6} | {'Hit%':<6} | {'ROI%':<6} | {'Profit':<8}")
    print("-" * 60)
    
    for band, group in grouped:
        n_bets = len(group)
        if n_bets == 0: continue
        
        hits = group[group['rank'] == 1]
        cost = n_bets * 100
        ret = (hits['odds_calc'] * 100).sum()
        
        roi = ret / cost
        hit_rate = len(hits) / n_bets
        profit = ret - cost
        
        print(f"{str(band):<12} | {n_bets:<6} | {hit_rate*100:>5.1f}% | {roi*100:>5.1f}% | {profit:>8.0f}")

def main():
    logger.info("🚀 Starting Phase T Step 3.5: Detailed Meta-Model Evaluation")
    
    # 1. Load Data
    df = load_data()
    logger.info(f"Loaded {len(df)} rows for 2025.")
    
    # 2. Load Model & Predict
    model_path = "models/experiments/exp_t1_meta/meta_lgbm.txt"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
        
    model = lgb.Booster(model_file=model_path)
    
    feature_cols = ['pred_prob', 'odds_10min', 'prob_market_10min', 'prob_diff_10min']
    
    # Filter valid features
    df_valid = df[df['odds_10min'].notna()].copy()
    
    logger.info("Predicting...")
    df_valid['meta_prob'] = model.predict(df_valid[feature_cols])
    
    # 3. Validation Metrics
    auc = roc_auc_score((df_valid['rank']==1), df_valid['meta_prob'])
    ll = log_loss((df_valid['rank']==1), df_valid['meta_prob'])
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"LogLoss: {ll:.4f}")
    
    evaluate_calibration(df_valid)
    
    # 4. ROI Simulations
    # Threshold Analysis
    logger.info("\n--- Threshold Analysis ---")
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print(f"{'Th':<5} | {'Count':<6} | {'Hit%':<6} | {'ROI%':<6} | {'Profit':<8}")
    print("-" * 50)
    for th in thresholds:
        bets = df_valid[df_valid['meta_prob'] >= th]
        if len(bets) == 0: continue
        
        cost = len(bets) * 100
        hits = bets[bets['rank'] == 1]
        ret = (hits['odds_calc'] * 100).sum()
        roi = ret / cost
        hit_rate = len(hits) / len(bets)
        profit = ret - cost
        print(f"{th:<5} | {len(bets):<6} | {hit_rate*100:>5.1f}% | {roi*100:>5.1f}% | {profit:>8.0f}")
        
    # Best Threshold for detailed analysis (e.g. 0.2)
    BEST_TH = 0.2
    
    evaluate_monthly_roi(df_valid, threshold=BEST_TH)
    evaluate_odds_bands(df_valid, threshold=BEST_TH)

if __name__ == "__main__":
    main()

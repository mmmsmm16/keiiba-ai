
import sys
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting Base Model ROI Grid Search (2024)...")
    
    # 1. Load Predictions (Base Model)
    pred_path = "data/temp_t2/T2_predictions_2024_2025.parquet"
    if not os.path.exists(pred_path):
        logger.error(f"Predictions not found: {pred_path}")
        return
    df_pred = pd.read_parquet(pred_path)
    df_pred['race_id'] = df_pred['race_id'].astype(str)
    
    # 2. Load Odds (from T1 features)
    odds_path = "data/temp_t1/T1_features_2024_2025.parquet"
    if not os.path.exists(odds_path):
        logger.error(f"Odds features not found: {odds_path}")
        return
    df_odds = pd.read_parquet(odds_path)
    df_odds['race_id'] = df_odds['race_id'].astype(str)
    
    # Check if df_odds has 'odds_final'
    if 'odds_final' not in df_odds.columns:
        logger.error("odds_final column missing in T1 features")
        return

    # Deduplicate odds just in case
    df_odds = df_odds.drop_duplicates(['race_id', 'horse_number'])
    
    # 3. Merge
    df = pd.merge(df_pred, df_odds[['race_id', 'horse_number', 'odds_final']], 
                  on=['race_id', 'horse_number'], how='inner')
    
    logger.info(f"Merged data shape: {df.shape}")
    
    # Filter for 2024 only (ensure data validity)
    df['date'] = pd.to_datetime(df['date'])
    df_2024 = df[df['date'].dt.year == 2024].copy()
    logger.info(f"Analyzing 2024 data: {len(df_2024)} rows")
    
    df_2024['is_win'] = (df_2024['rank'] == 1).astype(int)
    
    # 4. Grid Search
    # Parameters
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    odds_filters = [
        (1.0, 999.0, "All"),
        (1.5, 999.0, ">= 1.5"),
        (2.0, 999.0, ">= 2.0"),
        (2.0, 10.0, "2.0-10.0"),
        (2.0, 20.0, "2.0-20.0"),
        (3.0, 10.0, "3.0-10.0"),
        (3.0, 20.0, "3.0-20.0"),
        (4.0, 20.0, "4.0-20.0"),
        (5.0, 20.0, "5.0-20.0"),
        (1.5, 5.0, "1.5-5.0 (Fav)"),
        (5.0, 15.0, "5.0-15.0 (Mid)"),
    ]
    
    results = []
    
    for th in thresholds:
        for min_od, max_od, label in odds_filters:
            # Filter
            mask = (df_2024['pred_prob'] >= th) & \
                   (df_2024['odds_final'] >= min_od) & \
                   (df_2024['odds_final'] <= max_od)
            
            bets = df_2024[mask]
            n_bets = len(bets)
            
            if n_bets < 50: # Ignore low sample size
                continue
                
            n_wins = bets['is_win'].sum()
            return_amount = bets.loc[bets['is_win'] == 1, 'odds_final'].sum() * 100
            cost = n_bets * 100
            roi = return_amount / cost * 100
            hit_rate = n_wins / n_bets * 100
            
            results.append({
                'conf_th': th,
                'odds_label': label,
                'n_bets': n_bets,
                'roi': roi,
                'hit_rate': hit_rate,
                'wins': n_wins
            })
            
    # 5. Output
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roi', ascending=False)
    
    print("\n========= ROI Grid Search Results (Top 20) =========")
    print(results_df.head(20).to_string(index=False))
    
    if not results_df.empty:
        best = results_df.iloc[0]
        logger.info(f"Best Strategy: Conf>={best['conf_th']}, Odds={best['odds_label']} -> ROI={best['roi']:.1f}% (N={best['n_bets']})")
    else:
        logger.info("No strategy found with >50 bets.")

if __name__ == "__main__":
    main()

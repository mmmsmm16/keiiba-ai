
import sys
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting T2 Refined v3 ROI Grid Search (2024)...")
    
    # 1. Load Predictions (V3)
    pred_path = "data/temp_t2/T2_predictions_2024_refined_v3.parquet"
    if not os.path.exists(pred_path):
        logger.error(f"Predictions not found: {pred_path}. Run predict_t2_v3_2024.py first.")
        return
    df_pred = pd.read_parquet(pred_path)
    df_pred['race_id'] = df_pred['race_id'].astype(str)
    
    
    # Add project root to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # 2. Load Result & Odds from DB
    from src.preprocessing.loader import JraVanDataLoader
    loader = JraVanDataLoader()
    # Load 2024 data (including rank, odds)
    logger.info("Loading 2024 Results from DB...")
    # skip_training=True is fine as we don't need training data (hanro), just results
    df_odds = loader.load(history_start_date='2024-01-01', end_date='2024-12-31', jra_only=True, skip_training=True)
    
    # Ensure columns exist
    if 'odds' in df_odds.columns:
        df_odds['odds_final'] = df_odds['odds']
    else:
        logger.error("odds column missing in DB data")
        return

    # Filter columns (race_id is object in loader? No, loader returns strings usually? check)
    # create matching types
    df_odds['race_id'] = df_odds['race_id'].astype(str)
    
    # 3. Merge
    df = pd.merge(df_pred, df_odds[['race_id', 'horse_number', 'odds_final', 'rank']], 
                  on=['race_id', 'horse_number'], how='inner', suffixes=('', '_odds'))
    
    # Use rank from odds file if missing or create combined
    if 'rank' not in df.columns:
        df['rank'] = df['rank_odds']
    
    logger.info(f"Merged data shape: {df.shape}")
    
    df['is_win'] = (df['rank'] == 1).astype(int)
    
    # 4. Grid Search
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
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
            mask = (df['pred_prob'] >= th) & \
                   (df['odds_final'] >= min_od) & \
                   (df['odds_final'] <= max_od)
            
            bets = df[mask]
            n_bets = len(bets)
            
            if n_bets < 100: 
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
    if not results_df.empty:
        results_df = results_df.sort_values('roi', ascending=False)
        print("\n========= ROI Grid Search Results (Top 20) =========")
        print(results_df.head(20).to_string(index=False))
        
        # Save report
        results_df.to_csv(f"reports/simulations/roi_v3_search_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", index=False)
    else:
        logger.info("No strategy found with >100 bets.")

if __name__ == "__main__":
    main()

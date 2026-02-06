
import sys
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_bin_analysis(df, bin_col, bins, label=""):
    """
    Perform ROI analysis for binned values of a column.
    """
    logger.info(f"\n--- {label} Analysis ({bin_col}) ---")
    
    df['bin'] = pd.cut(df[bin_col], bins=bins)
    
    grouped = df.groupby('bin', observed=False)
    
    stats = pd.DataFrame({
        'count': grouped.size(),
        'win_count': grouped.apply(lambda x: (x['rank'] == 1).sum()),
        'return': grouped.apply(lambda x: x.loc[x['rank'] == 1, 'odds'].sum() * 100),
        'cost': grouped.size() * 100
    })
    
    stats['win_rate'] = stats['win_count'] / stats['count']
    stats['roi'] = stats['return'] / stats['cost']
    
    # Format for display
    print(f"{'Bin Range':<20} | {'Count':<6} | {'Win%':<6} | {'ROI%':<6} | {'Profit':<8}")
    print("-" * 60)
    
    for idx, row in stats.iterrows():
        if row['count'] == 0: continue
        print(f"{str(idx):<20} | {int(row['count']):<6} | {row['win_rate']*100:>5.1f}% | {row['roi']*100:>5.1f}% | {row['return'] - row['cost']:>8.0f}")

def main():
    logger.info("🚀 Starting Phase T Step 2: Residual & Odds Analysis")
    
    # 1. Load Data
    # Predictions
    pred_path_2024 = "models/experiments/exp_r3_ensemble/predictions_2024.parquet"
    pred_path_2025 = "models/experiments/exp_r3_ensemble/predictions_2025.parquet"
    
    preds = []
    if os.path.exists(pred_path_2024): preds.append(pd.read_parquet(pred_path_2024))
    if os.path.exists(pred_path_2025): preds.append(pd.read_parquet(pred_path_2025))
    
    if not preds:
        logger.error("No predictions found!")
        return
        
    df_preds = pd.concat(preds, ignore_index=True)
    logger.info(f"Loaded predictions: {len(df_preds)} rows")
    
    # T1 Features (Odds)
    t1_path = "data/temp_t1/T1_features_2024_2025.parquet"
    if not os.path.exists(t1_path):
        logger.error("T1 Features not found!")
        return
        
    df_features = pd.read_parquet(t1_path)
    logger.info(f"Loaded T1 features: {len(df_features)} rows")
    
    # Merge
    # df_preds has race_id, horse_number, rank, pred_prob, odds
    # df_features has race_id, horse_number, odds_ratio_10min, etc.
    # Note: df_features created using 'cleaned' raw data, structure should match.
    
    df = pd.merge(df_preds, df_features, on=['race_id', 'horse_number'], how='inner')
    logger.info(f"Merged Data: {len(df)} rows")
    
    # 2. Residual Calculation
    # Market Prob ~ 0.8 / Final Odds
    # Ensure odds > 0
    df = df[df['odds'] > 0].copy()
    
    df['prob_market'] = 0.8 / df['odds']
    df['residual'] = df['pred_prob'] - df['prob_market']
    
    # 3. Analysis 1: Residual vs ROI
    # Positive residual means AI prob > Market prob (Undervalued / "Buy")
    # Negative residual means AI prob < Market prob (Overvalued / "Sell")
    
    bins_resid = [-1.0, -0.2, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    perform_bin_analysis(df, 'residual', bins_resid, label="Residual (AI - Market)")
    
    # 4. Analysis 2: Odds Ratio (10min Fluctuation)
    # Ratio = Final / 10min
    # Ratio < 1.0: Dropped (Bought)
    # Ratio > 1.0: Rose (Sold)
    if 'odds_ratio_10min' in df.columns:
        # Drop nan
        df_valid_ratio = df.dropna(subset=['odds_ratio_10min']).copy()
        bins_ratio = [0.0, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 5.0]
        perform_bin_analysis(df_valid_ratio, 'odds_ratio_10min', bins_ratio, label="Odds Ratio (Final/10min)")
        
        # Interaction Analysis: High Residual + Odds Drop?
        # Typically, "Smart Money" drops the odds.
        # If AI agrees (High Residual) AND Smart Money agrees (Odds Drop), ROI should be high.
        
        logger.info("\n--- Interaction Analysis: Residual > 0.05 + Odds Trend ---")
        high_value = df_valid_ratio[df_valid_ratio['residual'] > 0.05].copy()
        
        perform_bin_analysis(high_value, 'odds_ratio_10min', bins_ratio, label="High Residual (>0.05) & Odds Trend")

    # 5. Analysis 3: Residual for Favorites (Odds < 5.0)
    logger.info("\n--- Favorites Analysis (Odds < 5.0) ---")
    favorites = df[df['odds'] < 5.0].copy()
    perform_bin_analysis(favorites, 'residual', bins_resid, label="Residual (Favorites)")

if __name__ == "__main__":
    main()

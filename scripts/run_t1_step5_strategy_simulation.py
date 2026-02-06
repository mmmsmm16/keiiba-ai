
import sys
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    # Load same data as evaluation
    pred_path_2025 = "models/experiments/exp_r3_ensemble/predictions_2025.parquet"
    t1_path = "data/temp_t1/T1_features_2024_2025.parquet"
    
    if not os.path.exists(pred_path_2025) or not os.path.exists(t1_path):
        raise FileNotFoundError("Data files missing.")
        
    df_preds = pd.read_parquet(pred_path_2025)
    df_preds['race_id'] = df_preds['race_id'].astype(str)
    
    df_features = pd.read_parquet(t1_path)
    df_features['race_id'] = df_features['race_id'].astype(str)
    
    df = pd.merge(df_preds, df_features, on=['race_id', 'horse_number'], how='inner')
    
    # Load Meta Predictions if available, otherwise recommend generating them first
    # Or we can just load the model and predict on the fly again? 
    # Better to predict on fly to be self-contained.
    
    import lightgbm as lgb
    model_path = "models/experiments/exp_t1_meta/meta_lgbm.txt"
    if os.path.exists(model_path):
        model = lgb.Booster(model_file=model_path)
        df['prob_market_10min'] = 0.8 / (df['odds_10min'] + 1e-9)
        df['prob_diff_10min'] = df['pred_prob'] - df['prob_market_10min']
        feature_cols = ['pred_prob', 'odds_10min', 'prob_market_10min', 'prob_diff_10min']
        # Handle NA
        df['odds_10min'] = df['odds_10min'].fillna(100) # Fallback for now
        df['prob_market_10min'] = df['prob_market_10min'].fillna(0)
        df['prob_diff_10min'] = df['prob_diff_10min'].fillna(0)
        
        df['meta_prob'] = model.predict(df[feature_cols])
    else:
        logger.warning(f"Meta model not found at {model_path}. Using base prob.")
        df['meta_prob'] = df['pred_prob']
        
    # Ensure Final Odds
    if 'odds_final' in df.columns:
        df['odds_calc'] = df['odds_final']
    elif 'odds' in df.columns:
        df['odds_calc'] = df['odds']
    else:
        df['odds_calc'] = 0
        
    return df

def simulate_strategy(df, strategy_name, filter_func):
    """
    Simulate a betting strategy.
    filter_func: lambda row: boolean
    """
    # Vectorized filter is faster
    mask = filter_func(df)
    bets = df[mask].copy()
    
    n_races_total = df['race_id'].nunique()
    n_bets = len(bets)
    n_races_bet = bets['race_id'].nunique()
    
    if n_bets == 0:
        return {
            'Strategy': strategy_name,
            'Bets': 0, 'Races': 0, 
            'Part%': 0.0, 'Hit%': 0.0, 'ROI%': 0.0, 'Profit': 0
        }
    
    hits = bets[bets['rank'] == 1]
    cost = n_bets * 100
    ret = (hits['odds_calc'] * 100).sum()
    
    return {
        'Strategy': strategy_name,
        'Bets': n_bets,
        'Races': int(n_races_bet),
        'Part%': (n_races_bet / n_races_total) * 100,
        'Hit%': (len(hits) / n_bets) * 100,
        'ROI%': (ret / cost) * 100,
        'Profit': int(ret - cost)
    }

def main():
    logger.info("🚀 Starting Phase T Step 5: EV Strategy Simulation")
    
    df = load_data()
    logger.info(f"Loaded {len(df)} rows. Total Races: {df['race_id'].nunique()}")
    
    # Calculate Expected Value (EV)
    # EV = Prob * Odds
    # Note: Using Meta-Prob (Conservative) vs Odds Final (Actual Return)
    # In reality, we use Odds 10min for decision making? Or Estimated Final Odds?
    # For this simulation, let's use Odds 10min to decide, but Final Odds to calculate return.
    # Because decision must be made BEFORE race.
    
    df['EV_10min'] = df['meta_prob'] * df['odds_10min']
    
    results = []
    
    # 1. Base Strategy (Prob Threshold)
    results.append(simulate_strategy(df, "Prob >= 0.20", lambda x: x['meta_prob'] >= 0.20))
    results.append(simulate_strategy(df, "Prob >= 0.15", lambda x: x['meta_prob'] >= 0.15))
    results.append(simulate_strategy(df, "Prob >= 0.10", lambda x: x['meta_prob'] >= 0.10))
    results.append(simulate_strategy(df, "Prob >= 0.08", lambda x: x['meta_prob'] >= 0.08))
    
    # 2. EV Strategy (EV >= X)
    # This should pick up low prob but high odds horses.
    # Theoretically EV > 1.0 is profit, but because model is conservative, maybe EV > 0.8 is enough?
    
    ev_thresholds = [0.8, 0.9, 1.0, 1.1, 1.2]
    for ev in ev_thresholds:
        # Constraint: meta_prob >= 0.05 (Don't buy extreme longshots with 1% chance)
        results.append(simulate_strategy(df, f"EV(10m) >= {ev}", 
                                         lambda x: (x['EV_10min'] >= ev) & (x['meta_prob'] >= 0.05)))
                                         
    # 4. Odds Drop Strategy (Market Follower)
    # Strategy: Buy if Odds Ratio (Final/10min) <= 0.7
    # Note: In real-time, we would compare Current Odds vs 10min ago. Assuming Final~Current.
    if 'odds_ratio_10min' in df.columns:
        results.append(simulate_strategy(df, "Odds Drop (Ratio <= 0.7)", 
                                         lambda x: x['odds_ratio_10min'] <= 0.7))
        results.append(simulate_strategy(df, "Odds Drop (Ratio <= 0.8)", 
                                         lambda x: x['odds_ratio_10min'] <= 0.8))                                 

    # Display Results
    res_df = pd.DataFrame(results)
    
    with open("strategy_results_2025.txt", "w") as f:
        f.write("=== Strategy Simulation Results (2025) ===\n")
        f.write(res_df.to_string(index=False, float_format=lambda x: "{:.1f}".format(x) if isinstance(x, float) else str(x)))
        f.write("\n\n")
        
    print("\n=== Saved results to strategy_results_2025.txt ===")
    # Let's look at "Prob >= 0.10" breakdown by Odds
    logger.info("\n--- Breakdown: Prob >= 0.10 (High Participation Candidate) ---")
    sub = df[df['meta_prob'] >= 0.10].copy()
    sub['odds_bin'] = pd.cut(sub['odds_calc'], bins=[1, 2, 5, 10, 20, 50, 100], right=False)
    grouped = sub.groupby('odds_bin', observed=False).apply(lambda x: pd.Series({
        'count': len(x),
        'roi': (x[x['rank']==1]['odds_calc'].sum() * 100) / (len(x)*100) if len(x)>0 else 0
    }))
    print(grouped)

if __name__ == "__main__":
    main()

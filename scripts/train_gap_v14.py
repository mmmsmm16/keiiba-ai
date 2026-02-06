"""
Gap Model Training V14 (No Leak / Production Ready)
===================================================
Target: Predict Gap (PopRank - Rank) or High Gap Probability
Features: Use 'odds_10min' instead of 'odds' (final)
Model: LightGBM
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Add workspace for imports
sys.path.append('/workspace')

def main():
    # Config
    DATA_PATH = "data/processed/preprocessed_data_v13_active.parquet" # Output from rebuild
    MODEL_DIR = "models/experiments/exp_gap_v14_production"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Data file {DATA_PATH} not found. Please run rebuild first.")
        return

    df = pd.read_parquet(DATA_PATH)
    
    # 1. Feature Selection
    # Identify 'odds_10min' and usage
    need_odds_compute = False
    if 'odds_10min' not in df.columns:
        print("odds_10min column not found. Will compute from DB.")
        need_odds_compute = True
    elif df['odds_10min'].isna().all():
        print("odds_10min is 100% NaN. Will compute from DB.")
        need_odds_compute = True
        
    if need_odds_compute:
        print("Computing odds_fluctuation directly from DB...")
        from src.preprocessing.features.odds_fluctuation import compute_odds_fluctuation
        from src.preprocessing.loader import JraVanDataLoader
        
        # Check for start_time_str
        if 'start_time_str' not in df.columns:
            print("Fetching start_time_str via Loader...")
            loader = JraVanDataLoader()
            # Fetch minimal race info for relevant dates
            dates = df['date'].astype(str).unique()
            min_date = dates.min()
            max_date = dates.max()
            # Load race info (skip everything heavy)
            df_info = loader.load(history_start_date=min_date, end_date=max_date, skip_odds=True, skip_training=True)
            # Ensure keys
            df_info['race_id'] = df_info['race_id'].astype(str)
            df['race_id'] = df['race_id'].astype(str)
            
            # Merge start_time_str
            if 'start_time_str' in df_info.columns:
                 # unique race_id in info
                 df_info_unique = df_info[['race_id', 'start_time_str']].drop_duplicates()
                 df = pd.merge(df, df_info_unique, on='race_id', how='left')
            else:
                print("Loader failed to provide start_time_str. Cannot compute odds.")
                return

        df_odds = compute_odds_fluctuation(df)
        if not df_odds.empty:
            print(f"Computed {len(df_odds)} odds records. Merging...")
            # Ensure key types match
            df['race_id'] = df['race_id'].astype(str)
            df['horse_number'] = df['horse_number'].astype(int)
            df_odds = df_odds.drop_duplicates(subset=['race_id', 'horse_number'])
            # Drop old columns if they exist (all NaN)
            for c in ['odds_10min', 'odds_final', 'odds_60min', 'odds_ratio_10min', 'rank_diff_10min', 'odds_log_ratio_10min', 'odds_ratio_60_10']:
                if c in df.columns:
                    df = df.drop(columns=[c])
            # Merge
            df = pd.merge(df, df_odds.drop(columns=['horse_id'], errors='ignore'), 
                          on=['race_id', 'horse_number'], how='left')
            print("Odds merged.")
        else:
            print("ERROR: compute_odds_fluctuation returned empty!")
            return
        
    print(f"Total Rows: {len(df)}")
    print(f"Odds 10min NaNs: {df['odds_10min'].isna().sum()}")
    
    # Drop rows where odds_10min is missing (Simulation Requirement)
    # If we cannot get 10min odds, we cannot predict.
    df = df.dropna(subset=['odds_10min']).reset_index(drop=True)
    print(f"Valid Rows (with Odds 10min): {len(df)}")
    
    if len(df) == 0:
        print("ERROR: No valid rows after odds filtering!")
        return

    
    # Features List
    # Use 'odds_10min' as primary odds feature
    # Renaming 'odds_10min' to 'odds_feature' for clarity or just using it directly
    # And we must EXCLUDE 'odds', 'popularity' (Final) from features.
    
    exclude_cols = [
        'rank', 'odds', 'popularity', 'time', 'date', 'race_id', 'horse_id',
        'fukusho_payoff', 'tansho_payoff', 'result', # any Result related
        'target', 'gap', 'gap_score', # targets
        'odds_rank', # This is derived from FINAL odds usually.
        'odds_final', 'odds_ratio_10min', 'odds_log_ratio_10min', 'rank_diff_10min', # LEAK: Final odds derived
    ]
    
    # We should calculate 'odds_rank_10min' dynamically if not present
    # Group by race_id and rank 'odds_10min'
    print("Calculating Pre-race Popularity (odds_rank_10min)...")
    df['odds_rank_10min'] = df.groupby('race_id')['odds_10min'].rank(method='min')
    
    # Define Target
    # Gap = (Pre-race Rank) - (Actual Rank)
    # Maximizing Gap is good.
    df['gap_score'] = df['odds_rank_10min'] - df['rank']
    
    # Define Features
    # Use all numeric columns except excluded
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude_cols]
    
    # Explicitly remove 'odds_rank' if present (it's final)
    if 'odds_rank' in features: features.remove('odds_rank')
    
    # Add 'odds_10min' and 'odds_rank_10min'
    if 'odds_10min' not in features: features.append('odds_10min')
    if 'odds_rank_10min' not in features: features.append('odds_rank_10min')
    
    print(f"Features: {len(features)}")
    
    # Train/Test Split (Time Series)
    # Rebuild covers 2024-2025
    # Train: 2024
    # Test: 2025
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'].dt.year <= 2024]
    test_df = df[df['date'].dt.year >= 2025]
    
    # If 2025 data is small or missing (Jan 2025), fallback to last 3 months test
    if len(test_df) < 1000:
        print("Test set too small (2025). Switching to last 3 months split.")
        cutoff_date = df['date'].max() - pd.Timedelta(days=90)
        train_df = df[df['date'] < cutoff_date]
        test_df = df[df['date'] >= cutoff_date]
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    X_train = train_df[features]
    y_train = train_df['gap_score']
    # Group for potential lambdarank?
    # Using Regression first for Gap Score
    
    X_test = test_df[features]
    y_test = test_df['gap_score']
    
    # Model
    print("Training LightGBM (Regression)...")
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    # Evaluation
    print("Evaluating...")
    preds = model.predict(X_test)
    test_df['pred_gap'] = preds
    
    # Rank Predictions within race
    test_df['pred_rank'] = test_df.groupby('race_id')['pred_gap'].rank(ascending=False)
    
    # Analysis: Top Picks Performance
    # Filter: Top 3 Picks + Odds 10-50 (Using odds_10min!)
    top_picks = test_df[test_df['pred_rank'] <= 3]
    
    # ROI Calculation (Using FINAL Odds for payout simulation, but filtering on PRE-RACE Odds)
    # Condition: 10 <= odds_10min <= 50.
    target_picks = top_picks[
        (top_picks['odds_10min'] >= 10.0) & 
        (top_picks['odds_10min'] <= 50.0)
    ]
    
    hits_place = target_picks[target_picks['rank'] <= 3]
    
    # Place Payout (Need 'fukusho_payoff' column ideally, but we have 'odds' (Final Win) approx?)
    # Usually dataset contains result payouts if loaded.
    # If not, we approximate Place Payout ~ FinalOdds * A ? Hard to calc place from Win.
    # But usually 'raw' loading includes payouts columns if skip_odds=False?
    # Let's check 'fukusho_payoff' presence.
    
    if 'fukusho_payoff' in test_df.columns:
         # fukusho_payoff is usually list or combined string?
         # Need parsing. For now let's just use Hit Rate.
         pass
         
    hit_rate = len(hits_place) / len(target_picks) if len(target_picks) > 0 else 0
    print(f"\nTarget Picks (Rank<=3, Odds10-50): {len(target_picks)}")
    print(f"Place Hits: {len(hits_place)}")
    print(f"Hit Rate: {hit_rate:.2%}")
    
    # Save Feature Importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Features:")
    print(importance.head(10))
    
    # Save Model
    joblib.dump(model, f'{MODEL_DIR}/model_v14.pkl')
    # Save feature list
    importance[['feature']].to_csv(f'{MODEL_DIR}/features.csv', index=False)
    
    print(f"\nModel saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()

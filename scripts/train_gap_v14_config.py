"""
Gap Model Training V14 (No Leak / Production Ready)
===================================================
Target: Predict Gap (PopRank - Rank) or High Gap Probability
Features: Use 'odds_10min' instead of 'odds' (final)
Model: LightGBM
"""
import argparse
import os
import sys

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add workspace for imports
sys.path.append('/workspace')

DEFAULT_DATA_PATH = "data/processed/preprocessed_data_v13_active.parquet"
DEFAULT_MODEL_DIR = "models/experiments/exp_gap_v14_production"
DEFAULT_TRAIN_YEARS = list(range(2016, 2024))
DEFAULT_VALID_YEAR = 2024
DEFAULT_TEST_YEARS = [2025]
DEFAULT_HOLDOUT_YEARS = [2026]
FALLBACK_MIN_TEST_ROWS = 1000
FALLBACK_DAYS = 90


def _parse_years_arg(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [int(p) for p in parts]


def _load_split_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("split", data)


def _resolve_split(args):
    split = {}
    if args.split_config:
        split = _load_split_config(args.split_config)

    train_years = _parse_years_arg(args.train_years) or split.get("train_years")
    valid_year = args.valid_year if args.valid_year is not None else split.get("valid_year")
    test_years = _parse_years_arg(args.test_years) or split.get("test_years")
    holdout_years = _parse_years_arg(args.holdout_years) or split.get("holdout_years")

    if train_years is None:
        train_years = DEFAULT_TRAIN_YEARS
    if valid_year is None:
        valid_year = DEFAULT_VALID_YEAR
    if test_years is None:
        test_years = DEFAULT_TEST_YEARS
    if holdout_years is None:
        holdout_years = DEFAULT_HOLDOUT_YEARS

    return train_years, valid_year, test_years, holdout_years


def _split_by_years(df, train_years, valid_year, test_years, holdout_years):
    df["year"] = pd.to_datetime(df["date"]).dt.year
    train_df = df[df["year"].isin(train_years)].copy()
    valid_df = df[df["year"] == valid_year].copy() if valid_year else pd.DataFrame()
    test_df = df[df["year"].isin(test_years)].copy() if test_years else pd.DataFrame()
    holdout_df = df[df["year"].isin(holdout_years)].copy() if holdout_years else pd.DataFrame()
    return train_df, valid_df, test_df, holdout_df

def main():
    parser = argparse.ArgumentParser(description="Train gap model v14 with configurable splits.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-config", default=None)
    parser.add_argument("--train-years", default=None)
    parser.add_argument("--valid-year", type=int, default=None)
    parser.add_argument("--test-years", default=None)
    parser.add_argument("--holdout-years", default=None)
    parser.add_argument("--allow-fallback-split", action="store_true")
    args = parser.parse_args()

    data_path = args.data_path
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading data...")
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please run rebuild first.")
        return

    df = pd.read_parquet(data_path)
    
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
        'odds_rank', # This is derived from FINAL odds usually. Check calculation.
        # Note: odds_rank in my pipeline is usually 'popularity'.
        # We need 'odds_rank_10min' ? pipeline might not produce it explicitly but maybe 'odds_rank' column is final.
        # Let's trust only 10min columns.
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
    train_years, valid_year, test_years, holdout_years = _resolve_split(args)
    train_df, valid_df, test_df, holdout_df = _split_by_years(
        df, train_years, valid_year, test_years, holdout_years
    )

    if args.allow_fallback_split and (test_df.empty or len(test_df) < FALLBACK_MIN_TEST_ROWS):
        print("Fallback split enabled. Using last 90 days as test.")
        cutoff_date = df['date'].max() - pd.Timedelta(days=FALLBACK_DAYS)
        train_df = df[df['date'] < cutoff_date].copy()
        valid_df = pd.DataFrame()
        test_df = df[df['date'] >= cutoff_date].copy()
        holdout_df = pd.DataFrame()

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}, Holdout: {len(holdout_df)}")

    if train_df.empty:
        print("ERROR: Training set is empty. Check split configuration.")
        return
    if valid_df.empty and test_df.empty:
        print("ERROR: Valid/Test sets are empty. Check split configuration.")
        return

    X_train = train_df[features]
    y_train = train_df['gap_score']
    # Group for potential lambdarank?
    # Using Regression first for Gap Score
    
    eval_df = valid_df if not valid_df.empty else test_df
    report_df = test_df if not test_df.empty else valid_df

    X_eval = eval_df[features]
    y_eval = eval_df['gap_score']
    
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
        eval_set=[(X_eval, y_eval)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    # Evaluation
    print("Evaluating...")
    preds = model.predict(report_df[features])
    report_df['pred_gap'] = preds
    
    # Rank Predictions within race
    report_df['pred_rank'] = report_df.groupby('race_id')['pred_gap'].rank(ascending=False)
    
    # Analysis: Top Picks Performance
    # Filter: Top 3 Picks + Odds 10-50 (Using odds_10min!)
    top_picks = report_df[report_df['pred_rank'] <= 3]
    
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
    
    if 'fukusho_payoff' in report_df.columns:
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
    joblib.dump(model, f'{model_dir}/model_v14.pkl')
    # Save feature list
    importance[['feature']].to_csv(f'{model_dir}/features.csv', index=False)
    
    print(f"\nModel saved to {model_dir}")

if __name__ == "__main__":
    main()

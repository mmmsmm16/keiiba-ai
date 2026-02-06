"""
JRA Incremental Walk-Forward Backtest (2025)

Methodology:
1. Base Phase: Load 2014-2024 data (JRA-VAN + T-10 Odds Snapshots), run FULL preprocessing.
   - Initialize IncrementalCategoryAggregator state from this base.
2. Incremental Phase: Loop through 2025 months.
   - Load monthly JRA-VAN.
   - Inject T-10 Odds.
   - Update feature set efficiently.
   - Train (Rolling Window or Expanding) & Predict.
   - Simulate Betting.
"""
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import logging
import gc
from sklearn.linear_model import LogisticRegression

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Imports
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.aggregators import HistoryAggregator
from src.preprocessing.category_aggregators import CategoryAggregator
from src.preprocessing.incremental_aggregators import IncrementalCategoryAggregator
from src.preprocessing.advanced_features import AdvancedFeatureEngineer
from src.preprocessing.experience_features import ExperienceFeatureEngineer
from src.preprocessing.relative_features import RelativeFeatureEngineer
from src.preprocessing.opposition_features import OppositionFeatureEngineer
from src.preprocessing.odds_features import OddsFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
REPORT_DIR = "reports/jra/wf_incremental"
MODEL_DIR = "models/production/jra/wf_incremental"
PRED_DIR = "models/production/jra/wf_incremental/predictions"
SNAPSHOT_DIR = "data/odds_snapshots"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# Training params (Standard)
PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 76,
    'min_data_in_leaf': 53,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.6,
    'bagging_freq': 7,
    'lambda_l1': 1.5e-05,
    'lambda_l2': 0.05,
    'random_state': 42,
    'verbose': -1
}

def load_t10_odds(years):
    """Load T-10 odds snapshots for specified years."""
    df_list = []
    for year in years:
        path = os.path.join(SNAPSHOT_DIR, str(year), "odds_T-10.parquet")
        if os.path.exists(path):
            try:
                # Load only Win odds
                # Columns: race_id, ticket_type, combination, odds, ninki
                df = pd.read_parquet(path, filters=[('ticket_type', '=', 'win')])
                df = df[['race_id', 'combination', 'odds', 'ninki']].copy()
                df['race_id'] = df['race_id'].astype(str)
                # Parse combination as horse_number
                df['horse_number'] = pd.to_numeric(df['combination'], errors='coerce')
                # Renaming
                df.rename(columns={'ninki': 'popularity'}, inplace=True)
                df_list.append(df[['race_id', 'horse_number', 'odds', 'popularity']])
                logger.info(f"Loaded T-10 odds for {year}")
            except Exception as e:
                logger.warning(f"Failed to load T-10 odds for {year}: {e}")
        else:
            logger.warning(f"T-10 odds not found for {year}: {path}")
            
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)

def inject_t10_odds(main_df, t10_df):
    """Merge T-10 odds into main DataFrame, overwriting existing odds/popularity."""
    if t10_df.empty:
        return main_df
    
    # Ensure keys match
    main_df['race_id'] = main_df['race_id'].astype(str)
    main_df['horse_number'] = pd.to_numeric(main_df['horse_number'], errors='coerce')
    t10_df['race_id'] = t10_df['race_id'].astype(str)
    t10_df['horse_number'] = pd.to_numeric(t10_df['horse_number'], errors='coerce')
    
    # Merge
    merged = pd.merge(main_df, t10_df, on=['race_id', 'horse_number'], suffixes=('', '_t10'), how='left')
    
    # Overwrite if T-10 exists
    merged['odds'] = merged['odds_t10'].fillna(merged['odds'])
    merged['popularity'] = merged['popularity_t10'].fillna(merged['popularity'])
    
    # Drop temp cols
    merged.drop(columns=['odds_t10', 'popularity_t10'], inplace=True, errors='ignore')
    
    # Recalculate or fill dependent?
    # Note: 'odds' might be NaN in main_df if JRA-VAN null. T-10 might fill it.
    
    return merged

def get_features(df):
    """Extract features, including odds/popularity and time-series odds."""
    drop_cols = [
        'race_id', 'date', 'horse_id', 'horse_name', 'title',
        'rank', 'target', 'rank_str', 'year',
        'time', 'raw_time', 'passing_rank', 'last_3f',
        # 'odds', 'popularity', <- ENABLED
        'weight', 'weight_diff_val', 'weight_diff_sign',
        'winning_numbers', 'payout', 'ticket_type',
        'pass_1', 'pass_2', 'pass_3', 'pass_4',
        'horse_number', 'frame_number',
        
        # Leak-prone (but odds/pop are now safe T-10)
        'slow_start_recovery', 'pace_disadvantage', 'wide_run',
        'track_bias_disadvantage', 'outer_frame_disadv',
        'odds_race_rank', 'popularity_race_rank',
        'odds_deviation', 'popularity_deviation',
        'trend_win_inner_rate', 'trend_win_mid_rate', 'trend_win_outer_rate',
        'trend_win_front_rate', 'trend_win_fav_rate',
        
        # Lag features are safe (previous race)
        # 'lag1_odds', 'lag1_popularity',
        
        # Speed Index temps
        'time_index', 'last_3f_index',
        
        # Keep intermediate T-30 cols but drop raw ninki (use rates instead)
        'ninki_t_30', 'ninki_t_10',
    ]
    
    # Time-series odds features to KEEP (not drop):
    # - odds_change_t30_t10, odds_diff_t30_t10, ninki_change_t30_t10
    # - is_odds_drop, is_odds_surge, log_odds_t_30, odds_t_30
    
    cat_cols = ['jockey_id', 'trainer_id', 'sire_id', 'mare_id', 'venue', 'weather', 'surface', 'state']
    
    X = df.drop(columns=drop_cols, errors='ignore').copy()
    
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype('category')
            
    for col in X.columns:
        if X[col].dtype.name == 'category':
            continue
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
    return X, X.columns.tolist()

def run_incremental_backtest():
    # Cache management - v16: Updated cache name for time-series odds features
    CACHE_BASE_DF = os.path.join(MODEL_DIR, "base_df_2014_2024_v16_odds.parquet")
    CACHE_STATE = os.path.join(MODEL_DIR, "base_state_2014_2024_v16_odds.joblib")
    
    # Initialize processors
    loader = JraVanDataLoader()
    cleanser = DataCleanser()
    engineer = FeatureEngineer()
    hist_agg = HistoryAggregator()
    cat_agg_std = CategoryAggregator()
    adv_eng = AdvancedFeatureEngineer()
    exp_eng = ExperienceFeatureEngineer()
    rel_eng = RelativeFeatureEngineer()
    opp_eng = OppositionFeatureEngineer()
    inc_cat_agg = IncrementalCategoryAggregator()
    odds_eng = OddsFeatureEngineer(SNAPSHOT_DIR, odds_drop_threshold=0.20)  # Phase 16: Time-series odds
    
    master_df = None
    
    # 1. Base Phase (2014-2024)
    logger.info("=== Phase 1: Base Data Processing (2014-2024 with T-10) ===")
    
    if os.path.exists(CACHE_BASE_DF) and os.path.exists(CACHE_STATE):
        logger.info("Found cached base data (T-10). Loading...")
        master_df = pd.read_parquet(CACHE_BASE_DF)
        inc_cat_agg.load_state(CACHE_STATE)
        logger.info(f"Loaded master_df: {len(master_df)} rows")
    else:
        logger.info("No cache found. Processing from scratch...")
        # Load Raw Data
        base_df = loader.load(history_start_date="2014-01-01", end_date="2024-12-31")
        
        # Inject T-10 Odds BEFORE engineering
        t10_data = load_t10_odds(range(2014, 2025)) # 2014-2024
        logger.info(f"Loaded T-10 Odds rows: {len(t10_data)}")
        base_df = inject_t10_odds(base_df, t10_data)
        logger.info("Injected T-10 Odds into Base Data.")
        
        # Preprocess
        base_df = cleanser.cleanse(base_df)
        base_df = engineer.add_features(base_df) # Now uses T-10 odds
        base_df = hist_agg.aggregate(base_df)
        base_df = cat_agg_std.aggregate(base_df)
        inc_cat_agg.fit(base_df)
        base_df = adv_eng.add_features(base_df) # Deviation features use T-10
        base_df = exp_eng.add_features(base_df)
        base_df = rel_eng.add_features(base_df)
        base_df = opp_eng.add_features(base_df)
        
        # Phase 16: Add time-series odds features (T-30 vs T-10)
        logger.info("Adding time-series odds features (T-30 vs T-10)...")
        t30_data = odds_eng.load_odds_snapshot(range(2014, 2025), "T-30")
        t10_data_for_features = odds_eng.load_odds_snapshot(range(2014, 2025), "T-10")
        base_df = odds_eng.add_features(base_df, t30_data, t10_data_for_features)
        
        master_df = base_df.copy()
        del base_df
        gc.collect()
        
        logger.info(f"Base processed. Master size: {len(master_df)}")
        logger.info("Saving base data to cache...")
        master_df.to_parquet(CACHE_BASE_DF)
        inc_cat_agg.save_state(CACHE_STATE)
        logger.info("Cache saved.")
    
    # 2. Incremental Phase (2025 Full Year)
    months = pd.date_range("2025-01-01", "2025-12-31", freq="MS").strftime("%Y-%m").tolist()
    results = []
    
    # Load 2025 T-10 for simulation (and injection)
    # We can perform injection per month to keep memory low
    t10_2025_all = load_t10_odds([2025])
    
    # Phase 16: Load 2025 T-30 for time-series odds features
    t30_2025_all = odds_eng.load_odds_snapshot([2025], "T-30")
    t10_2025_features = odds_eng.load_odds_snapshot([2025], "T-10")
    
    for month_str in months:
        logger.info(f"\n=== Processing Month: {month_str} ===")
        
        pred_cache_path = os.path.join(PRED_DIR, f"preds_{month_str}_v16_odds.parquet")
        test_df = None
        
        # NOTE: If we use cache, we skip injection/engineering. 
        # v16: Updated cache name for time-series odds features.
        
        if os.path.exists(pred_cache_path):
            logger.info("Found prediction cache. Loading...")
            test_df = pd.read_parquet(pred_cache_path)
        else:
            # Load Month Raw
            start_date = f"{month_str}-01"
            import calendar
            y, m = map(int, month_str.split('-'))
            last_day = calendar.monthrange(y, m)[1]
            end_date = f"{month_str}-{last_day}"
            
            m_df = loader.load(history_start_date=start_date, end_date=end_date)
            if len(m_df) == 0: continue
            
            # Inject T-10 (For Feature Engineering & Prediction)
            m_df = inject_t10_odds(m_df, t10_2025_all)
            
            m_df = cleanser.cleanse(m_df)
            m_df = engineer.add_features(m_df)
            m_df_with_cat = inc_cat_agg.transform_update(m_df)
            
            # Optimization: Context Slicing
            # Instead of appending raw to master and recalculating everything,
            # we slice recent context from master, append raw, calculate, then extract.
            
            # Context: Last 2 years
            ctx_date = (pd.to_datetime(start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
            context_mask = (master_df['date'] >= ctx_date)
            context_df = master_df[context_mask].copy()
            
            # Combine
            proc_df = pd.concat([context_df, m_df_with_cat], ignore_index=True)
            proc_df = proc_df.sort_values(['date', 'race_id'])
            
            # Engineer
            # Note: aggregators will recalculate on proc_df (2 years). Fast.
            # Existing features in context_df will be overwritten/updated, which is fine (consistency).
            proc_df = hist_agg.aggregate(proc_df)
            proc_df = adv_eng.add_features(proc_df)
            proc_df = exp_eng.add_features(proc_df)
            proc_df = rel_eng.add_features(proc_df)
            proc_df = opp_eng.add_features(proc_df)
            
            # Phase 16: Add time-series odds features for 2025 data
            proc_df = odds_eng.add_features(proc_df, t30_2025_all, t10_2025_features)
            
            # Extract Finished Test Data
            test_mask = (proc_df['date'] >= start_date) & (proc_df['date'] <= end_date)
            test_df = proc_df[test_mask].copy()
            
            # Append finished test data to master_df for future context & training
            if not test_df.empty:
                master_df = pd.concat([master_df, test_df], ignore_index=True)
                master_df = master_df.sort_values(['date', 'race_id'])
            
            # Prepare Training Data (Full Master)
            # Use data strictly before this month
            train_mask = (master_df['date'] < start_date)
            full_train_df = master_df[train_mask].copy()
            
            # Calibration Split
            full_train_df['target'] = full_train_df['rank'].apply(lambda r: 3 if r == 1 else (2 if r == 2 else (1 if r == 3 else 0)))
            full_train_df = full_train_df.sort_values(['date', 'race_id'])
            
            u_races = full_train_df['race_id'].unique()
            split_idx = int(len(u_races) * 0.8)
            train_races = u_races[:split_idx]
            calib_races = u_races[split_idx:]
            
            train_df = full_train_df[full_train_df['race_id'].isin(train_races)].copy()
            calib_df = full_train_df[full_train_df['race_id'].isin(calib_races)].copy()
            
            # Train
            train_df = train_df.sort_values('race_id')
            X_train, cols = get_features(train_df)
            model = lgb.train(PARAMS, lgb.Dataset(X_train, train_df['target'], group=train_df.groupby('race_id').size().values), num_boost_round=100)
            
            # Calibrate (Logistic)
            X_calib, _ = get_features(calib_df)
            # Ensure cols match
            for c in cols:
                if c not in X_calib.columns: X_calib[c] = 0
            calib_preds = model.predict(X_calib[cols])
            
            calibrator = LogisticRegression(random_state=42, solver='lbfgs')
            calib_targets = (calib_df['rank'] == 1).astype(int).values
            calibrator.fit(calib_preds.reshape(-1, 1), calib_targets)
            
            logger.info(f"Calib Input Mean: {np.mean(calib_preds):.4f}, Std: {np.std(calib_preds):.4f}")
            logger.info(f"Calib Target Mean: {np.mean(calib_targets):.4f}")
            logger.info(f"LR Coef: {calibrator.coef_[0][0]:.4f}, Intercept: {calibrator.intercept_[0]:.4f}")
            
            # Predict
            X_test, _ = get_features(test_df)
            for c in cols:
                 if c not in X_test.columns: X_test[c] = 0
            
            raw_preds = model.predict(X_test[cols])
            calib_probs = calibrator.predict_proba(raw_preds.reshape(-1, 1))[:, 1]
            
            test_df['pred_prob'] = raw_preds
            test_df['calib_prob'] = calib_probs
            test_df['odds'] = test_df['odds'].fillna(0) # Ensure no NaN for betting
            
            test_df.to_parquet(pred_cache_path)
            
            # Feature Importance (First month only)
            if month_str == "2025-01":
                imp = pd.DataFrame({'feature': cols, 'gain': model.feature_importance(importance_type='gain')}).sort_values('gain', ascending=False)
                logger.info(f"Features:\n{imp.head(10)}")

            del train_df, calib_df, full_train_df, X_train, X_calib, X_test
            gc.collect()
            
        # Simulate (using test_df which now has T-10 odds injected)
        monthly_invest = 0
        monthly_return = 0
        monthly_bets = 0
        
        test_df['ev'] = test_df['calib_prob'] * test_df['odds']
        bet_df = test_df[test_df['ev'] > 1.0].copy() # Back to 1.0 threshold
        
        for _, row in bet_df.iterrows():
            monthly_invest += 100
            monthly_bets += 1
            if row['rank'] == 1:
                monthly_return += 100 * row['odds']
                
        profit = monthly_return - monthly_invest
        roi = (monthly_return / monthly_invest * 100) if monthly_invest > 0 else 0
        logger.info(f"{month_str} Results: Bets {monthly_bets}, ROI {roi:.1f}%")
        
        results.append({
            'month': month_str,
            'bets': monthly_bets,
            'invest': monthly_invest,
            'return': monthly_return,
            'profit': profit,
            'roi': roi
        })

    # Summary
    res_df = pd.DataFrame(results)
    total_invest = res_df['invest'].sum()
    total_return = res_df['return'].sum()
    total_profit = res_df['profit'].sum()
    total_roi = (total_return / total_invest * 100) if total_invest > 0 else 0
    
    report_path = os.path.join(REPORT_DIR, "incremental_summary_t10.md")
    with open(report_path, 'w') as f:
        f.write("# JRA Incremental Walk-Forward 2025 (T-10 Odds)\n\n")
        f.write(f"**Total ROI: {total_roi:.2f}%**\n")
        f.write(f"Profit: {total_profit:+,.0f}\n\n")
        f.write(res_df.to_markdown(index=False))
        
    logger.info("All Done.")

if __name__ == "__main__":
    run_incremental_backtest()

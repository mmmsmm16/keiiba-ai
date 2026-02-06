"""
JRA Incremental Walk-Forward Backtest with Modular Pipeline (Phase 16)

階層的キャッシュパイプラインを使用したバックテスト。
特徴量追加時に変更箇所以降のみ再処理することで処理時間を大幅短縮。
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
from sklearn.isotonic import IsotonicRegression  # Better calibration for ranking models
import calendar

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
from src.preprocessing.odds_features import OddsFeatureEngineer
from src.preprocessing.pace_features import PaceFeatureEngineer
from src.preprocessing.relative_features import RelativeFeatureEngineer
from src.preprocessing.opposition_features import OppositionFeatureEngineer
from src.preprocessing.odds_features import OddsFeatureEngineer
from src.preprocessing.pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
CACHE_DIR = "data/cache"
REPORT_DIR = "reports/jra/wf_incremental"
MODEL_DIR = "models/production/jra/wf_incremental"
PRED_DIR = "models/production/jra/wf_incremental/predictions"
SNAPSHOT_DIR = "data/odds_snapshots"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# Training params (Optuna Optimized - 2025-12-19)
PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'boosting_type': 'gbdt',
    'learning_rate': 0.044449,        # Optimized (was 0.05)
    'num_leaves': 65,                 # Optimized (was 31)
    'min_data_in_leaf': 83,           # Optimized (was 100)
    'feature_fraction': 0.860430,     # Optimized (was 0.8)
    'bagging_fraction': 0.787839,     # Optimized (was 0.7)
    'bagging_freq': 5,                # Optimized (was 5)
    'lambda_l1': 0.010220,            # Optimized (was 0.1)
    'lambda_l2': 0.000172,            # Optimized (was 0.1)
    'random_state': 42,
    'verbose': -1
}




DROP_FEATURES = [
    'mare_id', 'sire_id', 'trainer_id', 'jockey_id', 'bms_id',
    'odds', 'popularity',
    'owner_id', 'breeder_id',
    # P0: No odds model - exclude past odds/popularity
    'lag1_odds', 'lag1_popularity',
    # P3: Redundant features
    'race_nige_horse_count', 'race_pace_cat', 'nige_candidate_count', 'senkou_ratio',
    'is_long_break', 'is_weight_changed_huge', 'is_class_up',
    # Strict leak prevention: Exclude current-race outcome data
    'time', 'raw_time', 'last_3f', 'time_diff',
    'honshokin', 'fukashokin',
    'passing_rank', 'pass_1', 'pass_2', 'pass_3', 'pass_4',
    'slow_start_recovery', 'pace_disadvantage', 'wide_run', 
    'outer_frame_disadv', 'track_bias_disadvantage',
    'relative_strength', 'relative_popularity_rank', 'estimated_place_rate',
    'race_avg_prize', 'n_horses',
    # Importance 0 Features (v18 Pruning)
    'lag1_last_3f_is_missing', 'momentum_slope', 'surface', 'weather', 'state',
    'mean_rank_all_is_missing', 'mean_time_diff_5_is_missing', 'title',
    'race_opponent_strength_is_missing', 'lag1_time_diff_is_missing',
    'mean_last_3f_5_is_missing', 'weight_diff_sign', 'abnormal_code', 'horse_name',
    'distance_category', 'sex', 'mean_rank_norm_5_is_missing', 'distance_type',
    'lag1_rank_is_missing', 'frame_zone', 'direction', 'first_distance_cat',
    'weight_is_missing', 'year', 'lag1_rank_norm_is_missing'
]


import re
import glob
from sklearn.model_selection import KFold
from scipy.special import expit

def optimize_residual_weight(y_true, pred_m, pred_r, odds, step_coarse=0.1, step_fine=0.01):
    """
    Optimize weight 'w' for: score = w * sigmoid(pred_m) + (1-w) * sigmoid(pred_r)
    Objective: Maximize Profit with fixed strategy (EV > 1.0, 100 yen bet)
    """
    best_w = 0.5
    best_profit = -float('inf')
    
    # Force sigmoid conversion for LambdaRank scores (LambdaRank outputs raw scores, not probabilities)
    # This is critical for EV calculation which expects probability-like values.
    # Note: If models were classified as binary, they would output probability by default (0-1).
    # But Phase 20 models are 'lambdarank' with 'ndcg', so outputs are unbounded scores.
    prob_m = expit(pred_m)
    prob_r = expit(pred_r)
    
    # Coarse search
    w_list = np.arange(0.0, 1.01, step_coarse)
    
    for w in w_list:
        p_blend = w * prob_m + (1 - w) * prob_r
        
        # Fixed strategy: Bet where (p_blend * odds) > 1.0
        bet_mask = (p_blend * odds) > 1.0
        n_bets = bet_mask.sum()
        
        if n_bets == 0:
            profit = 0
        else:
            hits = (bet_mask & (y_true == 1))
            ret = odds[hits].sum() * 100
            inv = n_bets * 100
            profit = ret - inv
        
        if profit > best_profit:
            best_profit = profit
            best_w = w
            
    # Fine search around best_w
    w_min = max(0.0, best_w - step_coarse)
    w_max = min(1.0, best_w + step_coarse)
    
    # Avoid 0 step
    if w_max > w_min:
        w_list_fine = np.arange(w_min, w_max + step_fine/2, step_fine)
        for w in w_list_fine:
            p_blend = w * prob_m + (1 - w) * prob_r
            bet_mask = (p_blend * odds) > 1.0
            n_bets = bet_mask.sum()
            
            if n_bets == 0:
                profit = 0
            else:
                hits = (bet_mask & (y_true == 1))
                ret = odds[hits].sum() * 100
                inv = n_bets * 100
                profit = ret - inv
                
            if profit > best_profit:
                best_profit = profit
                best_w = w
            
    return best_w

def get_oof_predictions(params, X, y, cv=5):
    """
    Generate Out-Of-Fold predictions for the entire dataset X.
    """
    kf = KFold(n_splits=cv, shuffle=False)
    oof_preds = np.zeros(len(X))
    
    # Check if X is DataFrame
    if hasattr(X, 'values'):
        X_vals = X
    else:
        X_vals = pd.DataFrame(X) # Should be DF usually
        
    for train_index, val_index in kf.split(X):
        # Use iloc for DataFrame
        X_tr = X.iloc[train_index]
        y_tr = y.iloc[train_index]
        X_val = X.iloc[val_index]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        
        # Train
        m = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain], # Validation on train just for logging? No, just train.
            callbacks=[lgb.log_evaluation(0)]
        )
        oof_preds[val_index] = m.predict(X_val)
        
    return oof_preds

def check_leakage(X_cols):
    """Enhanced Gate to prevent odds leakage (Regex based)"""
    # Deny only strictly future/target columns
    deny_pattern = re.compile(r"(final|確定)")
    # Allow pattern not strictly needed if we don't deny 'odds', but keep for safety/future
    allow_pattern = re.compile(r"^(odds_t_|popularity_t_|log_odds_t_)")
    
    leaked = []
    for c in X_cols:
        if deny_pattern.search(c) and not allow_pattern.match(c):
            leaked.append(c)
            
    if leaked:
        raise ValueError(f"CRITICAL: Leaked features detected in training set: {leaked}")
    # logger.info("✅ Leakage Gate Passed")

def filter_features(X: pd.DataFrame, mode: str = 'full') -> pd.DataFrame:
    """Filter features based on mode"""
    if mode == 'full':
        return X
    
    # Identify odds features
    odds_cols = [c for c in X.columns if 'odds' in c or 'popularity' in c]
    
    if mode == 'odds_only':
        # Keep only odds + basic info (n_horses, etc - assumed basic)
        # Actually, let's allow 'n_horses' and 'venue' etc?
        # User defined: {odds_t_10, odds_t_30, log_odds_t_30, n_horses}
        # Ideally we should drop all domain features.
        # Simple heuristic: Keep odds_cols AND meta-data (handled outside)
        # But X contains everything.
        # Let's drop non-odds columns except a few basics if possible, 
        # but determining "basic" is hard.
        # Strategy: Keep odds_cols. Drop everything else that looks like a feature.
        # Safest: Drop columns NOT in odds_cols AND NOT in basic properties.
        # For simplicity in this script, we just SELECT odds components.
        keep_cols = odds_cols + ['n_horses', 'total_races', 'interval'] # Minimal basics
        # Check if they exist
        final_cols = [c for c in keep_cols if c in X.columns]
        return X[final_cols]
    
    elif mode == 'no_odds':
        # Drop all odds columns
        return X.drop(columns=odds_cols, errors='ignore')
        
    elif mode == 'residual':
        # This function is not used for 'residual' mode splitting logic directly here
        # Residual mode uses two models.
        pass
        
    return X

# ============================================================
# Step Processors
# ============================================================

def step_load_raw(start_date: str = "2014-01-01", end_date: str = "2024-12-31") -> pd.DataFrame:
    """Step 1: Raw data load"""
    loader = JraVanDataLoader()
    return loader.load(history_start_date=start_date, end_date=end_date, jra_only=True)

def step_cleanse(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Data cleansing"""
    cleanser = DataCleanser()
    return cleanser.cleanse(df)

def step_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Basic feature engineering"""
    engineer = FeatureEngineer()
    df = engineer.add_features(df)
    
    # Speed Index用のtrack_condition_codeエイリアス追加
    if 'state_num' in df.columns and 'track_condition_code' not in df.columns:
        df['track_condition_code'] = df['state_num']
    
    return df

def step_inject_odds(df: pd.DataFrame, years: range = None) -> pd.DataFrame:
    """Step 3.5: Inject T-10 odds"""
    if years is None:
        years = range(2014, 2025)
    
    df_list = []
    for year in years:
        path = os.path.join(SNAPSHOT_DIR, str(year), "odds_T-10.parquet")
        if os.path.exists(path):
            try:
                t10 = pd.read_parquet(path, filters=[('ticket_type', '=', 'win')])
                t10 = t10[['race_id', 'combination', 'odds', 'ninki']].copy()
                t10['race_id'] = t10['race_id'].astype(str)
                t10['horse_number'] = pd.to_numeric(t10['combination'], errors='coerce')
                t10.rename(columns={'ninki': 'popularity'}, inplace=True)
                df_list.append(t10[['race_id', 'horse_number', 'odds', 'popularity']])
            except Exception as e:
                logger.warning(f"Failed to load T-10 odds for {year}: {e}")
    
    if not df_list:
        # T-10データがない場合、確定オッズが残らないようにNaN化する (Leak Prevention)
        logger.warning(f"No T-10 odds found for years {years}. Clearing 'odds' and 'popularity' to prevent leakage.")
        if 'odds' in df.columns:
            df['odds'] = np.nan
        if 'popularity' in df.columns:
            df['popularity'] = np.nan
        return df
    
    t10_df = pd.concat(df_list, ignore_index=True)
    
    df['race_id'] = df['race_id'].astype(str)
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')
    t10_df['race_id'] = t10_df['race_id'].astype(str)
    t10_df['horse_number'] = pd.to_numeric(t10_df['horse_number'], errors='coerce')
    
    merged = pd.merge(df, t10_df, on=['race_id', 'horse_number'], suffixes=('', '_t10'), how='left')
    # [v16.1 Fix] 確定オッズへのフォールバックを廃止。T-10がない場合はNaNとする。
    if 'odds_t10' in merged.columns:
        merged['odds'] = merged['odds_t10']
        merged['popularity'] = merged['popularity_t10']
        merged.drop(columns=['odds_t10', 'popularity_t10'], inplace=True, errors='ignore')
    else:
        # マージ後もない場合（念のため）
        merged['odds'] = np.nan
        merged['popularity'] = np.nan
    
    return merged

def step_history_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: History aggregation"""
    hist_agg = HistoryAggregator()
    return hist_agg.aggregate(df)

def step_category_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Category aggregation"""
    cat_agg = CategoryAggregator()
    return cat_agg.aggregate(df)

def step_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: Advanced features"""
    from src.preprocessing.rating_features import RatingFeatureEngineer
    
    adv_eng = AdvancedFeatureEngineer()
    exp_eng = ExperienceFeatureEngineer()
    rel_eng = RelativeFeatureEngineer()
    opp_eng = OppositionFeatureEngineer()
    rating_eng = RatingFeatureEngineer()  # P2: Elo Rating
    
    df = adv_eng.add_features(df)
    df = exp_eng.add_features(df)
    df = rel_eng.add_features(df)
    df = opp_eng.add_features(df)
    df = rating_eng.add_features(df)  # P2: Elo Rating + Field Strength
    
    return df


def step_bloodline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5b: Bloodline features"""
    from src.preprocessing.bloodline_features import BloodlineFeatureEngineer
    eng = BloodlineFeatureEngineer()
    return eng.add_features(df)

def step_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5d: Profile features"""
    from src.preprocessing.profile_features import JockeyTrainerProfileEngineer
    eng = JockeyTrainerProfileEngineer()
    return eng.add_features(df)

def step_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5c: Pace Analysis features"""
    pace_eng = PaceFeatureEngineer()
    return pace_eng.add_features(df)

def step_timeseries_odds(df: pd.DataFrame, years: range = None) -> pd.DataFrame:
    """Step 7: Time-series odds features"""
    if years is None:
        years = range(2014, 2025)
    
    odds_eng = OddsFeatureEngineer(SNAPSHOT_DIR, odds_drop_threshold=0.20)
    t30_odds = odds_eng.load_odds_snapshot(years, "T-30")
    t10_odds = odds_eng.load_odds_snapshot(years, "T-10")
    
    return odds_eng.add_features(df, t30_odds, t10_odds)


def build_base_pipeline() -> FeaturePipeline:
    """Base data (2014-2024) のパイプラインを構築"""
    pipeline = FeaturePipeline(CACHE_DIR, "jra_base")
    
    pipeline.add_step("raw", step_load_raw, [], 
                      version="1.1",  # v1.1: first_3f, grade_code, joken_code追加
                      params={"start_date": "2014-01-01", "end_date": "2025-12-14"})
    
    pipeline.add_step("cleanse", step_cleanse, ["raw"], version="1.0")
    
    pipeline.add_step("basic", step_basic_features, ["cleanse"], version="1.2")  # v1.2: Class Level判定強化
    
    pipeline.add_step("inject_odds", step_inject_odds, ["basic"], 
                      version="1.0",
                      params={"years": range(2014, 2026)})
    
    pipeline.add_step("history", step_history_aggregation, ["inject_odds"], version="1.0")
    
    pipeline.add_step("category", step_category_aggregation, ["history"], version="1.0")
    
    # Phase 18 Features
    pipeline.add_step("bloodline", step_bloodline_features, ["category"], version="1.0")
    pipeline.add_step("profile", step_profile_features, ["bloodline"], version="1.0")
    
    pipeline.add_step("pace", step_pace_features, ["profile"], version="1.0")
    
    pipeline.add_step("advanced", step_advanced_features, ["pace"], version="1.2")  # v1.2: Nige Intensity, Early Speed Std, First 3F Index
    
    pipeline.add_step("timeseries_odds", step_timeseries_odds, ["advanced"], 
                      version="1.0",
                      params={"years": range(2014, 2026)})
    
    return pipeline


def get_features(df):
    """Extract features for model training"""
    drop_cols = [
        'race_id', 'date', 'horse_id', 'horse_name', 'title',
        'rank', 'target', 'rank_str', 'year',
        'time', 'raw_time', 'passing_rank', 'last_3f',
        'weight', 'weight_diff_val', 'weight_diff_sign',
        'winning_numbers', 'payout', 'ticket_type',
        'pass_1', 'pass_2', 'pass_3', 'pass_4',
        'horse_number', 'frame_number',
        
        # Leak-prone
        'slow_start_recovery', 'pace_disadvantage', 'wide_run',
        'track_bias_disadvantage', 'outer_frame_disadv',
        'odds_race_rank', 'popularity_race_rank',
        'odds_deviation', 'popularity_deviation',
        'trend_win_inner_rate', 'trend_win_mid_rate', 'trend_win_outer_rate',
        'trend_win_front_rate', 'trend_win_fav_rate',
        
        # Speed Index temps
        'time_index', 'last_3f_index',
        
        # Odds temps
        'ninki_t_30', 'ninki_t_10',
        
        # Time temps
        # Time temps
        'time_diff',
        
        # [v16.1] Leakage columns (Result data)
        'honshokin', 'fukashokin', 'abnormal_code',
    ] + DROP_FEATURES
    
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


def run_backtest_with_pipeline(force_from: str = None, feature_mode: str = 'full'):
    """パイプラインを使用したバックテスト実行
    feature_mode: 'full', 'odds_only', 'no_odds', 'residual'
    """
    logger.info(f"Starting Backtest with Feature Mode: {feature_mode}")
    
    # 1. Build and run base pipeline
    logger.info("=== Phase 1: Base Data Processing with Pipeline ===")
    pipeline = build_base_pipeline()
    
    # Print cache status
    pipeline.print_status()
    
    # Run pipeline
    master_df = pipeline.run(force_from=force_from)
    logger.info(f"Base data loaded: {len(master_df):,} rows")
    gc.collect()
    
    # Initialize incremental aggregator from base data
    inc_cat_agg = IncrementalCategoryAggregator()
    inc_cat_agg.fit(master_df)
    gc.collect()
    
    # 2. Incremental Phase (2025)
    logger.info("=== Phase 2: Incremental Processing (2025) ===")
    
    loader = JraVanDataLoader()
    cleanser = DataCleanser()
    engineer = FeatureEngineer()
    hist_agg = HistoryAggregator()
    adv_eng = AdvancedFeatureEngineer()
    exp_eng = ExperienceFeatureEngineer()
    rel_eng = RelativeFeatureEngineer()
    opp_eng = OppositionFeatureEngineer()
    odds_eng = OddsFeatureEngineer(SNAPSHOT_DIR, odds_drop_threshold=0.20)
    
    # P2: Elo Rating
    from src.preprocessing.rating_features import RatingFeatureEngineer
    rating_eng = RatingFeatureEngineer()
    
    months = pd.date_range("2025-01-01", "2025-12-31", freq="MS").strftime("%Y-%m").tolist()
    results = []
    
    # Load 2025 odds
    t10_2025_list = []
    path = os.path.join(SNAPSHOT_DIR, "2025", "odds_T-10.parquet")
    if os.path.exists(path):
        t10 = pd.read_parquet(path, filters=[('ticket_type', '=', 'win')])
        t10 = t10[['race_id', 'combination', 'odds', 'ninki']].copy()
        t10['race_id'] = t10['race_id'].astype(str)
        t10['horse_number'] = pd.to_numeric(t10['combination'], errors='coerce')
        t10.rename(columns={'ninki': 'popularity'}, inplace=True)
        t10_2025_list.append(t10[['race_id', 'horse_number', 'odds', 'popularity']])
    t10_2025_all = pd.concat(t10_2025_list, ignore_index=True) if t10_2025_list else pd.DataFrame()
    
    t30_2025 = odds_eng.load_odds_snapshot([2025], "T-30")
    t10_2025_features = odds_eng.load_odds_snapshot([2025], "T-10")
    
    for month_str in months:
        logger.info(f"\n=== Processing Month: {month_str} ===")
        
        pred_cache_path = os.path.join(PRED_DIR, f"preds_{month_str}_pipeline.parquet")
        test_df = None
        
        if False and os.path.exists(pred_cache_path):
            logger.info("Found prediction cache. Loading...")
            test_df = pd.read_parquet(pred_cache_path)
        else:
            # Load Month Raw
            start_date = f"{month_str}-01"
            y, m = map(int, month_str.split('-'))
            last_day = calendar.monthrange(y, m)[1]
            end_date = f"{month_str}-{last_day}"
            
            m_df = loader.load(history_start_date=start_date, end_date=end_date, jra_only=True)
            if len(m_df) == 0:
                continue
            
            # Process
            m_df = cleanser.cleanse(m_df)
            m_df = engineer.add_features(m_df)
            
            # Add track_condition_code
            if 'state_num' in m_df.columns and 'track_condition_code' not in m_df.columns:
                m_df['track_condition_code'] = m_df['state_num']
            
            # Inject T-10 odds
            if not t10_2025_all.empty:
                m_df['race_id'] = m_df['race_id'].astype(str)
                m_df['horse_number'] = pd.to_numeric(m_df['horse_number'], errors='coerce')
                m_df = pd.merge(m_df, t10_2025_all, on=['race_id', 'horse_number'], 
                              suffixes=('', '_t10'), how='left')
                # [v16.1 Fix] 確定オッズへのフォールバックを廃止
                m_df['odds'] = m_df['odds_t10']
                m_df['popularity'] = m_df['popularity_t10']
                m_df.drop(columns=['odds_t10', 'popularity_t10'], inplace=True, errors='ignore')
            else:
                # T-10がない場合、確定オッズをクリア
                if 'odds' in m_df.columns:
                    m_df['odds'] = np.nan
                if 'popularity' in m_df.columns:
                    m_df['popularity'] = np.nan
            
            m_df_with_cat = inc_cat_agg.transform_update(m_df)
            
            # Context Slicing
            ctx_date = (pd.to_datetime(start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
            context_mask = (master_df['date'] >= ctx_date)
            context_df = master_df[context_mask].copy()
            
            # Combine
            proc_df = pd.concat([context_df, m_df_with_cat], ignore_index=True)
            proc_df = proc_df.sort_values(['date', 'race_id'])
            
            # Engineer
            proc_df = hist_agg.aggregate(proc_df)
            proc_df = adv_eng.add_features(proc_df)
            proc_df = exp_eng.add_features(proc_df)
            proc_df = rel_eng.add_features(proc_df)
            proc_df = opp_eng.add_features(proc_df)
            proc_df = rating_eng.add_features(proc_df)  # P2: Elo Rating
            proc_df = odds_eng.add_features(proc_df, t30_2025, t10_2025_features)
            
            # Extract Test Data
            test_mask = (proc_df['date'] >= start_date) & (proc_df['date'] <= end_date)
            test_df = proc_df[test_mask].copy()
            
            # Append to master
            if not test_df.empty:
                master_df = pd.concat([master_df, test_df], ignore_index=True)
                master_df = master_df.sort_values(['date', 'race_id'])
            
            # Train
            train_mask = (master_df['date'] < start_date)
            full_train_df = master_df[train_mask].copy()
            
            # Calibration Split
            full_train_df['target'] = full_train_df['rank'].apply(
                lambda r: 3 if r == 1 else (2 if r == 2 else (1 if r == 3 else 0))
            )
            full_train_df = full_train_df.sort_values(['date', 'race_id'])
            
            u_races = full_train_df['race_id'].unique()
            split_idx = int(len(u_races) * 0.8)
            train_races = u_races[:split_idx]
            calib_races = u_races[split_idx:]
            
            train_df = full_train_df[full_train_df['race_id'].isin(train_races)].copy()
            calib_df = full_train_df[full_train_df['race_id'].isin(calib_races)].copy()
            
            # Train
            train_df = train_df.sort_values('race_id')
            if feature_mode.startswith('residual'):
                # --- Residual Family Modes ---
                
                # Check Leakage first for safety (on sample)
                X_sample, _ = get_features(train_df.head(100))
                # Only check no_odds part for strictness? Both check.
                
                # 1. Market Model (Odds Only)
                X_train_m, _ = get_features(train_df)
                X_train_m = filter_features(X_train_m, 'odds_only')
                check_leakage(X_train_m.columns)
                
                # 2. Residual Model (No Odds)
                X_train_r, _ = get_features(train_df)
                X_train_r = filter_features(X_train_r, 'no_odds')
                check_leakage(X_train_r.columns)
                
                # Prepare Test Features
                X_test_m = filter_features(get_features(test_df)[0], 'odds_only')
                X_test_r = filter_features(get_features(test_df)[0], 'no_odds')
                
                # Prepare Calib Features
                X_calib_m = filter_features(get_features(calib_df)[0], 'odds_only')
                X_calib_r = filter_features(get_features(calib_df)[0], 'no_odds')
                
                if feature_mode == 'residual_direct':
                    # --- B. Direct Market Error Learning ---
                    # 1. OOF on Train for Market Model (Binary Objective Required)
                    logger.info("Generating Market OOF (Binary)...")
                    params_binary = PARAMS.copy()
                    params_binary['objective'] = 'binary'
                    params_binary['metric'] = 'binary_logloss'
                    
                    oof_market = get_oof_predictions(params_binary, X_train_m, train_df['target'], cv=4)
                    
                    # 2. Residual Target
                    # Residual = Target - MarketProb
                    residuals = train_df['target'] - oof_market
                    
                    # 3. Train Residual Model (Regression)
                    params_reg = PARAMS.copy()
                    params_reg['objective'] = 'regression'
                    params_reg['metric'] = 'rmse'
                    
                    model_r_direct = lgb.train(
                        params_reg, 
                        lgb.Dataset(X_train_r, label=residuals, group=train_df.groupby('race_id').size().values),
                        num_boost_round=100
                    )
                    
                    # 4. Train Final Market Model (Binary)
                    model_m = lgb.train(
                        params_binary, 
                        lgb.Dataset(X_train_m, label=train_df['target'], group=train_df.groupby('race_id').size().values),
                        num_boost_round=100
                    )
                    
                    # Inference
                    p_m_test = model_m.predict(X_test_m)
                    p_r_test = model_r_direct.predict(X_test_r)
                    y_test_raw = np.clip(p_m_test + p_r_test, 0.0, 1.0)
                    
                    # For Calib (Using predictions on Calib set)
                    p_m_c = model_m.predict(X_calib_m)
                    p_r_c = model_r_direct.predict(X_calib_r)
                    y_calib = np.clip(p_m_c + p_r_c, 0.0, 1.0)
                    
                else:
                    # --- A. Weighted Ensemble (residual, residual_opt) ---
                    # Train Models on train_df
                    model_m = lgb.train(PARAMS, lgb.Dataset(X_train_m, label=train_df['target'], group=train_df.groupby('race_id').size().values), num_boost_round=80)
                    model_r = lgb.train(PARAMS, lgb.Dataset(X_train_r, label=train_df['target'], group=train_df.groupby('race_id').size().values), num_boost_round=80)
                    
                    # Predict on Calib (Validation)
                    p_m_c = model_m.predict(X_calib_m)
                    p_r_c = model_r.predict(X_calib_r)
                    
                    best_w = 0.5
                    if feature_mode == 'residual_opt':
                        # Optimize w on Calib set
                        best_w = optimize_residual_weight(
                            calib_df['target'].values,
                            p_m_c, 
                            p_r_c,
                            calib_df['odds'].fillna(0).values # Need odds
                        )
                        logger.info(f"Month {month_str}: Optimal w={best_w:.2f}")
                        results.append({'month': month_str, 'best_w': best_w, 'type': 'meta'})
                        
                    # Ensemble Calib (for Logistic Calibration later)
                    y_calib = best_w * p_m_c + (1 - best_w) * p_r_c
                    
                    # Ensemble Test
                    p_m_t = model_m.predict(X_test_m)
                    p_r_t = model_r.predict(X_test_r)
                    y_test_raw = best_w * p_m_t + (1 - best_w) * p_r_t
                
            else:
                X_train, cols = get_features(train_df)
                X_train = filter_features(X_train, feature_mode)
                check_leakage(X_train.columns)
                
                model = lgb.train(
                    PARAMS, 
                    lgb.Dataset(X_train, train_df['target'], group=train_df.groupby('race_id').size().values), 
                    num_boost_round=100
                )
                
                # Calib Params
                X_calib, _ = get_features(calib_df)
                X_calib = filter_features(X_calib, feature_mode)
                y_calib = model.predict(X_calib)
                
                # Test Params
                X_test, _ = get_features(test_df)
                X_test = filter_features(X_test, feature_mode)
                y_test_raw = model.predict(X_test)
                
                # Feature Importance (Standard mode only)
                if month_str == "2025-01" and feature_mode == 'no_odds':
                     # Save FI for checks
                     pass

            # Common Calibration - Use IsotonicRegression for better probability calibration
            # LambdaRank scores are ordinal, not probabilities
            calib_target = (calib_df['rank'] == 1).astype(int)
            
            # IsotonicRegression: monotonic transformation, better for ranking scores
            # y_min=0, y_max=1 to produce proper probabilities
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso.fit(y_calib, calib_target)
            calib_probs_raw = iso.predict(y_test_raw)
            
            # Race-level normalization (NOT softmax, just divide by sum)
            # This ensures probabilities sum to 1.0 per race
            test_df['pred_prob'] = y_test_raw
            test_df['calib_prob_raw'] = calib_probs_raw
            
            # Simple normalization: each horse's prob / sum of probs in race
            test_df['calib_prob'] = test_df.groupby('race_id')['calib_prob_raw'].transform(
                lambda x: x / x.sum() if x.sum() > 0 else x
            )
            
            # Drop intermediate column
            test_df.drop(columns=['calib_prob_raw'], inplace=True)
            
            test_df['odds'] = test_df['odds'].fillna(0)

            
            # test_df.to_parquet(pred_cache_path) # Disabled for experiments
            
            del train_df, calib_df, full_train_df
            gc.collect()
        
            gc.collect()
        
        # Simulate betting
        monthly_invest = 0
        monthly_return = 0
        monthly_bets = 0
        
        # Calculate EV for stats (minimal logic)
        # Note: 'ev' calculation might differ in analysis script, careful.
        # Here we just use a simple proxy for logging.
        if 'calib_prob' in test_df.columns:
            # Filter out odds=0 (invalid data)
            valid_df = test_df[test_df['odds'] > 0].copy()
            ev_proxy = valid_df['calib_prob'] * valid_df['odds']
            bet_df = valid_df[ev_proxy > 1.0]
            monthly_bets = len(bet_df)
            hits = bet_df[bet_df['rank'] == 1]
            monthly_return = (hits['odds'] * 100).sum()
            monthly_invest = monthly_bets * 100
        
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
        
        # Save Monthly Chunk (Safety)
        month_dir = os.path.join(REPORT_DIR, "monthly")
        os.makedirs(month_dir, exist_ok=True)
        
        # Ensure we save necessary columns for analysis
        # race_id, horse_number, pred, odds, rank, payout(dropped), calib_prob, pred_prob
        save_cols = [c for c in test_df.columns if c in [
            'race_id', 'horse_number', 'pred', 'odds', 'rank', 'pred_prob', 'calib_prob', 'ev'
        ]]
        chunk_path = os.path.join(month_dir, f"results_{feature_mode}_{month_str}.parquet")
        test_df[save_cols].to_parquet(chunk_path)
        logger.info(f"Saved chunk to {chunk_path}")

    # Summary
    res_df = pd.DataFrame(results)
    total_invest = res_df['invest'].sum()
    total_return = res_df['return'].sum()
    total_roi = (total_return / total_invest * 100) if total_invest > 0 else 0
    
    logger.info(f"=== Final Results ({feature_mode}) ===")
    logger.info(f"Total ROI: {total_roi:.2f}% (Inv: {total_invest}, Ret: {total_return})")
    
    # ================================================================
    # Accuracy Metrics: 予測順位別の精度
    # ================================================================
    import glob
    month_dir = os.path.join(REPORT_DIR, "monthly")
    parquet_files = sorted(glob.glob(os.path.join(month_dir, f"results_{feature_mode}_2025-*.parquet")))
    
    if parquet_files:
        all_preds = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        
        if 'calib_prob' in all_preds.columns and 'race_id' in all_preds.columns and 'rank' in all_preds.columns:
            # 予測順位を付与
            all_preds['pred_rank'] = all_preds.groupby('race_id')['calib_prob'].rank(ascending=False, method='first').astype(int)
            
            logger.info("=== Prediction Accuracy by Rank ===")
            logger.info("予測順位 | 勝率  | 連対率 | 複勝率 | N")
            logger.info("-" * 45)
            
            for pred_rank in range(1, 6):
                subset = all_preds[all_preds['pred_rank'] == pred_rank]
                n = len(subset)
                if n == 0:
                    continue
                win = (subset['rank'] == 1).mean()
                place = (subset['rank'] <= 2).mean()
                show = (subset['rank'] <= 3).mean()
                logger.info(f"  {pred_rank}位予測 | {win:5.1%} | {place:5.1%} | {show:5.1%} | {n}")
            
            # 平均頭数とランダム比較
            avg_horses = all_preds.groupby('race_id').size().mean()
            top1 = all_preds[all_preds['pred_rank'] == 1]
            top1_win = (top1['rank'] == 1).mean()
            random_win = 1 / avg_horses
            
            logger.info(f"=== Summary ===")
            logger.info(f"平均頭数: {avg_horses:.1f}, ランダム勝率: {random_win:.1%}")
            logger.info(f"Top1予測 勝率: {top1_win:.1%} (ランダム比: {top1_win/random_win:.2f}x)")
    
    # Save results
    all_res = pd.DataFrame(results) if results else pd.DataFrame()
    out_path = f"{REPORT_DIR}/results_{feature_mode}.parquet"
    if not all_res.empty:
        all_res.to_parquet(out_path)
    
    logger.info(f"Results saved to {out_path}")
    return all_res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-from', type=str, default=None)
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'odds_only', 'no_odds', 'residual', 'residual_opt', 'residual_direct'])
    args = parser.parse_args()
    
    run_backtest_with_pipeline(force_from=args.force_from, feature_mode=args.mode)

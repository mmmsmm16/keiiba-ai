"""
JRA Jan 2025 Pilot Backtest (Leak-Free Verification)

Methodology:
1. Load raw data up to 2025-01-31 ONLY.
   - Ensures NO future data (Feb-Dec) influences preprocessing.
2. Run full preprocessing pipeline.
   - Since we only have data up to Jan 31, all expanding stats are valid.
3. Split:
   - Train: < 2025-01-01
   - Test: 2025-01-xx
4. Train & Predict.
5. Simulate betting using T-10 odds.
"""
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import logging
from scipy.special import softmax

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
ODDS_PATH = "data/odds_snapshots/2025/odds_T-10.parquet"
REPORT_DIR = "reports/jra/wf_pilot"
os.makedirs(REPORT_DIR, exist_ok=True)

# Training params
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

def load_and_preprocess(end_date_str='2025-01-31'):
    from src.preprocessing.loader import JraVanDataLoader
    from src.preprocessing.cleansing import DataCleanser
    from src.preprocessing.feature_engineering import FeatureEngineer
    from src.preprocessing.aggregators import HistoryAggregator
    from src.preprocessing.category_aggregators import CategoryAggregator
    from src.preprocessing.advanced_features import AdvancedFeatureEngineer
    from src.preprocessing.experience_features import ExperienceFeatureEngineer
    from src.preprocessing.relative_features import RelativeFeatureEngineer
    from src.preprocessing.opposition_features import OppositionFeatureEngineer
    from src.preprocessing.race_level_features import RaceLevelFeatureEngineer
    
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    # 1. Load Data
    logger.info(f"Loading data up to {end_date_str}...")
    loader = JraVanDataLoader()
    df = loader.load(jra_only=True)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= end_date].copy()
    logger.info(f"Filtered data: {len(df)} rows (Max date: {df['date'].max()})")
    
    # 2. Preprocessing
    logger.info("Running preprocessing pipeline...")
    
    cleanser = DataCleanser()
    df = cleanser.cleanse(df)
    
    engineer = FeatureEngineer()
    df = engineer.add_features(df)
    
    aggregator = HistoryAggregator()
    df = aggregator.aggregate(df)
    
    cat_aggregator = CategoryAggregator()
    df = cat_aggregator.aggregate(df)
    
    adv_engineer = AdvancedFeatureEngineer()
    df = adv_engineer.add_features(df)
    
    exp_engineer = ExperienceFeatureEngineer()
    df = exp_engineer.add_features(df)
    
    rel_engineer = RelativeFeatureEngineer()
    df = rel_engineer.add_features(df)
    
    opp_engineer = OppositionFeatureEngineer()
    df = opp_engineer.add_features(df)
    
    return df

def get_features(df):
    """Extract features removing ALL odds-related and leak-prone columns"""
    drop_cols = [
        'race_id', 'date', 'horse_id', 'horse_name', 'title',
        'jockey_id', 'trainer_id', 'sire_id', 'mare_id',
        'rank', 'target', 'rank_str', 'year',
        'time', 'raw_time', 'passing_rank', 'last_3f',
        'odds', 'popularity', 'weight', 'weight_diff_val', 'weight_diff_sign',
        'winning_numbers', 'payout', 'ticket_type',
        'pass_1', 'pass_2', 'pass_3', 'pass_4',
        
        # Potentially leaky / future info
        'slow_start_recovery', 'pace_disadvantage', 'wide_run',
        'track_bias_disadvantage', 'outer_frame_disadv',
        'odds_race_rank', 'popularity_race_rank',
        'odds_deviation', 'popularity_deviation',
        'trend_win_inner_rate', 'trend_win_mid_rate', 'trend_win_outer_rate',
        'trend_win_front_rate', 'trend_win_fav_rate',
        'lag1_odds', 'lag1_popularity',
        'time_index', 'last_3f_index',
        
        # Explicit ban list for odds from production model
        'dlog_odds_t30_t10', 'dlog_odds_t60_t10', 'log_odds_t10',
        'odds_drop_rate_t60_t10', 'odds_volatility', 'rank_change_t60_t10'
    ]
    
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(exclude=['object', 'datetime64'])
    
    # Force float64
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].cat.codes.astype('float64')
        elif X[col].dtype != 'float64':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype('float64')
            
    # Debug dropped features
    all_cols = set(df.columns)
    kept_cols = set(X.columns)
    dropped_cols = all_cols - kept_cols
    # print(f"Dropped columns ({len(dropped_cols)}): {sorted(list(dropped_cols))}")
    
    return X, X.columns.tolist(), list(dropped_cols)

def run_pilot():
    # 1. Preprocess
    df = load_and_preprocess('2025-01-31')
    
    # 2. Split
    train_cutoff = datetime(2025, 1, 1)
    train_df = df[df['date'] < train_cutoff].copy()
    test_df = df[df['date'] >= train_cutoff].copy()
    
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    # 3. Create Target
    train_df['rank'] = pd.to_numeric(train_df['rank'], errors='coerce')
    train_df['target'] = train_df['rank'].apply(lambda r: 3 if r == 1 else (2 if r == 2 else (1 if r == 3 else 0)))
    
    train_df = train_df.sort_values('race_id')
    
    # 4. Train
    X_train, feature_names, dropped = get_features(train_df)
    logger.info(f"Dropped features sample: {dropped[:20]}")
    
    y_train = train_df['target'].values
    q_train = train_df.groupby('race_id').size().values
    
    logger.info(f"Training with {len(feature_names)} features...")
    
    lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
    
    model = lgb.train(
        PARAMS,
        lgb_train,
        num_boost_round=100
    )
    
    # Check feature importance
    imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    logger.info(f"Top 10 features:\n{imp.head(10)}")
    
    # 5. Predict
    X_test, _, _ = get_features(test_df)
    # Align features
    for c in feature_names:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[feature_names]
    
    test_df['pred_prob'] = model.predict(X_test)
    
    # 6. Simulate with Odds
    logger.info("Loading T-10 Odds...")
    if not os.path.exists(ODDS_PATH):
        logger.error(f"Odds file not found: {ODDS_PATH}")
        return
        
    odds_df = pd.read_parquet(ODDS_PATH)
    odds_df['race_id'] = odds_df['race_id'].astype(str)
    
    # Filter for Jan 2025
    races = test_df['race_id'].unique()
    odds_jan = odds_df[odds_df['race_id'].isin(races)].copy()
    
    invest = 0
    ret = 0
    bets = 0
    
    logger.info("Simulating bets...")
    for rid in races:
        # Drop 'odds' from race_df before merge to avoid collision
        race_df = test_df[test_df['race_id'] == rid].drop(columns=['odds'], errors='ignore').copy()
        
        race_odds = odds_jan[(odds_jan['race_id'] == rid) & (odds_jan['ticket_type'] == 'win')].copy()
        
        if race_odds.empty or len(race_df) < 5:
            continue
            
        # Parse combination to get horse number
        race_odds['horse_number'] = pd.to_numeric(race_odds['combination'], errors='coerce')
        
        # Merge odds to predictions
        race_df['horse_number'] = pd.to_numeric(race_df['horse_number'], errors='coerce')
        
        merged = pd.merge(race_df, race_odds[['horse_number', 'odds']], on='horse_number', how='inner')
        
        if merged.empty:
            continue
            
        # Calc EV
        merged['softmax_prob'] = softmax(merged['pred_prob'].values)
        merged['ev'] = merged['softmax_prob'] * merged['odds']
        
        # Bet EV > 1.0 (Fixed Stake)
        bet_candidates = merged[merged['ev'] > 1.0]
        
        for _, row in bet_candidates.iterrows():
            invest += 100
            bets += 1
            if row['rank'] == 1:
                ret += 100 * row['odds']
    
    profit = ret - invest
    roi = (ret / invest * 100) if invest > 0 else 0
    
    logger.info(f"\nJan 2025 Results (Leak-Free):")
    logger.info(f"Bets: {bets}")
    logger.info(f"Invest: {invest:,.0f}")
    logger.info(f"Return: {ret:,.0f}")
    logger.info(f"Profit: {profit:+,.0f}")
    logger.info(f"ROI: {roi:.2f}%")
    
    # Save Report
    with open(os.path.join(REPORT_DIR, 'jan2025_pilot.md'), 'w') as f:
        f.write("# Jan 2025 Pilot Backtest (Leak-Free Verification)\n\n")
        f.write(f"ROI: **{roi:.2f}%**\n")
        f.write(f"Profit: {profit:,.0f} JPY\n")
        f.write(f"Bets: {bets}\n")

if __name__ == "__main__":
    run_pilot()

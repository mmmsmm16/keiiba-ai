"""
JRA Walk-Forward Backtest

Proper backtest methodology:
- For each month in 2025, train model on data UP TO the end of previous month
- Use that model to predict all races in the current month
- This avoids data leakage (future data in training)

Monthly retraining schedule:
- January 2025: Train on 2014-01-01 ~ 2024-12-31
- February 2025: Train on 2014-01-01 ~ 2025-01-31
- ...and so on
"""
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
import joblib
from sklearn.isotonic import IsotonicRegression
from scipy.special import softmax

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
ODDS_PATH = "data/odds_snapshots/2025/odds_T-10.parquet"
REPORT_DIR = "reports/jra/wf"

os.makedirs(REPORT_DIR, exist_ok=True)

# Training params (same as production models)
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

def load_data():
    """Load preprocessed JRA data"""
    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['target'] = df['rank'].apply(lambda r: 3 if r == 1 else (2 if r == 2 else (1 if r == 3 else 0)))
    return df

def load_odds():
    """Load T-10 odds for 2025"""
    if not os.path.exists(ODDS_PATH):
        logger.warning(f"Odds file not found: {ODDS_PATH}")
        return None
    odds_df = pd.read_parquet(ODDS_PATH)
    odds_df['race_id'] = odds_df['race_id'].astype(str)
    return odds_df

def get_features(df):
    """Extract features (same as dataset.py)"""
    drop_cols = [
        'race_id', 'date', 'horse_id', 'horse_name', 'title',
        'jockey_id', 'trainer_id', 'sire_id', 'mare_id',
        'rank', 'target', 'rank_str', 'year',
        'time', 'raw_time', 'passing_rank', 'last_3f',
        'odds', 'popularity', 'weight', 'weight_diff_val', 'weight_diff_sign',
        'winning_numbers', 'payout', 'ticket_type',
        'pass_1', 'pass_2', 'pass_3', 'pass_4',
        # Leakage features
        'slow_start_recovery', 'pace_disadvantage', 'wide_run',
        'track_bias_disadvantage', 'outer_frame_disadv',
        'odds_race_rank', 'popularity_race_rank',
        'odds_deviation', 'popularity_deviation',
        'trend_win_inner_rate', 'trend_win_mid_rate', 'trend_win_outer_rate',
        'trend_win_front_rate', 'trend_win_fav_rate',
        'lag1_odds', 'lag1_popularity',
        'time_index', 'last_3f_index'
    ]
    
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(exclude=['object', 'datetime64'])
    
    # Force float64 for consistency
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].cat.codes.astype('float64')
        elif X[col].dtype != 'float64':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype('float64')
    
    return X

def train_model_for_period(df, cutoff_date):
    """Train model using only data before cutoff_date"""
    train_df = df[df['date'] < cutoff_date].copy()
    
    if len(train_df) < 10000:
        logger.warning(f"Not enough training data before {cutoff_date}: {len(train_df)} rows")
        return None, None, None
    
    logger.info(f"Training model with data before {cutoff_date.strftime('%Y-%m-%d')} ({len(train_df)} rows)")
    
    train_df = train_df.sort_values('race_id')
    
    X = get_features(train_df)
    y = train_df['target'].values
    q = train_df.groupby('race_id').size().values
    
    lgb_train = lgb.Dataset(X, y, group=q)
    
    model = lgb.train(
        PARAMS,
        lgb_train,
        num_boost_round=100,  # Fewer iterations for speed
    )
    
    # Simple calibration using recent validation data (last 20%)
    val_start_idx = int(len(train_df) * 0.8)
    val_df = train_df.iloc[val_start_idx:].copy()
    
    X_val = get_features(val_df)
    for c in X.columns:
        if c not in X_val.columns:
            X_val[c] = 0
    X_val = X_val[X.columns]
    
    preds = model.predict(X_val)
    val_df['raw_score'] = preds
    val_df['prob'] = val_df.groupby('race_id')['raw_score'].transform(lambda x: softmax(x))
    
    y_true = (val_df['rank'] == 1).astype(int).values
    X_prob = val_df['prob'].values
    
    calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    calibrator.fit(X_prob, y_true)
    
    return model, calibrator, X.columns.tolist()

def run_walk_forward_backtest(start_date='2025-01-01', end_date='2025-12-31'):
    """Run walk-forward backtest with monthly retraining"""
    logger.info("=" * 70)
    logger.info("JRA Walk-Forward Backtest")
    logger.info("=" * 70)
    
    # Load all data
    df = load_data()
    odds_df = load_odds()
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Limit to available data
    max_date = df['date'].max()
    if end > max_date:
        end = max_date
        logger.info(f"Limiting end date to available data: {end.strftime('%Y-%m-%d')}")
    
    current_month_start = start.replace(day=1)
    
    all_results = []
    
    while current_month_start <= end:
        month_end = (current_month_start + relativedelta(months=1)) - timedelta(days=1)
        if month_end > end:
            month_end = end
        
        train_cutoff = current_month_start
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Period: {current_month_start.strftime('%Y-%m')} (Train cutoff: {train_cutoff.strftime('%Y-%m-%d')})")
        logger.info(f"{'='*60}")
        
        # Train model for this period
        model, calibrator, feature_names = train_model_for_period(df, train_cutoff)
        
        if model is None:
            logger.warning(f"Skipping {current_month_start.strftime('%Y-%m')} due to insufficient training data")
            current_month_start = current_month_start + relativedelta(months=1)
            continue
        
        # Get races for this month
        month_df = df[(df['date'] >= current_month_start) & (df['date'] <= month_end)].copy()
        
        if month_df.empty:
            logger.info(f"No races in {current_month_start.strftime('%Y-%m')}")
            current_month_start = current_month_start + relativedelta(months=1)
            continue
        
        # Predict
        X = get_features(month_df)
        for c in feature_names:
            if c not in X.columns:
                X[c] = 0
        X = X[feature_names]
        
        preds = model.predict(X)
        month_df['pred_prob'] = preds
        
        # Betting simulation
        month_invest = 0
        month_return = 0
        month_bets = 0
        
        for rid in month_df['race_id'].unique():
            race_df = month_df[month_df['race_id'] == rid].copy()
            race_df['softmax_prob'] = softmax(race_df['pred_prob'].values)
            race_df['calibrated_prob'] = calibrator.predict(race_df['softmax_prob'].values)
            
            # Get odds from dataset
            race_df['odds'] = pd.to_numeric(race_df['odds'], errors='coerce').fillna(100)
            race_df['ev'] = race_df['calibrated_prob'] * race_df['odds']
            
            # Bet on horses with EV > 1.0
            bet_df = race_df[race_df['ev'] > 1.0].copy()
            
            for _, row in bet_df.iterrows():
                bet_amount = 100  # Fixed 100 yen per bet
                month_invest += bet_amount
                month_bets += 1
                
                # Check if won
                if row['rank'] == 1:
                    month_return += bet_amount * row['odds']
        
        # Calculate monthly ROI
        if month_invest > 0:
            month_roi = month_return / month_invest * 100
            month_profit = month_return - month_invest
        else:
            month_roi = 0
            month_profit = 0
        
        logger.info(f"Month {current_month_start.strftime('%Y-%m')}: "
                   f"Bets={month_bets}, Invest={month_invest:,.0f}, "
                   f"Return={month_return:,.0f}, Profit={month_profit:+,.0f}, ROI={month_roi:.1f}%")
        
        all_results.append({
            'month': current_month_start.strftime('%Y-%m'),
            'bets': month_bets,
            'invest': month_invest,
            'return': month_return,
            'profit': month_profit,
            'roi': month_roi
        })
        
        current_month_start = current_month_start + relativedelta(months=1)
    
    # Summary
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        total_invest = results_df['invest'].sum()
        total_return = results_df['return'].sum()
        total_profit = total_return - total_invest
        total_roi = (total_return / total_invest * 100) if total_invest > 0 else 0
        
        logger.info("\n" + "=" * 70)
        logger.info("WALK-FORWARD BACKTEST RESULTS (NO DATA LEAKAGE)")
        logger.info("=" * 70)
        logger.info(f"Period: {start_date} ~ {end.strftime('%Y-%m-%d')}")
        logger.info(f"Total Bets: {results_df['bets'].sum():,}")
        logger.info(f"Total Investment: {total_invest:,.0f} JPY")
        logger.info(f"Total Return: {total_return:,.0f} JPY")
        logger.info(f"Total Profit: {total_profit:+,.0f} JPY")
        logger.info(f"Overall ROI: {total_roi:.2f}%")
        
        # Save report
        report_path = os.path.join(REPORT_DIR, 'walkforward_2025_summary.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# JRA 2025 ウォークフォワードバックテスト結果\n\n")
            f.write("## 概要\n")
            f.write("- **方式**: ウォークフォワード（月次再学習）\n")
            f.write("- **データリーク**: なし（各月の予測は前月末までのデータで学習）\n\n")
            
            f.write("## 結果サマリー\n\n")
            f.write("| 項目 | 値 |\n|---|---|\n")
            f.write(f"| 期間 | {start_date} ~ {end.strftime('%Y-%m-%d')} |\n")
            f.write(f"| 総賭け数 | {results_df['bets'].sum():,} |\n")
            f.write(f"| 総投資額 | {total_invest:,.0f} 円 |\n")
            f.write(f"| 総回収額 | {total_return:,.0f} 円 |\n")
            f.write(f"| 純利益 | {total_profit:+,.0f} 円 |\n")
            f.write(f"| **総合ROI** | **{total_roi:.2f}%** |\n\n")
            
            f.write("## 月別推移\n\n")
            f.write("| 月 | 賭け数 | 投資額 | 回収額 | 利益 | ROI |\n")
            f.write("|---|---|---|---|---|---|\n")
            for _, row in results_df.iterrows():
                f.write(f"| {row['month']} | {row['bets']} | {row['invest']:,.0f} | "
                       f"{row['return']:,.0f} | {row['profit']:+,.0f} | {row['roi']:.1f}% |\n")
            
            f.write(f"\n---\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
        
        logger.info(f"\nReport saved to: {report_path}")
    
    return results_df

if __name__ == "__main__":
    run_walk_forward_backtest('2025-01-01', '2025-12-31')

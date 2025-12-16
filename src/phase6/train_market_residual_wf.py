"""
Phase 6: Train Market-Residual Model (v13)
市場残差モデル: logit(p_win) = logit(p_market) + Δ(features)

Usage (in container):
    docker compose exec app python src/phase6/train_market_residual_wf.py
    docker compose exec app python src/phase6/train_market_residual_wf.py --wf_years 2021 2022 2023
"""

import sys
import os
import argparse
import logging
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from typing import Dict, List, Tuple
from scipy.special import expit, logit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.race_filter import filter_jra_only
from utils.calibration import normalize_prob_per_race

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# LightGBM default params (similar to existing v12)
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'verbose': -1,
    'seed': 42,
}


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safe logit transform"""
    p = np.clip(p, eps, 1 - eps)
    return logit(p)


def softmax_per_race(logits: np.ndarray, race_ids: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax per race (numerically stable)"""
    import pandas as pd
    
    df = pd.DataFrame({'race_id': race_ids, 'logit': logits / temperature})
    
    # Subtract max for numerical stability
    df['logit_shifted'] = df.groupby('race_id')['logit'].transform(lambda x: x - x.max())
    df['exp'] = np.exp(df['logit_shifted'])
    df['softmax'] = df.groupby('race_id')['exp'].transform(lambda x: x / x.sum())
    
    return df['softmax'].values


def compute_race_nll(df: pd.DataFrame, prob_col: str, race_col: str = 'race_id', rank_col: str = 'rank') -> float:
    """Compute Race-wise NLL: mean(-log(p_winner)) across races"""
    # Filter winners only
    winners = df[df[rank_col] == 1].copy()
    
    if len(winners) == 0:
        return np.nan
    
    # Clip probabilities
    probs = np.clip(winners[prob_col].values, 1e-15, 1 - 1e-15)
    
    # NLL = -log(p)
    nll = -np.log(probs)
    
    return float(nll.mean())


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * np.abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / len(y_true) if len(y_true) > 0 else 0.0


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, name: str, df: pd.DataFrame = None, prob_col: str = None) -> Dict:
    """Evaluate probabilities with optional Race NLL"""
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    result = {
        'name': name,
        'logloss': log_loss(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'auc': roc_auc_score(y_true, y_prob),
        'ece': compute_ece(y_true, y_prob),
        'samples': len(y_true)
    }
    
    # Add Race NLL if df and prob_col provided
    if df is not None and prob_col is not None and prob_col in df.columns:
        result['race_nll'] = compute_race_nll(df, prob_col)
    
    return result


def evaluate_by_odds_bins(df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, name: str) -> List[Dict]:
    """Odds bin evaluation"""
    bins = [(1,2,'1-2'), (2,5,'2-5'), (5,10,'5-10'), (10,20,'10-20'), 
            (20,50,'20-50'), (50,100,'50-100'), (100,9999,'100+')]
    results = []
    for lo, hi, label in bins:
        mask = (df['odds'] >= lo) & (df['odds'] < hi)
        if mask.sum() < 100:
            continue
        y_t = y_true[mask]
        y_p = np.clip(y_prob[mask], 1e-15, 1 - 1e-15)
        try:
            results.append({
                'name': name, 'bin': label, 'count': int(mask.sum()),
                'logloss': log_loss(y_t, y_p), 'ece': compute_ece(y_t, y_p)
            })
        except:
            pass
    return results


def load_feature_columns(dataset_path: str) -> List[str]:
    """Load feature column names from lgbm_datasets.pkl"""
    if not os.path.exists(dataset_path):
        return []
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data.get('feature_cols', [])


def get_default_features() -> List[str]:
    """Default feature set (excluding leak columns)"""
    # These are safe features from v12
    exclude_patterns = ['rank', 'time', 'prize_money', 'odds', 'prob', 'p_market', 
                       'is_winner', 'place', 'finish', 'payout']
    return exclude_patterns


def prepare_features_for_residual(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Prepare features for market-residual model"""
    df = df.copy()
    
    # Add market logit as a feature
    df['baseline_logit'] = safe_logit(df['p_market'].values)
    
    # Filter available feature columns
    available_features = [c for c in feature_cols if c in df.columns]
    
    # Exclude leak columns
    exclude = ['rank', 'time', 'raw_time', 'prize_money', 'prob', 'p_market',
               'prob_model_raw', 'prob_model_norm', 'is_winner', 'finish_time',
               'race_id', 'horse_id', 'horse_key', 'date', 'year', 'fold_year',
               'model_version', 'odds', 'p_market_raw', 'overround']
    
    safe_features = [c for c in available_features if c not in exclude and not c.startswith('prob_')]
    
    return df, safe_features


def train_market_residual_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    params: Dict = None
) -> Tuple[lgb.Booster, pd.DataFrame]:
    """Train market-residual model for one fold"""
    
    if params is None:
        params = LGBM_PARAMS
    
    # Prepare data
    train_df, safe_features = prepare_features_for_residual(train_df, feature_cols)
    valid_df, _ = prepare_features_for_residual(valid_df, feature_cols)
    
    # Include baseline_logit as feature (model will learn delta on top of it)
    train_features = safe_features + ['baseline_logit']
    
    X_train = train_df[train_features].values
    y_train = (train_df['rank'] == 1).astype(int).values
    
    X_valid = valid_df[train_features].values
    y_valid = (valid_df['rank'] == 1).astype(int).values
    
    logger.info(f"Training with {len(train_features)} features, {len(X_train):,} train rows, {len(X_valid):,} valid rows")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=train_features)
    valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=train_features, reference=train_data)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predict on validation
    pred_raw = model.predict(X_valid, num_iteration=model.best_iteration)
    
    # Create result DataFrame
    result_df = valid_df[['race_id', 'horse_id', 'date', 'year', 'p_market', 'odds', 'rank']].copy()
    result_df['prob_residual_raw'] = pred_raw
    
    # Store score_logit (baseline + delta)
    baseline_logit = valid_df['baseline_logit'].values
    result_df['score_logit'] = safe_logit(pred_raw)  # This is the combined logit from model output
    result_df['delta_logit'] = result_df['score_logit'] - baseline_logit
    
    # Method A: Normalize per race (divide by sum)
    result_df['prob_residual_norm'] = normalize_prob_per_race(
        result_df['prob_residual_raw'].values,
        result_df['race_id'].values
    )
    
    # Method B: Softmax per race (more principled for multi-class)
    result_df['prob_residual_softmax'] = softmax_per_race(
        result_df['score_logit'].values,
        result_df['race_id'].values
    )
    
    return model, result_df


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    wf_years: List[int],
    screen_year: int = 2024,
    model_out_dir: str = None
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Run Walk-Forward training and evaluation"""
    
    all_predictions = []
    wf_results = []
    
    # Walk-forward: for each year, train on previous years
    all_years = sorted(wf_years + [screen_year])
    
    for eval_year in all_years:
        train_years = [y for y in range(2021, eval_year)]
        
        if not train_years:
            logger.warning(f"No training years for eval_year={eval_year}, skipping")
            continue
        
        train_df = df[df['year'].isin(train_years)].copy()
        valid_df = df[df['year'] == eval_year].copy()
        
        # Intersection filter
        train_df = train_df[
            (train_df['p_market'].notna()) & 
            (train_df['odds'].notna()) & 
            (train_df['rank'].notna())
        ]
        valid_df = valid_df[
            (valid_df['p_market'].notna()) & 
            (valid_df['odds'].notna()) & 
            (valid_df['rank'].notna())
        ]
        
        if len(train_df) < 1000 or len(valid_df) < 100:
            logger.warning(f"Insufficient data for eval_year={eval_year}")
            continue
        
        logger.info(f"=== Fold: Train {train_years} -> Eval {eval_year} ===")
        
        # Train
        model, pred_df = train_market_residual_fold(train_df, valid_df, feature_cols)
        
        # Add metadata
        pred_df['fold_year'] = eval_year
        pred_df['model_version'] = 'v13_market_residual'
        
        all_predictions.append(pred_df)
        
        # Save model
        if model_out_dir:
            os.makedirs(model_out_dir, exist_ok=True)
            model.save_model(os.path.join(model_out_dir, f'v13_fold_{eval_year}.txt'))
        
        # Evaluate
        y_true = (pred_df['rank'] == 1).astype(int).values
        
        results = {
            'eval_year': eval_year,
            'train_samples': len(train_df),
            'eval_samples': len(pred_df),
            'n_races': pred_df['race_id'].nunique(),
            'market': evaluate_probs(y_true, pred_df['p_market'].values, 'Market', pred_df, 'p_market'),
            'v13_raw': evaluate_probs(y_true, pred_df['prob_residual_raw'].values, 'v13_raw', pred_df, 'prob_residual_raw'),
            'v13_norm': evaluate_probs(y_true, pred_df['prob_residual_norm'].values, 'v13_norm', pred_df, 'prob_residual_norm'),
            'v13_softmax': evaluate_probs(y_true, pred_df['prob_residual_softmax'].values, 'v13_softmax', pred_df, 'prob_residual_softmax'),
        }
        
        # Odds bin evaluation
        results['odds_bins_market'] = evaluate_by_odds_bins(pred_df, y_true, pred_df['p_market'].values, 'Market')
        results['odds_bins_v13_norm'] = evaluate_by_odds_bins(pred_df, y_true, pred_df['prob_residual_norm'].values, 'v13_norm')
        results['odds_bins_v13_softmax'] = evaluate_by_odds_bins(pred_df, y_true, pred_df['prob_residual_softmax'].values, 'v13_softmax')
        
        wf_results.append(results)
        
        logger.info(f"  Market LL={results['market']['logloss']:.5f}, v13_norm LL={results['v13_norm']['logloss']:.5f}, v13_softmax LL={results['v13_softmax']['logloss']:.5f}")
    
    # Combine all predictions
    combined_df = pd.concat(all_predictions, ignore_index=True)
    
    return combined_df, wf_results


def generate_report(wf_results: List[Dict], output_path: str, model_version: str = 'v13_market_residual'):
    """Generate WF evaluation report"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = f"""# Phase 6: Market-Residual Model ({model_version}) WF Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model**: {model_version}
**Architecture**: logit(p_win) = logit(p_market) + Δ(features)

## Walk-Forward Results

"""
    
    for wf in wf_results:
        n_races = wf.get('n_races', wf['eval_samples'] // 13)  # approx
        report += f"### Eval Year: {wf['eval_year']} (Rows: {wf['eval_samples']:,}, Races: {n_races:,})\n\n"
        report += "| Method | LogLoss | RaceNLL | Brier | AUC | ECE |\n"
        report += "|--------|---------|---------|-------|-----|-----|\n"
        
        for key in ['market', 'v13_raw', 'v13_norm', 'v13_softmax']:
            r = wf[key]
            race_nll = r.get('race_nll', 0) or 0
            report += f"| {r['name']} | {r['logloss']:.5f} | {race_nll:.5f} | {r['brier']:.5f} | {r['auc']:.5f} | {r['ece']:.5f} |\n"
        
        # Odds bins (norm vs softmax)
        report += "\n#### Odds Bin LogLoss (norm vs softmax)\n\n"
        report += "| Bin | Market | v13_norm | v13_softmax | Best |\n"
        report += "|-----|--------|----------|-------------|------|\n"
        
        market_bins = {b['bin']: b['logloss'] for b in wf['odds_bins_market']}
        norm_bins = {b['bin']: b['logloss'] for b in wf.get('odds_bins_v13_norm', [])}
        softmax_bins = {b['bin']: b['logloss'] for b in wf.get('odds_bins_v13_softmax', [])}
        
        for bin_name in ['1-2', '2-5', '5-10', '10-20', '20-50', '50-100', '100+']:
            if bin_name in market_bins:
                m_ll = market_bins[bin_name]
                n_ll = norm_bins.get(bin_name, 0)
                s_ll = softmax_bins.get(bin_name, 0)
                best = 'norm' if n_ll < s_ll else 'softmax'
                report += f"| {bin_name} | {m_ll:.4f} | {n_ll:.4f} | {s_ll:.4f} | {best} |\n"
        
        report += "\n"
    
    # Summary
    report += "## Summary\n\n"
    report += "| Year | Market LL | v13_norm LL | v13_softmax LL | Best | RaceNLL Comparison |\n"
    report += "|------|-----------|-------------|----------------|------|-------------------|\n"
    
    for wf in wf_results:
        m_ll = wf['market']['logloss']
        n_ll = wf['v13_norm']['logloss']
        s_ll = wf['v13_softmax']['logloss']
        best = 'norm' if n_ll < s_ll else 'softmax'
        m_nll = wf['market'].get('race_nll', 0) or 0
        n_nll = wf['v13_norm'].get('race_nll', 0) or 0
        s_nll = wf['v13_softmax'].get('race_nll', 0) or 0
        nll_best = 'norm' if n_nll < s_nll else 'softmax'
        report += f"| {wf['eval_year']} | {m_ll:.5f} | {n_ll:.5f} | {s_ll:.5f} | {best} | M:{m_nll:.3f} N:{n_nll:.3f} S:{s_nll:.3f} ({nll_best}) |\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Train Market-Residual Model")
    parser.add_argument('--input', type=str, default='data/predictions/calibrated/v12_oof_unified.parquet')
    parser.add_argument('--dataset_pkl', type=str, default='data/processed/lgbm_datasets_v12.pkl')
    parser.add_argument('--pred_out', type=str, default='data/predictions/v13_market_residual_oof.parquet')
    parser.add_argument('--model_out', type=str, default='models/v13_market_residual')
    parser.add_argument('--report_out', type=str, default='reports/phase6_market_residual_wf.md')
    parser.add_argument('--wf_years', type=int, nargs='+', default=[2021, 2022, 2023])
    parser.add_argument('--screen_year', type=int, default=2024)
    parser.add_argument('--jra_only', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    logger.info(f"Loaded: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Load feature columns
    feature_cols = load_feature_columns(args.dataset_pkl)
    if not feature_cols:
        # Fallback: use all numeric columns except known problematic ones
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    logger.info(f"Feature columns: {len(feature_cols)}")
    
    # Run Walk-Forward
    predictions_df, wf_results = run_walk_forward(
        df, feature_cols, args.wf_years, args.screen_year, args.model_out
    )
    
    # Save predictions
    os.makedirs(os.path.dirname(args.pred_out), exist_ok=True)
    predictions_df.to_parquet(args.pred_out, index=False)
    logger.info(f"Predictions saved to {args.pred_out}")
    
    # Generate report
    generate_report(wf_results, args.report_out)
    
    # Verify normalization
    norm_check = predictions_df.groupby('race_id')['prob_residual_norm'].sum()
    n_bad = ((norm_check - 1).abs() > 0.01).sum()
    logger.info(f"Normalization check: {n_bad} races with |sum - 1| > 0.01")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

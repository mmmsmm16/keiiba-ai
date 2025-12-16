"""
Phase 6: Calibration Check
モデル確率のCalibration評価（ECE）

Usage (in container):
    docker compose exec app python src/evaluation/calibration_check.py --period screening
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import log_loss, brier_score_loss

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.period_guard import add_period_args, parse_period_args, filter_dataframe_by_period, PeriodConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) を計算
    
    ECE = sum(|bin_count| / n * |accuracy - confidence|)
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        # Bin mask
        in_bin = (y_pred > bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            # Accuracy (actual positive rate in bin)
            bin_accuracy = np.mean(y_true[in_bin])
            # Confidence (average predicted probability in bin)
            bin_confidence = np.mean(y_pred[in_bin])
            # Weighted absolute difference
            ece += (bin_count / n_samples) * np.abs(bin_accuracy - bin_confidence)
    
    return ece


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibration curve data を計算
    
    Returns:
        (bin_centers, bin_accuracies, bin_counts)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        in_bin = (y_pred > bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
        bin_counts[i] = np.sum(in_bin)
        
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(y_true[in_bin])
        else:
            bin_accuracies[i] = np.nan
    
    return bin_centers, bin_accuracies, bin_counts


class CalibrationEvaluator:
    """Calibration評価器"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.results = {}
    
    def evaluate(
        self,
        df: pd.DataFrame,
        prob_cols: Dict[str, str],  # {'model': 'prob', 'market': 'p_market'}
        target_col: str = 'rank'
    ) -> Dict:
        """
        複数確率カラムのCalibration評価
        
        Args:
            df: Data with probability columns
            prob_cols: Mapping of name -> column name
            target_col: Target column name
        """
        # ターゲット作成
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        y_win = (df[target_col] == 1).astype(int).values
        y_top3 = (df[target_col] <= 3).astype(int).values
        
        results = {}
        
        for name, col in prob_cols.items():
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping {name}")
                continue
            
            probs = df[col].values
            valid_mask = ~np.isnan(probs)
            
            if valid_mask.sum() == 0:
                continue
            
            probs_valid = np.clip(probs[valid_mask], 1e-7, 1 - 1e-7)
            y_win_valid = y_win[valid_mask]
            y_top3_valid = y_top3[valid_mask]
            
            # Win evaluation
            ece_win = compute_ece(y_win_valid, probs_valid, self.n_bins)
            ll_win = log_loss(y_win_valid, probs_valid)
            bs_win = brier_score_loss(y_win_valid, probs_valid)
            
            results[name] = {
                'ece_win': ece_win,
                'logloss_win': ll_win,
                'brier_win': bs_win,
                'n_samples': int(valid_mask.sum())
            }
            
            logger.info(f"{name}: ECE={ece_win:.5f}, LogLoss={ll_win:.5f}, Brier={bs_win:.5f}")
        
        self.results = results
        return results
    
    def compare_to_baseline(self, model_name: str, baseline_name: str) -> Dict:
        """モデルとベースラインを比較"""
        if model_name not in self.results or baseline_name not in self.results:
            raise ValueError("Both model and baseline must be evaluated first")
        
        model = self.results[model_name]
        baseline = self.results[baseline_name]
        
        comparison = {
            'ece_delta': model['ece_win'] - baseline['ece_win'],
            'logloss_delta': model['logloss_win'] - baseline['logloss_win'],
            'brier_delta': model['brier_win'] - baseline['brier_win'],
            'model_better_ece': model['ece_win'] < baseline['ece_win'],
            'model_better_logloss': model['logloss_win'] < baseline['logloss_win'],
        }
        
        return comparison
    
    def generate_report(self, output_path: str, period: PeriodConfig):
        """レポート生成"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = f"""# Phase 6: Calibration Check Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Period**: {period.start_year}-{period.end_year}

## Calibration Metrics

| Source | ECE (Win) | LogLoss (Win) | Brier (Win) | Samples |
|--------|-----------|---------------|-------------|---------|
"""
        for name, metrics in self.results.items():
            report += f"| **{name}** | {metrics['ece_win']:.5f} | {metrics['logloss_win']:.5f} | {metrics['brier_win']:.5f} | {metrics['n_samples']:,} |\n"
        
        # Comparison
        if 'model' in self.results and 'market' in self.results:
            comp = self.compare_to_baseline('model', 'market')
            report += f"""
## Model vs Market Comparison

| Metric | Delta (Model - Market) | Model Better? |
|--------|------------------------|---------------|
| ECE | {comp['ece_delta']:+.5f} | {'✅' if comp['model_better_ece'] else '❌'} |
| LogLoss | {comp['logloss_delta']:+.5f} | {'✅' if comp['model_better_logloss'] else '❌'} |
| Brier | {comp['brier_delta']:+.5f} | |

"""
            if comp['model_better_logloss']:
                report += "> ✅ **Model provides value over market baseline**\n"
            else:
                report += "> ⚠️ **Model does not beat market baseline**\n"
        
        report += """
## Interpretation

- **ECE (Expected Calibration Error)**: Lower is better. Measures how well predicted probabilities match actual frequencies.
- **LogLoss**: Lower is better. Measures probabilistic accuracy.
- **Brier Score**: Lower is better. Mean squared error of probability predictions.

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Calibration Check"
    )
    add_period_args(parser)
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/derived/preprocessed_with_prob_v12.parquet',
        help='Input data path'
    )
    parser.add_argument(
        '--n_bins',
        type=int,
        default=10,
        help='Number of bins for ECE'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    try:
        period = parse_period_args(args)
    except ValueError as e:
        logger.error(f"Period error: {e}")
        sys.exit(1)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    df = filter_dataframe_by_period(df, period)
    
    # Filter to rows with prob
    df = df[df['prob'].notna()].copy()
    
    # Calculate p_market if needed
    if 'p_market' not in df.columns and 'odds' in df.columns:
        df['raw_prob'] = 1.0 / df['odds'].replace(0, np.nan)
        overround = df.groupby('race_id')['raw_prob'].transform('sum')
        df['p_market'] = df['raw_prob'] / overround
    
    logger.info(f"Data loaded: {len(df):,} rows")
    
    # Evaluate
    evaluator = CalibrationEvaluator(n_bins=args.n_bins)
    
    prob_cols = {'model': 'prob'}
    if 'p_market' in df.columns:
        prob_cols['market'] = 'p_market'
    
    evaluator.evaluate(df, prob_cols)
    evaluator.generate_report(
        os.path.join(args.output_dir, 'phase6_calibration.md'),
        period
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

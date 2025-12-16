"""
Phase 6: Calibration Methods Comparison with Walk-Forward Validation
校正手法比較（Temperature / Isotonic / Beta）WF検証

Usage (in container):
    docker compose exec app python src/evaluation/calibration_methods_wf.py
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.race_filter import filter_jra_only

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            ece += mask.sum() * np.abs(bin_accuracy - bin_confidence)
    
    return ece / len(y_true) if len(y_true) > 0 else 0.0


class TemperatureScaling:
    """温度スケーリング校正"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.optimal_T = 1.0
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray, search_range: Tuple[float, float] = (0.1, 5.0)):
        """最適温度を探索"""
        def nll(T):
            scaled_probs = expit(logits / T)
            scaled_probs = np.clip(scaled_probs, 1e-15, 1 - 1e-15)
            return log_loss(y_true, scaled_probs)
        
        result = minimize_scalar(nll, bounds=search_range, method='bounded')
        self.optimal_T = result.x
        return self
    
    def predict(self, logits: np.ndarray) -> np.ndarray:
        """温度スケーリング適用"""
        return expit(logits / self.optimal_T)
    
    @staticmethod
    def prob_to_logit(prob: np.ndarray) -> np.ndarray:
        """確率をlogitに変換"""
        prob = np.clip(prob, 1e-15, 1 - 1e-15)
        return logit(prob)


class IsotonicCalibrator:
    """Isotonic Regression校正"""
    
    def __init__(self):
        self.iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    
    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        self.iso_reg.fit(probs, y_true)
        return self
    
    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self.iso_reg.predict(probs)


class BetaCalibrator:
    """Beta Calibration"""
    
    def __init__(self):
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
    
    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        """Beta calibrationのパラメータ推定"""
        logits = TemperatureScaling.prob_to_logit(probs)
        
        def nll(params):
            a, b, c = params
            calibrated_logits = a * logits + b
            calibrated_probs = expit(calibrated_logits) + c
            calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
            return log_loss(y_true, calibrated_probs)
        
        from scipy.optimize import minimize
        result = minimize(nll, [1.0, 0.0, 0.0], method='L-BFGS-B',
                          bounds=[(0.1, 10), (-5, 5), (-0.1, 0.1)])
        self.a, self.b, self.c = result.x
        return self
    
    def predict(self, probs: np.ndarray) -> np.ndarray:
        logits = TemperatureScaling.prob_to_logit(probs)
        calibrated_logits = self.a * logits + self.b
        calibrated_probs = expit(calibrated_logits) + self.c
        return np.clip(calibrated_probs, 1e-15, 1 - 1e-15)


def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> Dict:
    """校正メトリクス評価"""
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    
    return {
        'name': name,
        'logloss': log_loss(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'auc': roc_auc_score(y_true, y_prob),
        'ece': compute_ece(y_true, y_prob),
        'samples': len(y_true)
    }


def run_wf_calibration(
    df: pd.DataFrame,
    train_years: List[int],
    eval_year: int,
    prob_col: str = 'prob'
) -> Dict:
    """Walk-Forward校正評価"""
    
    # Train/Eval split
    train_df = df[df['year'].isin(train_years)].copy()
    eval_df = df[df['year'] == eval_year].copy()
    
    logger.info(f"Train years {train_years}: {len(train_df):,} rows")
    logger.info(f"Eval year {eval_year}: {len(eval_df):,} rows")
    
    # Intersection filter: prob存在 & odds存在 & rank存在
    train_df = train_df[(train_df[prob_col].notna()) & (train_df['odds'].notna()) & (train_df['rank'].notna())]
    eval_df = eval_df[(eval_df[prob_col].notna()) & (eval_df['odds'].notna()) & (eval_df['rank'].notna())]
    
    # Win label
    train_df['is_winner'] = (train_df['rank'] == 1).astype(int)
    eval_df['is_winner'] = (eval_df['rank'] == 1).astype(int)
    
    y_train = train_df['is_winner'].values
    y_eval = eval_df['is_winner'].values
    
    prob_train = train_df[prob_col].values
    prob_eval = eval_df[prob_col].values
    
    logits_train = TemperatureScaling.prob_to_logit(prob_train)
    logits_eval = TemperatureScaling.prob_to_logit(prob_eval)
    
    results = []
    
    # 1. Raw (uncalibrated)
    results.append(evaluate_calibration(y_eval, prob_eval, 'Raw'))
    
    # 2. Temperature Scaling
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(logits_train, y_train)
    prob_temp = temp_scaler.predict(logits_eval)
    res_temp = evaluate_calibration(y_eval, prob_temp, f'Temperature (T={temp_scaler.optimal_T:.3f})')
    res_temp['temperature'] = temp_scaler.optimal_T
    results.append(res_temp)
    
    # 3. Isotonic Regression
    iso_cal = IsotonicCalibrator()
    iso_cal.fit(prob_train, y_train)
    prob_iso = iso_cal.predict(prob_eval)
    results.append(evaluate_calibration(y_eval, prob_iso, 'Isotonic'))
    
    # 4. Beta Calibration
    try:
        beta_cal = BetaCalibrator()
        beta_cal.fit(prob_train, y_train)
        prob_beta = beta_cal.predict(prob_eval)
        res_beta = evaluate_calibration(y_eval, prob_beta, f'Beta (a={beta_cal.a:.2f})')
        results.append(res_beta)
    except Exception as e:
        logger.warning(f"Beta calibration failed: {e}")
    
    # 5. Market baseline
    eval_df['p_market_raw'] = 1.0 / eval_df['odds']
    eval_df['p_market'] = eval_df.groupby('race_id')['p_market_raw'].transform(lambda x: x / x.sum())
    results.append(evaluate_calibration(y_eval, eval_df['p_market'].values, 'Market'))
    
    return {
        'train_years': train_years,
        'eval_year': eval_year,
        'results': results,
        'optimal_temperature': temp_scaler.optimal_T
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Calibration Methods WF Comparison")
    parser.add_argument('--input', type=str, default='data/derived/preprocessed_with_prob_v12.parquet')
    parser.add_argument('--output_dir', type=str, default='reports')
    parser.add_argument('--config_dir', type=str, default='config')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    # JRA-only filter
    df = filter_jra_only(df)
    
    logger.info(f"JRA-only: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Walk-Forward validation: train on 2021-2023, eval on 2024
    wf_results = []
    
    # WF fold 1: train 2021-2022, eval 2023
    wf_results.append(run_wf_calibration(df, [2021, 2022], 2023))
    
    # WF fold 2: train 2021-2023, eval 2024
    wf_results.append(run_wf_calibration(df, [2021, 2022, 2023], 2024))
    
    # Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'phase6_calibration_methods_wf.md')
    
    report = f"""# Phase 6: Calibration Methods WF Comparison

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Filter**: JRA-only (intersection: prob & odds & rank)

## Walk-Forward Results

"""
    
    for wf in wf_results:
        report += f"### Fold: Train {wf['train_years']} → Eval {wf['eval_year']}\n\n"
        report += f"| Method | LogLoss | Brier | AUC | ECE | Samples |\n"
        report += f"|--------|---------|-------|-----|-----|---------|\n"
        
        for r in wf['results']:
            ll = r['logloss']
            br = r['brier']
            auc = r['auc']
            ece = r['ece']
            n = r['samples']
            name = r['name']
            report += f"| {name} | {ll:.5f} | {br:.5f} | {auc:.5f} | {ece:.5f} | {n:,} |\n"
        
        report += f"\n**Optimal Temperature**: {wf['optimal_temperature']:.4f}\n\n"
    
    # Summary
    best_methods = []
    for wf in wf_results:
        # Exclude Market and find best LogLoss
        model_results = [r for r in wf['results'] if r['name'] != 'Market']
        best = min(model_results, key=lambda x: x['logloss'])
        best_methods.append({'year': wf['eval_year'], 'method': best['name'], 'logloss': best['logloss']})
    
    report += """## Summary

| Eval Year | Best Method | LogLoss |
|-----------|-------------|---------|
"""
    for b in best_methods:
        report += f"| {b['year']} | {b['method']} | {b['logloss']:.5f} |\n"
    
    # Temperature config
    latest_temp = wf_results[-1]['optimal_temperature']
    
    report += f"""
## Recommendation

- **Optimal Temperature**: {latest_temp:.4f} (2024評価ベース)
- Temperature scaling is recommended for production use

"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    
    # Save temperature config
    os.makedirs(args.config_dir, exist_ok=True)
    config_path = os.path.join(args.config_dir, 'temperature.yaml')
    
    config = {
        'temperature': {
            'win': float(latest_temp),
            'top3': 1.25,  # From previous exploration
            'validated_on': '2024',
            'method': 'minimize_logloss'
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Temperature config saved to {config_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

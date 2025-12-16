"""
Phase 6: Market Blend Lambda Exploration with WF Validation
市場確率ブレンドのλ探索・検証

Usage (in container):
    docker compose exec app python src/evaluation/market_blend_lambda_wf.py --prob_base prob_model_norm
    docker compose exec app python src/evaluation/market_blend_lambda_wf.py --prob_base prob_model_calib_isotonic
    docker compose exec app python src/evaluation/market_blend_lambda_wf.py --prob_base prob_model_calib_beta_full
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.race_filter import filter_jra_only
from utils.calibration import (
    MarketBlend, normalize_prob_per_race
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Available prob_base options
VALID_PROB_BASES = [
    'prob_model_raw',
    'prob_model_norm',
    'prob_model_calib_temp',
    'prob_model_calib_isotonic',
    'prob_model_calib_beta_full',
]


def log_input_schema(df: pd.DataFrame, prob_base: str):
    """Check if prob_base exists in data"""
    logger.info(f"=== Input Data Schema ===")
    logger.info(f"Rows: {len(df):,}, Columns: {len(df.columns)}")
    
    # Check prob columns
    for col in VALID_PROB_BASES + ['p_market', 'odds', 'rank']:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            pct = n_valid / len(df) * 100
            marker = " ✅" if col == prob_base else ""
            logger.info(f"  {col}: {n_valid:,} ({pct:.1f}%){marker}")
        else:
            marker = " ❌ MISSING" if col == prob_base else ""
            logger.info(f"  {col}: NOT FOUND{marker}")
    
    if prob_base not in df.columns or df[prob_base].isna().all():
        raise ValueError(f"prob_base '{prob_base}' not found or all NaN. "
                        f"Run: python src/phase6/build_predictions_table.py first")


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


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> Dict:
    """確率評価"""
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    
    return {
        'name': name,
        'logloss': log_loss(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'auc': roc_auc_score(y_true, y_prob),
        'ece': compute_ece(y_true, y_prob),
        'samples': len(y_true)
    }


def evaluate_by_odds_bins(
    df: pd.DataFrame, 
    y_true: np.ndarray, 
    y_prob: np.ndarray,
    prob_name: str
) -> List[Dict]:
    """オッズbin別評価"""
    odds_bins = [
        (1.0, 2.0, '1-2'),
        (2.0, 5.0, '2-5'),
        (5.0, 10.0, '5-10'),
        (10.0, 20.0, '10-20'),
        (20.0, 50.0, '20-50'),
        (50.0, 100.0, '50-100'),
        (100.0, 9999.0, '100+'),
    ]
    
    results = []
    for min_o, max_o, label in odds_bins:
        mask = (df['odds'] >= min_o) & (df['odds'] < max_o)
        if mask.sum() < 100:
            continue
        
        y_t = y_true[mask]
        y_p = np.clip(y_prob[mask], 1e-15, 1 - 1e-15)
        
        try:
            ll = log_loss(y_t, y_p)
            ece = compute_ece(y_t, y_p)
        except:
            continue
        
        results.append({
            'prob_name': prob_name,
            'odds_bin': label,
            'count': int(mask.sum()),
            'logloss': ll,
            'ece': ece
        })
    
    return results


def run_lambda_search(
    df: pd.DataFrame,
    calib_year: int,
    lambda_grid: List[float],
    prob_base: str = 'prob_model_norm'
) -> Tuple[float, List[Dict]]:
    """
    λ探索（calibration year）
    
    Args:
        df: データ（prob_base, p_market, rank, odds列が必要）
        calib_year: 探索年
        lambda_grid: λグリッド
        prob_base: 使用するモデル確率列
    
    Returns:
        (best_lambda, results)
    """
    calib_df = df[df['year'] == calib_year].copy()
    
    # Intersection filter
    calib_df = calib_df[
        (calib_df[prob_base].notna()) & 
        (calib_df['p_market'].notna()) & 
        (calib_df['odds'].notna()) & 
        (calib_df['odds'] > 0) &
        (calib_df['rank'].notna())
    ]
    
    if len(calib_df) == 0:
        raise ValueError(f"No valid data for calib_year={calib_year} with prob_base={prob_base}")
    
    y_true = (calib_df['rank'] == 1).astype(int).values
    p_market = calib_df['p_market'].values
    p_model = calib_df[prob_base].values
    
    logger.info(f"Lambda search on {calib_year} with {prob_base}: {len(calib_df):,} rows")
    
    results = []
    for lam in lambda_grid:
        blender = MarketBlend(lambda_=lam)
        p_blend = blender.blend(p_market, p_model, normalize_per_race=True,
                                race_ids=calib_df['race_id'].values)
        
        res = evaluate_probs(y_true, p_blend, f'λ={lam:.2f}')
        res['lambda'] = lam
        results.append(res)
    
    # Find best λ by LogLoss
    best_result = min(results, key=lambda x: x['logloss'])
    best_lambda = best_result['lambda']
    
    logger.info(f"Best λ={best_lambda:.2f}, LogLoss={best_result['logloss']:.5f}")
    
    return best_lambda, results


def run_wf_validation(
    df: pd.DataFrame,
    eval_year: int,
    best_lambda: float,
    prob_base: str = 'prob_model_norm'
) -> Dict:
    """Walk-Forward検証"""
    
    eval_df = df[df['year'] == eval_year].copy()
    
    # Intersection filter
    eval_df = eval_df[
        (eval_df[prob_base].notna()) & 
        (eval_df['p_market'].notna()) & 
        (eval_df['odds'].notna()) & 
        (eval_df['odds'] > 0) &
        (eval_df['rank'].notna())
    ]
    
    if len(eval_df) == 0:
        return {'eval_year': eval_year, 'samples': 0, 'overall': [], 'odds_bins': []}
    
    y_true = (eval_df['rank'] == 1).astype(int).values
    p_market = eval_df['p_market'].values
    p_model = eval_df[prob_base].values
    
    logger.info(f"WF eval on {eval_year}: {len(eval_df):,} rows")
    
    results = []
    
    # Market (λ=0)
    results.append(evaluate_probs(y_true, p_market, 'Market (λ=0)'))
    
    # Model (λ=1)
    results.append(evaluate_probs(y_true, p_model, f'Model ({prob_base}, λ=1)'))
    
    # Blend (best λ)
    blender = MarketBlend(lambda_=best_lambda)
    p_blend = blender.blend(p_market, p_model, normalize_per_race=True,
                            race_ids=eval_df['race_id'].values)
    results.append(evaluate_probs(y_true, p_blend, f'Blend (λ={best_lambda:.2f})'))
    
    # Odds bin evaluation
    odds_bin_results = []
    for name, probs in [('Market', p_market), ('Model', p_model), ('Blend', p_blend)]:
        odds_bin_results.extend(evaluate_by_odds_bins(eval_df, y_true, probs, name))
    
    return {
        'eval_year': eval_year,
        'best_lambda': best_lambda,
        'overall': results,
        'odds_bins': odds_bin_results,
        'samples': len(eval_df)
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Market Blend Lambda WF")
    parser.add_argument('--input', type=str, default='data/predictions/calibrated/v12_oof_unified.parquet')
    parser.add_argument('--output_dir', type=str, default='reports')
    parser.add_argument('--config_dir', type=str, default='config')
    parser.add_argument('--calib_year', type=int, default=2024)
    parser.add_argument('--wf_years', type=int, nargs='+', default=[2021, 2022, 2023])
    parser.add_argument('--prob_base', type=str, default='prob_model_norm',
                        choices=VALID_PROB_BASES,
                        help='Model probability column to use for blending')
    parser.add_argument('--lambda_step', type=float, default=0.05)
    
    args = parser.parse_args()
    
    # Lambda grid
    lambda_grid = np.arange(0, 1.01, args.lambda_step).tolist()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    # Log input schema and validate
    log_input_schema(df, args.prob_base)
    
    # JRA-only filter if not already applied
    if 'race_id' in df.columns:
        jra_ratio = df['race_id'].astype(str).str[4:5].eq('0').mean()
        if jra_ratio < 0.99:
            df = filter_jra_only(df)
    
    logger.info(f"Data: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Lambda search on calib_year
    best_lambda, search_results = run_lambda_search(
        df, args.calib_year, lambda_grid, args.prob_base
    )
    
    # WF validation on each wf_year
    wf_results = []
    for year in args.wf_years:
        if year >= args.calib_year:
            continue
        wf_res = run_wf_validation(df, year, best_lambda, args.prob_base)
        wf_results.append(wf_res)
    
    # Generate report (separate file per prob_base)
    os.makedirs(args.output_dir, exist_ok=True)
    prob_base_short = args.prob_base.replace('prob_model_', '').replace('calib_', '')
    report_path = os.path.join(args.output_dir, f'phase6_market_blend_{prob_base_short}.md')
    
    report = f"""# Phase 6: Market Blend Lambda WF Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Filter**: JRA-only (intersection)
**Calibration Year**: {args.calib_year}
**Prob Base**: `{args.prob_base}`

## Lambda Search on {args.calib_year}

| λ | LogLoss | Brier | AUC | ECE |
|---|---------|-------|-----|-----|
"""
    
    for r in search_results:
        marker = ' ✅' if r['lambda'] == best_lambda else ''
        report += f"| {r['lambda']:.2f}{marker} | {r['logloss']:.5f} | {r['brier']:.5f} | {r['auc']:.5f} | {r['ece']:.5f} |\n"
    
    report += f"\n**Best λ = {best_lambda:.2f}**\n\n"
    
    # WF results
    report += "## Walk-Forward Validation\n\n"
    
    for wf in wf_results:
        if wf['samples'] == 0:
            report += f"### Eval Year: {wf['eval_year']}\n\n⚠️ No valid data\n\n"
            continue
            
        report += f"### Eval Year: {wf['eval_year']} (Samples: {wf['samples']:,})\n\n"
        report += "| Method | LogLoss | Brier | AUC | ECE |\n"
        report += "|--------|---------|-------|-----|-----|\n"
        
        for r in wf['overall']:
            report += f"| {r['name']} | {r['logloss']:.5f} | {r['brier']:.5f} | {r['auc']:.5f} | {r['ece']:.5f} |\n"
        
        report += "\n#### Odds Bin LogLoss\n\n"
        report += "| Odds Bin | Market | Model | Blend |\n"
        report += "|----------|--------|-------|-------|\n"
        
        # Group by odds bin
        bin_data = {}
        for ob in wf['odds_bins']:
            bin_name = ob['odds_bin']
            if bin_name not in bin_data:
                bin_data[bin_name] = {}
            bin_data[bin_name][ob['prob_name']] = ob['logloss']
        
        for bin_name in ['1-2', '2-5', '5-10', '10-20', '20-50', '50-100', '100+']:
            if bin_name in bin_data:
                bd = bin_data[bin_name]
                m = bd.get('Market', 0)
                mo = bd.get('Model', 0)
                bl = bd.get('Blend', 0)
                report += f"| {bin_name} | {m:.4f} | {mo:.4f} | {bl:.4f} |\n"
        
        report += "\n"
    
    # Summary
    report += f"""## Summary

- **Prob Base**: `{args.prob_base}`
- **Best λ = {best_lambda:.2f}**
- λ=0: Pure market (no model)
- λ=1: Pure model (no market)
- λ={best_lambda:.2f}: Optimal blend

"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    
    # Save config
    os.makedirs(args.config_dir, exist_ok=True)
    config_path = os.path.join(args.config_dir, f'blend_{prob_base_short}.yaml')
    
    config = {
        'blend': {
            'best_lambda': float(best_lambda),
            'calibration_year': args.calib_year,
            'prob_base': args.prob_base,
            'method': 'market_model_blend'
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Config saved to {config_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()


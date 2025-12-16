"""
Phase 5: Win Optimization with Multiple Probability Columns
複数の確率列でROI/MaxDD/MinFoldROIを比較

Usage (in container):
    docker compose exec app python src/phase5/prob_sweep_roi.py \
        --input data/predictions/v13_market_residual_oof.parquet \
        --prob_cols p_market,prob_residual_norm,prob_residual_softmax
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.race_filter import filter_jra_only

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Default betting strategy
DEFAULT_MIN_EV = 1.2
DEFAULT_MAX_ODDS = 50.0
DEFAULT_MIN_PROB = 0.05
DEFAULT_TOP_N = 3


def simulate_win_strategy(
    df: pd.DataFrame,
    prob_col: str,
    min_ev: float = DEFAULT_MIN_EV,
    max_odds: float = DEFAULT_MAX_ODDS,
    min_prob: float = DEFAULT_MIN_PROB,
    top_n: int = DEFAULT_TOP_N,
    bet_amount: int = 100,
    return_bets_df: bool = False
) -> Dict:
    """
    単勝購入シミュレーション
    
    購入条件:
    - EV = prob * odds >= min_ev
    - odds <= max_odds
    - prob >= min_prob
    - レース内順位 <= top_n
    """
    # Add prob rank per race
    df = df.copy()
    df['prob_rank'] = df.groupby('race_id')[prob_col].rank(ascending=False, method='first')
    
    # Intersection filter
    valid_df = df[
        (df[prob_col].notna()) &
        (df['odds'].notna()) &
        (df['odds'] > 0) &
        (df['rank'].notna())
    ].copy()
    
    # Calculate EV
    valid_df['ev'] = valid_df[prob_col] * valid_df['odds']
    
    # Bet filter
    bet_mask = (
        (valid_df['ev'] >= min_ev) &
        (valid_df['odds'] <= max_odds) &
        (valid_df[prob_col] >= min_prob) &
        (valid_df['prob_rank'] <= top_n)
    )
    
    bets_df = valid_df[bet_mask].copy()
    
    if len(bets_df) == 0:
        result = {
            'prob_col': prob_col,
            'total_bets': 0,
            'total_cost': 0,
            'total_return': 0,
            'roi': 0.0,
            'hit_rate': 0.0,
            'avg_odds': 0.0,
            'max_dd': 0.0,
            'min_fold_roi': 0.0,
        }
        if return_bets_df:
            result['bets_df'] = pd.DataFrame()
        return result
    
    # Calculate results
    bets_df['is_win'] = (bets_df['rank'] == 1).astype(int)
    bets_df['bet_amount'] = bet_amount
    bets_df['returns'] = bets_df['is_win'] * bets_df['odds'] * bet_amount
    bets_df['profit'] = bets_df['returns'] - bets_df['bet_amount']
    bets_df['payout'] = bets_df['returns']  # Alias for sanity check
    
    # Rename prob column for sanity check
    bets_df['prob'] = bets_df[prob_col]
    
    # Year-wise results
    yearly = bets_df.groupby('year').agg({
        'bet_amount': 'sum',
        'returns': 'sum',
        'is_win': 'sum',
        'profit': 'sum'
    }).reset_index()
    yearly['cost'] = yearly['bet_amount']
    yearly['roi'] = yearly['returns'] / yearly['cost'] * 100
    
    # Max Drawdown calculation
    cumulative_profit = bets_df.sort_values('date')['profit'].cumsum()
    running_max = cumulative_profit.cummax()
    drawdown = running_max - cumulative_profit
    max_dd = drawdown.max() if len(drawdown) > 0 else 0
    
    total_cost = bets_df['bet_amount'].sum()
    total_return = bets_df['returns'].sum()
    
    result = {
        'prob_col': prob_col,
        'total_bets': len(bets_df),
        'n_races': bets_df['race_id'].nunique(),
        'total_cost': int(total_cost),
        'total_return': int(total_return),
        'roi': float(total_return / total_cost * 100) if total_cost > 0 else 0,
        'hit_rate': float(bets_df['is_win'].mean() * 100),
        'avg_odds': float(bets_df['odds'].mean()),
        'avg_ev': float(bets_df['ev'].mean()),
        'max_dd': float(max_dd),
        'min_fold_roi': float(yearly['roi'].min()) if len(yearly) > 0 else 0,
        'yearly': yearly.to_dict('records')
    }
    
    if return_bets_df:
        result['bets_df'] = bets_df
    
    return result


def run_prob_sweep(
    df: pd.DataFrame,
    prob_cols: List[str],
    **strategy_params
) -> List[Dict]:
    """Run simulation for multiple probability columns"""
    results = []
    
    for prob_col in prob_cols:
        if prob_col not in df.columns:
            logger.warning(f"Column {prob_col} not found, skipping")
            continue
        
        logger.info(f"Simulating with prob_col={prob_col}")
        res = simulate_win_strategy(df, prob_col, **strategy_params)
        results.append(res)
        
        logger.info(f"  ROI={res['roi']:.1f}%, Hits={res['hit_rate']:.1f}%, MaxDD={res['max_dd']:.0f}")
    
    return results


def generate_report(
    results: List[Dict],
    output_path: str,
    strategy_params: Dict,
    sanity_results: List = None,
    odds_info: Dict = None
):
    """Generate comparison report with sanity checks"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Odds info section
    odds_section = ""
    if odds_info:
        odds_section = f"""
**Odds Configuration** (IMPORTANT):
- Odds Source: `{odds_info.get('odds_source', 'N/A')}`
- Slippage Factor: {odds_info.get('slippage_factor', 1.0)}
- Allow Final Odds: {odds_info.get('allow_final_odds', False)}
- Payout Match Rate: {odds_info.get('payout_match_rate', 'N/A')}

"""
        if odds_info.get('odds_source') == 'final':
            odds_section += "> ⚠️ WARNING: Using final odds which match confirmed payouts. This may not reflect real executable odds.\n\n"
    
    report = f"""# Phase 5: Win Optimization - Probability Column Comparison

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Strategy Params**:
- Min EV: {strategy_params.get('min_ev', DEFAULT_MIN_EV)}
- Max Odds: {strategy_params.get('max_odds', DEFAULT_MAX_ODDS)}
- Min Prob: {strategy_params.get('min_prob', DEFAULT_MIN_PROB)}
- Top N: {strategy_params.get('top_n', DEFAULT_TOP_N)}
- Bet Amount: ¥{strategy_params.get('bet_amount', 100)}
{odds_section}
## Summary Comparison

| Prob Column | Bets | Hit% | ROI | MaxDD | MinFold ROI | AvgOdds | AvgEV |
|-------------|------|------|-----|-------|-------------|---------|-------|
"""
    
    for r in results:
        report += f"| {r['prob_col']} | {r['total_bets']:,} | {r['hit_rate']:.1f}% | **{r['roi']:.1f}%** | ¥{r['max_dd']:,.0f} | {r['min_fold_roi']:.1f}% | {r['avg_odds']:.1f} | {r['avg_ev']:.2f} |\n"
    
    # Best by ROI
    if results:
        best_roi = max(results, key=lambda x: x['roi'])
        best_minf = max(results, key=lambda x: x['min_fold_roi'])
        
        report += f"\n**Best ROI**: `{best_roi['prob_col']}` ({best_roi['roi']:.1f}%)\n"
        report += f"**Best MinFold ROI**: `{best_minf['prob_col']}` ({best_minf['min_fold_roi']:.1f}%)\n\n"
    
    # Yearly breakdown
    report += "## Yearly Breakdown\n\n"
    
    for r in results:
        report += f"### {r['prob_col']}\n\n"
        report += "| Year | Bets | Cost | Return | ROI |\n"
        report += "|------|------|------|--------|-----|\n"
        
        for yr in r.get('yearly', []):
            report += f"| {int(yr['year'])} | {int(yr['is_win'])} wins | ¥{int(yr['cost']):,} | ¥{int(yr['returns']):,} | {yr['roi']:.1f}% |\n"
        
        report += "\n"
    
    # Sanity Checks section
    if sanity_results:
        report += "## Sanity Checks (Win)\n\n"
        
        all_passed = all(r.passed for r in sanity_results)
        status = "✅ ALL PASSED" if all_passed else "❌ SOME FAILED"
        report += f"**Status**: {status}\n\n"
        
        report += "| Check | Status | Details |\n"
        report += "|-------|--------|----------|\n"
        
        for sr in sanity_results:
            status = "✅ PASS" if sr.passed else "❌ FAIL"
            msg = sr.message[:50] + "..." if len(sr.message) > 50 else sr.message
            report += f"| {sr.check_name} | {status} | {msg} |\n"
        
        report += "\n### Odds vs Payout Details\n\n"
        for sr in sanity_results:
            if sr.check_name == 'odds_vs_payout_separation' and 'odds_stats' in sr.details:
                odds_s = sr.details['odds_stats']
                payout_s = sr.details['payout_stats']
                
                report += "**Distribution:**\n"
                report += "| Statistic | Odds | Payout |\n"
                report += "|-----------|------|--------|\n"
                report += f"| Min | {odds_s['min']:.1f} | {payout_s['min']:.0f} |\n"
                report += f"| Median | {odds_s['median']:.1f} | {payout_s['median']:.0f} |\n"
                report += f"| P95 | {odds_s['p95']:.1f} | {payout_s['p95']:.0f} |\n"
                report += f"| Max | {odds_s['max']:.1f} | {payout_s['max']:.0f} |\n"
                
                if 'exact_match_rate' in sr.details:
                    report += f"\n**Payout/Bet vs Odds Match Rate**: {sr.details['exact_match_rate']:.1%}\n"
                
                if 'diff_stats' in sr.details:
                    ds = sr.details['diff_stats']
                    report += f"\n**Diff (payout/bet - odds) Distribution:**\n"
                    report += f"- Min: {ds['min']:.4f}\n"
                    report += f"- Median: {ds['median']:.4f}\n"
                    report += f"- Mean: {ds['mean']:.4f}\n"
                    report += f"- P95: {ds['p95']:.4f}\n"
                    report += f"- Max: {ds['max']:.4f}\n"
                    report += f"- Zero Rate (|diff|<0.01): {ds['zero_rate']:.1%}\n"
        
        # p_market recomputation section
        report += "\n### p_market Recomputation\n\n"
        for sr in sanity_results:
            if sr.check_name == 'p_market_recomputation' and 'diff_stats' in sr.details:
                ds = sr.details['diff_stats']
                report += f"**p_market = (1/odds) / Σ(1/odds) Check:**\n"
                report += f"- Max Diff: {ds['max']:.2e}\n"
                report += f"- P99 Diff: {ds['p99']:.2e}\n"
                report += f"- Mean Diff: {ds['mean']:.2e}\n"
                report += f"- Above Tolerance Rate: {ds['above_tol_rate']:.2%}\n"
                report += f"- Total Rows: {sr.details.get('total_rows', 'N/A')}\n"
                report += f"- Total Races: {sr.details.get('total_races', 'N/A')}\n"
    
    # Recommendation
    report += """\n## Recommendation

Based on the comparison:
"""
    if results:
        best = max(results, key=lambda x: x['roi'])
        report += f"- **Recommended prob column**: `{best['prob_col']}`\n"
        report += f"- **Expected ROI**: {best['roi']:.1f}%\n"
        report += f"- **Risk (MaxDD)**: ¥{best['max_dd']:,.0f}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Win Optimization Prob Sweep")
    parser.add_argument('--input', type=str, default='data/predictions/v13_market_residual_oof.parquet')
    parser.add_argument('--report_out', type=str, default='reports/phase5_win_optimization_v13_prob_compare.md')
    parser.add_argument('--sanity_out', type=str, default='reports/phase5_sanity_sample.md')
    parser.add_argument('--prob_cols', type=str, default='p_market,prob_residual_norm,prob_residual_softmax')
    parser.add_argument('--min_ev', type=float, default=DEFAULT_MIN_EV)
    parser.add_argument('--max_odds', type=float, default=DEFAULT_MAX_ODDS)
    parser.add_argument('--min_prob', type=float, default=DEFAULT_MIN_PROB)
    parser.add_argument('--top_n', type=int, default=DEFAULT_TOP_N)
    parser.add_argument('--bet_amount', type=int, default=100)
    parser.add_argument('--jra_only', action='store_true', default=True)
    parser.add_argument('--years', type=int, nargs='+', default=[2022, 2023, 2024])
    parser.add_argument('--strict_sanity', action='store_true', default=True,
                        help='Fail on sanity check violations')
    # Odds source options (Leak Guard)
    parser.add_argument('--odds_source', type=str, default='final',
                        choices=['pre_close', 'close', 'final'],
                        help='Odds timestamp source: pre_close, close, or final')
    parser.add_argument('--allow_final_odds', action='store_true', default=False,
                        help='Explicitly allow final odds usage (required when odds_source=final)')
    parser.add_argument('--slippage_factor', type=float, default=1.0,
                        help='Multiply odds by this factor (e.g., 0.95 for conservative)')
    
    args = parser.parse_args()
    
    # LEAK GUARD: Require explicit flag for final odds
    if args.odds_source == 'final' and not args.allow_final_odds:
        raise ValueError(
            "LEAK GUARD: --odds_source=final requires --allow_final_odds flag.\n"
            "Final odds may match confirmed payouts and could leak post-close information.\n"
            "Add --allow_final_odds to explicitly acknowledge this risk, or use --slippage_factor<1 for conservative evaluation."
        )
    
    # Warning for high slippage
    if args.slippage_factor < 0.8:
        logger.warning(f"Very high slippage ({args.slippage_factor}) - results may be overly pessimistic")
    elif args.slippage_factor < 1.0:
        logger.info(f"Slippage factor: {args.slippage_factor} (conservative evaluation)")
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    # JRA filter if needed
    if args.jra_only and df['race_id'].astype(str).str[4:5].eq('0').mean() < 0.99:
        df = filter_jra_only(df)
    
    # Year filter
    df = df[df['year'].isin(args.years)]
    
    # Apply odds source selection
    odds_col_map = {
        'pre_close': 'odds_pre_close',
        'close': 'odds_close', 
        'final': 'odds'  # Default column
    }
    target_odds_col = odds_col_map.get(args.odds_source, 'odds')
    
    # Check if target column exists, fall back to 'odds'
    if target_odds_col not in df.columns:
        if args.odds_source != 'final':
            logger.warning(f"Column {target_odds_col} not found, falling back to 'odds' (may be final odds)")
        target_odds_col = 'odds'
    
    # Store original odds for payout calculation, create adjusted odds for betting decisions
    df['odds_original'] = df['odds'].copy()
    df['odds_used'] = df[target_odds_col] * args.slippage_factor
    
    # Track excluded rows
    bad_odds = df['odds_used'].isna() | (df['odds_used'] <= 1.0)
    n_excluded = bad_odds.sum()
    if n_excluded > 0:
        logger.info(f"Excluded {n_excluded:,} rows with invalid odds_used (<=1 or NaN)")
    
    # For simulation, use adjusted odds for EV calculation but original odds for payout
    df['odds'] = df['odds_used']  # Use adjusted odds for betting decision
    
    # Log odds source info
    logger.info(f"Odds source: {args.odds_source} (column: {target_odds_col})")
    logger.info(f"Slippage factor: {args.slippage_factor}")
    logger.info(f"Allow final odds: {args.allow_final_odds}")
    
    logger.info(f"Data: {len(df):,} rows, {df['race_id'].nunique():,} races, years {df['year'].unique().tolist()}")
    
    # Parse prob_cols
    prob_cols = [c.strip() for c in args.prob_cols.split(',')]
    
    # Strategy params
    strategy_params = {
        'min_ev': args.min_ev,
        'max_odds': args.max_odds,
        'min_prob': args.min_prob,
        'top_n': args.top_n,
        'bet_amount': args.bet_amount
    }
    
    # Run sweep and collect bets for sanity check
    results = []
    all_bets = []
    
    for prob_col in prob_cols:
        if prob_col not in df.columns:
            logger.warning(f"Column {prob_col} not found, skipping")
            continue
        
        logger.info(f"Simulating with prob_col={prob_col}")
        res = simulate_win_strategy(df, prob_col, **strategy_params, return_bets_df=True)
        
        if 'bets_df' in res:
            bets_df = res.pop('bets_df')
            if len(bets_df) > 0:
                bets_df['source_prob_col'] = prob_col
                all_bets.append(bets_df)
        
        results.append(res)
        logger.info(f"  ROI={res['roi']:.1f}%, Hits={res['hit_rate']:.1f}%, MaxDD={res['max_dd']:.0f}")
    
    # Run sanity checks on combined bets
    sanity_results = []
    sample_stats = {}
    if all_bets:
        combined_bets = pd.concat(all_bets, ignore_index=True)
        logger.info(f"Running sanity checks on {len(combined_bets):,} bet rows...")
        
        try:
            from utils.sanity_checks import (
                validate_win_payout_integrity,
                validate_odds_vs_payout_separation, 
                validate_ev_definition,
                validate_p_market_recomputation,
                generate_sample_table
            )
            
            # Check 1: Win/Payout integrity (bets_only=True since this is bet rows only)
            result1 = validate_win_payout_integrity(
                combined_bets, 
                bet_col='bet_amount',
                payout_col='payout',
                strict=False,
                bets_only=True
            )
            sanity_results.append(result1)
            logger.info(f"  Win/Payout Integrity: {'PASS' if result1.passed else 'FAIL'} - {result1.message}")
            
            # Check 2: Odds vs Payout (with diff distribution)
            result2 = validate_odds_vs_payout_separation(
                combined_bets,
                bet_col='bet_amount',
                payout_col='returns'
            )
            sanity_results.append(result2)
            warn_flag = "[WARNING]" if result2.warnings > 0 else ""
            logger.info(f"  Odds/Payout: {warn_flag} {result2.message}")
            
            # Check 3: EV definition
            result3 = validate_ev_definition(combined_bets)
            sanity_results.append(result3)
            logger.info(f"  EV Definition: {'PASS' if result3.passed else 'FAIL'} - {result3.message}")
            
            # Check 4: p_market recomputation (if p_market exists)
            if 'p_market' in df.columns:
                result4 = validate_p_market_recomputation(df)
                sanity_results.append(result4)
                logger.info(f"  p_market Recompute: {'PASS' if result4.passed else 'WARN'} - {result4.message}")
            
            # Generate sample table with FULL RACE DATA for accurate winner extraction
            os.makedirs(os.path.dirname(args.sanity_out), exist_ok=True)
            sample_report, sample_stats = generate_sample_table(
                bets_df=combined_bets,
                full_race_df=df,  # Pass full race data for winner lookup
                n_races=100,
                bet_col='bet_amount',
                payout_col='returns'
            )
            with open(args.sanity_out, 'w', encoding='utf-8') as f:
                f.write(sample_report)
            logger.info(f"Sample table saved to {args.sanity_out}")
            logger.info(f"  Winner stats: {sample_stats['races_with_winner']} found, {sample_stats['races_without_winner']} missing, {sample_stats['multi_winner_races']} multi")
            
            # Strict mode check - validate sample table has no missing winners
            if args.strict_sanity:
                # Check for failed sanity results
                failed = [r for r in sanity_results if not r.passed]
                if failed:
                    msgs = [f"{r.check_name}: {r.message}" for r in failed]
                    raise ValueError(f"Sanity checks FAILED: {'; '.join(msgs)}")
                
                # Check for missing winners in sample
                if sample_stats.get('races_without_winner', 0) > 0:
                    raise ValueError(f"Sample has {sample_stats['races_without_winner']} races without winner detected")
            
        except ImportError as e:
            logger.warning(f"sanity_checks module not found: {e}")
    
    # Build odds info for report
    odds_info = {
        'odds_source': args.odds_source,
        'slippage_factor': args.slippage_factor,
        'allow_final_odds': args.allow_final_odds,
        'payout_match_rate': 'N/A'
    }
    
    # Calculate payout match rate from sanity results
    for sr in sanity_results:
        if sr.check_name == 'odds_vs_payout_separation' and 'exact_match_rate' in sr.details:
            odds_info['payout_match_rate'] = f"{sr.details['exact_match_rate']:.0%}"
    
    # Generate report
    generate_report(results, args.report_out, strategy_params, sanity_results, odds_info)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

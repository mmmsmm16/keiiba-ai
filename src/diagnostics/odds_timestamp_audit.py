"""
Odds Timestamp Audit - Improved Version
オッズの時点/ソースを監査し、バックテスト用オッズの選択を支援

勝ち馬の payout/bet と odds の一致率でオッズ時点を判定

Usage (in container):
    docker compose exec app python src/diagnostics/odds_timestamp_audit.py
    docker compose exec app python src/diagnostics/odds_timestamp_audit.py \
        --input data/predictions/v13_market_residual_oof.parquet
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_odds_columns(df: pd.DataFrame) -> List[str]:
    """Detect all odds-related columns"""
    odds_patterns = ['odds']
    exclude = ['lag', 'used', 'original']
    
    odds_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in odds_patterns):
            if not any(e in col_lower for e in exclude):
                odds_cols.append(col)
    
    return odds_cols


def prepare_data(df: pd.DataFrame, bet_unit: float = 100.0) -> pd.DataFrame:
    """Prepare data with y_win and payout columns"""
    df = df.copy()
    
    # Create y_win from rank
    if 'y_win' not in df.columns:
        if 'rank' in df.columns:
            df['y_win'] = (df['rank'] == 1).astype(int)
        else:
            raise ValueError("Need 'rank' or 'y_win' column to identify winners")
    
    # Create payout from odds for winners (payout = odds * bet_unit for winners, 0 otherwise)
    if 'payout' not in df.columns:
        if 'odds' in df.columns:
            df['payout'] = df['y_win'] * df['odds'] * bet_unit
        else:
            df['payout'] = 0
    
    return df


def analyze_odds_column(
    df: pd.DataFrame, 
    col: str,
    bet_unit: float = 100.0
) -> Dict:
    """Analyze a single odds column for timing determination"""
    valid = df[df[col].notna() & (df[col] > 0)]
    
    stats = {
        'column': col,
        'total_rows': len(df),
        'valid_rows': len(valid),
        'valid_races': valid['race_id'].nunique() if 'race_id' in valid.columns else 0,
        'missing_rate': (len(df) - len(valid)) / len(df) * 100 if len(df) > 0 else 0,
        'min_odds': float(valid[col].min()) if len(valid) > 0 else None,
        'median_odds': float(valid[col].median()) if len(valid) > 0 else None,
        'max_odds': float(valid[col].max()) if len(valid) > 0 else None,
    }
    
    # Winner analysis
    if 'y_win' in df.columns:
        winners = df[(df['y_win'] == 1) & (df[col].notna()) & (df[col] > 0)]
        stats['winners_count'] = len(winners)
        
        if 'payout' in df.columns:
            winners_with_payout = winners[winners['payout'].notna() & (winners['payout'] > 0)]
            stats['winners_payout_available_rate'] = len(winners_with_payout) / len(winners) * 100 if len(winners) > 0 else 0
            
            if len(winners_with_payout) > 0:
                # Calculate diff = payout/bet_unit - odds
                implied_odds = winners_with_payout['payout'] / bet_unit
                odds_val = winners_with_payout[col]
                diff = implied_odds - odds_val
                abs_diff = diff.abs()
                
                stats['diff_min'] = float(diff.min())
                stats['diff_median'] = float(diff.median())
                stats['diff_mean'] = float(diff.mean())
                stats['diff_p95'] = float(diff.quantile(0.95))
                stats['diff_max'] = float(diff.max())
                stats['zero_rate'] = float((abs_diff < 0.01).mean())  # |diff| < 0.01 as "zero"
                stats['exact_match_rate'] = float((abs_diff < 0.001).mean())
                
                # Determine odds timing
                if stats['zero_rate'] > 0.95:
                    stats['timing_estimate'] = 'FINAL (確定オッズ)'
                    stats['timing_confidence'] = 'HIGH'
                    stats['warning'] = 'payout/bet = odds 一致率95%超 → 最終確定オッズと推定'
                elif stats['zero_rate'] > 0.80:
                    stats['timing_estimate'] = 'CLOSE (締切直前)'
                    stats['timing_confidence'] = 'MEDIUM'
                    stats['warning'] = None
                else:
                    stats['timing_estimate'] = 'PRE_CLOSE (締切前)'
                    stats['timing_confidence'] = 'LOW-MEDIUM'
                    stats['warning'] = None
    
    return stats


def generate_sample_races(
    df: pd.DataFrame,
    odds_col: str,
    n_races: int = 20,
    seed: int = 42,
    bet_unit: float = 100.0
) -> str:
    """Generate sample races table"""
    np.random.seed(seed)
    
    # Get races with winners
    if 'y_win' in df.columns:
        winner_races = df[df['y_win'] == 1]['race_id'].unique()
    else:
        return "(No y_win column - cannot identify winners)\n"
    
    sample_races = np.random.choice(winner_races, min(n_races, len(winner_races)), replace=False)
    
    lines = []
    lines.append("## Sample Races (Winner Analysis)\n\n")
    lines.append(f"**Sampled**: {len(sample_races)} races (seed={seed})\n\n")
    
    lines.append("| Race ID | Winner Horse | Odds | Payout/100 | Diff | Top3 Favorites |\n")
    lines.append("|---------|--------------|------|------------|------|----------------|\n")
    
    for race_id in sample_races[:20]:
        race_df = df[df['race_id'] == race_id].copy()
        
        # Winner
        winner = race_df[race_df['y_win'] == 1]
        if len(winner) == 0:
            continue
        
        w = winner.iloc[0]
        w_horse = w.get('horse_id', '?')
        w_odds = w.get(odds_col, 0)
        w_payout = w.get('payout', 0)
        w_implied = w_payout / bet_unit if w_payout > 0 else 0
        w_diff = w_implied - w_odds if w_odds > 0 else 0
        
        # Top 3 favorites (lowest odds)
        race_df_valid = race_df[race_df[odds_col].notna() & (race_df[odds_col] > 0)]
        top3 = race_df_valid.nsmallest(3, odds_col)
        top3_str = ', '.join([f"{row[odds_col]:.1f}" for _, row in top3.iterrows()])
        
        diff_str = f"{w_diff:.4f}" if abs(w_diff) > 0.001 else "0"
        lines.append(f"| {race_id} | {w_horse} | {w_odds:.1f} | {w_implied:.1f} | {diff_str} | {top3_str} |\n")
    
    return "".join(lines)


def generate_audit_report(
    df: pd.DataFrame,
    odds_cols: List[str],
    output_path: str,
    bet_unit: float = 100.0
):
    """Generate comprehensive odds timestamp audit report"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data
    df = prepare_data(df, bet_unit)
    
    report = f"""# Odds Timestamp Audit Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Input Rows**: {len(df):,}
**Unique Races**: {df['race_id'].nunique():,}
**Winners (y_win=1)**: {(df['y_win'] == 1).sum():,}

"""
    
    # Analyze each odds column
    all_stats = []
    for col in odds_cols:
        stats = analyze_odds_column(df, col, bet_unit)
        all_stats.append(stats)
    
    # Summary conclusion
    report += "## 結論\n\n"
    
    for stats in all_stats:
        timing = stats.get('timing_estimate', 'UNKNOWN')
        confidence = stats.get('timing_confidence', 'N/A')
        zero_rate = stats.get('zero_rate', 0)
        warning = stats.get('warning', '')
        
        report += f"**`{stats['column']}`**:\n"
        report += f"- 推定: **{timing}** (確信度: {confidence})\n"
        report += f"- payout/bet = odds 一致率: **{zero_rate:.1%}**\n"
        if warning:
            report += f"- ⚠️ {warning}\n"
        report += "\n"
    
    # Odds columns overview
    report += "## Detected Odds Columns\n\n"
    report += "| Column | Valid Rows | Races | Missing% | Min | Median | Max |\n"
    report += "|--------|------------|-------|----------|-----|--------|-----|\n"
    
    for stats in all_stats:
        report += f"| {stats['column']} | {stats['valid_rows']:,} | {stats['valid_races']:,} | {stats['missing_rate']:.1f}% | {stats['min_odds']:.1f} | {stats['median_odds']:.1f} | {stats['max_odds']:.1f} |\n"
    
    # Winner payout match analysis
    report += "\n## Winner Payout Match Analysis\n\n"
    report += "勝ち馬の `payout/100` と `odds` の一致を分析\n\n"
    report += "| Column | Winners | Payout Available | Zero Rate | Exact Match | Mean Diff |\n"
    report += "|--------|---------|------------------|-----------|-------------|----------|\n"
    
    for stats in all_stats:
        winners = stats.get('winners_count', 'N/A')
        payout_rate = stats.get('winners_payout_available_rate', 0)
        zero_rate = stats.get('zero_rate', 0)
        exact = stats.get('exact_match_rate', 0)
        mean_diff = stats.get('diff_mean', 0)
        
        report += f"| {stats['column']} | {winners:,} | {payout_rate:.1f}% | **{zero_rate:.1%}** | {exact:.1%} | {mean_diff:.4f} |\n"
    
    # Diff distribution
    report += "\n## Diff Distribution (payout/bet - odds)\n\n"
    report += "| Column | Min | Median | Mean | P95 | Max |\n"
    report += "|--------|-----|--------|------|-----|-----|\n"
    
    for stats in all_stats:
        if 'diff_min' in stats:
            report += f"| {stats['column']} | {stats['diff_min']:.4f} | {stats['diff_median']:.4f} | {stats['diff_mean']:.4f} | {stats['diff_p95']:.4f} | {stats['diff_max']:.4f} |\n"
        else:
            report += f"| {stats['column']} | N/A | N/A | N/A | N/A | N/A |\n"
    
    # Interpretation guide
    report += """\n## Interpretation

### Zero Rate (payout/bet = odds 一致率) の意味

| Zero Rate | 解釈 | 推奨アクション |
|-----------|------|---------------|
| >95% | **最終確定オッズ** (リーク疑い要注意) | `--allow_final_odds` + `--slippage_factor 0.90` |
| 80-95% | 締切直前オッズ | 標準使用可、slippage考慮推奨 |
| <80% | 締切前オッズ | `--odds_source pre_close` で使用 |

### Phase5/7での使用方法

"""
    
    for stats in all_stats:
        timing = stats.get('timing_estimate', 'UNKNOWN')
        if 'FINAL' in timing:
            report += f"""```bash
# {stats['column']} は最終確定オッズ → slippage推奨
python src/phase5/prob_sweep_roi.py \\
  --odds_source final \\
  --allow_final_odds \\
  --slippage_factor 0.90
```
"""
    
    # Sample races
    report += "\n"
    report += generate_sample_races(df, odds_cols[0], n_races=20, bet_unit=bet_unit)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Audit report saved to {output_path}")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Odds Timestamp Audit")
    parser.add_argument('--input', type=str, default='data/predictions/v13_market_residual_oof.parquet')
    parser.add_argument('--output', type=str, default='reports/diagnostics/odds_timestamp_audit.md')
    parser.add_argument('--bet_unit', type=float, default=100.0)
    parser.add_argument('--jra_only', action='store_true', default=False)
    
    args = parser.parse_args()
    
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    # JRA filter if needed
    if args.jra_only:
        from utils.race_filter import filter_jra_only
        df = filter_jra_only(df)
    
    logger.info(f"Data: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Detect odds columns
    odds_cols = detect_odds_columns(df)
    logger.info(f"Detected odds columns: {odds_cols}")
    
    if not odds_cols:
        logger.warning("No odds columns found!")
        return
    
    # Generate report
    stats = generate_audit_report(df, odds_cols, args.output, args.bet_unit)
    
    # Print summary
    for s in stats:
        logger.info(f"  {s['column']}: Zero Rate = {s.get('zero_rate', 0):.1%}, Timing = {s.get('timing_estimate', 'UNKNOWN')}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

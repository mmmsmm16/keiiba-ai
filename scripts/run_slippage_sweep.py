"""
Slippage Sweep for Phase 5 Win Optimization
複数のslippage_factorでバックテストを実行し、保守的評価を行う

Usage:
    docker compose exec app python scripts/run_slippage_sweep.py
    docker compose exec app python scripts/run_slippage_sweep.py --slippages 1.0,0.95,0.90
"""

import subprocess
import argparse
import os
from datetime import datetime


def run_sweep(
    base_cmd: str,
    slippages: list,
    output_dir: str = 'reports/phase5',
    prob_col: str = 'prob_residual_softmax'
):
    """Run backtest with multiple slippage factors and generate summary"""
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for slippage in slippages:
        print(f"\n=== Running with slippage_factor={slippage} ===")
        
        # Build command
        cmd = f"{base_cmd} --slippage_factor {slippage} --allow_final_odds"
        report_out = os.path.join(output_dir, f"phase5_v13_softmax_slip{int(slippage*100)}.md")
        cmd += f" --report_out {report_out}"
        
        # Run
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"SUCCESS: slippage={slippage}")
            results.append({
                'slippage': slippage,
                'status': 'success',
                'report': report_out,
                'output': result.stdout
            })
        else:
            print(f"FAILED: slippage={slippage}")
            print(result.stderr)
            results.append({
                'slippage': slippage,
                'status': 'failed',
                'error': result.stderr
            })
    
    # Generate summary report
    generate_summary(results, slippages, output_dir, prob_col)
    
    return results


def extract_roi_from_output(output: str) -> dict:
    """Extract ROI and other metrics from command output"""
    metrics = {}
    
    for line in output.split('\n'):
        if 'ROI=' in line:
            parts = line.split(',')
            for part in parts:
                if 'ROI=' in part:
                    try:
                        roi = float(part.split('=')[1].replace('%', '').strip())
                        metrics['roi'] = roi
                    except:
                        pass
                elif 'Hits=' in part:
                    try:
                        hits = float(part.split('=')[1].replace('%', '').strip())
                        metrics['hit_rate'] = hits
                    except:
                        pass
                elif 'MaxDD=' in part:
                    try:
                        maxdd = float(part.split('=')[1].replace('¥', '').replace(',', '').strip())
                        metrics['max_dd'] = maxdd
                    except:
                        pass
    
    return metrics


def generate_summary(results: list, slippages: list, output_dir: str, prob_col: str):
    """Generate slippage sweep summary report"""
    
    summary_path = os.path.join(output_dir, 'phase5_v13_softmax_slippage_sweep.md')
    
    report = f"""# Phase 5: Slippage Sweep Summary

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Prob Column**: `{prob_col}`
**Purpose**: Evaluate ROI sensitivity to odds slippage (conservative evaluation)

## Results

| Slippage Factor | Expected Odds | Status | ROI | Hit% | MaxDD |
|-----------------|---------------|--------|-----|------|-------|
"""
    
    for r in results:
        slip = r['slippage']
        status = r['status']
        
        if status == 'success':
            metrics = extract_roi_from_output(r.get('output', ''))
            roi = metrics.get('roi', 'N/A')
            hit = metrics.get('hit_rate', 'N/A')
            maxdd = metrics.get('max_dd', 'N/A')
            
            roi_str = f"{roi:.1f}%" if isinstance(roi, (int, float)) else roi
            hit_str = f"{hit:.1f}%" if isinstance(hit, (int, float)) else hit
            maxdd_str = f"¥{maxdd:,.0f}" if isinstance(maxdd, (int, float)) else maxdd
            
            report += f"| {slip} | ×{slip} | ✅ | {roi_str} | {hit_str} | {maxdd_str} |\n"
        else:
            report += f"| {slip} | ×{slip} | ❌ | - | - | - |\n"
    
    report += """
## Interpretation

| Slippage | Meaning |
|----------|---------|
| 1.00 | 最終オッズそのまま（楽観シナリオ） |
| 0.95 | 5%悪いオッズで購入できた想定（標準保守） |
| 0.90 | 10%悪いオッズで購入できた想定（保守的） |

## Recommendation

"""
    
    # Find successful results
    success_results = [r for r in results if r['status'] == 'success']
    
    if success_results:
        # Check if conservative scenario still profitable
        conservative = [r for r in success_results if r['slippage'] <= 0.95]
        if conservative:
            c = conservative[-1]  # Most conservative
            metrics = extract_roi_from_output(c.get('output', ''))
            roi = metrics.get('roi', 0)
            
            if roi > 100:
                report += f"- **保守シナリオ (slippage={c['slippage']}) でもROI {roi:.1f}%** → 戦略は堅牢\n"
            elif roi > 0:
                report += f"- **保守シナリオで微益 (ROI {roi:.1f}%)** → 実運用は慎重に\n"
            else:
                report += f"- **保守シナリオで損失** → 実運用非推奨\n"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run slippage sweep for Phase 5")
    parser.add_argument('--slippages', type=str, default='1.0,0.95,0.90',
                        help='Comma-separated slippage factors')
    parser.add_argument('--input', type=str, 
                        default='data/predictions/v13_market_residual_oof.parquet')
    parser.add_argument('--prob_col', type=str, default='prob_residual_softmax')
    parser.add_argument('--output_dir', type=str, default='reports/phase5')
    
    args = parser.parse_args()
    
    # Parse slippages
    slippages = [float(s.strip()) for s in args.slippages.split(',')]
    
    # Build base command
    base_cmd = (
        f"python src/phase5/prob_sweep_roi.py "
        f"--input {args.input} "
        f"--prob_cols {args.prob_col} "
        f"--odds_source final "
    )
    
    # Run sweep
    run_sweep(base_cmd, slippages, args.output_dir, args.prob_col)


if __name__ == "__main__":
    main()

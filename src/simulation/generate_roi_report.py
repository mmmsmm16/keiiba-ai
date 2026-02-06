import pandas as pd

def generate_report(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    df = df.sort_values('roi', ascending=False)
    
    # Format for markdown
    df['roi'] = (df['roi'] * 100).round(2).astype(str) + '%'
    df['hit_rate'] = (df['hit_rate'] * 100).round(2).astype(str) + '%'
    df['max_drawdown'] = df['max_drawdown'].round(0).astype(int)
    
    md_table = df[['strategy', 'num_bets', 'roi', 'hit_rate', 'max_drawdown']].to_markdown(index=False)
    
    report = f"""# ROI Simulation Report (v13_e1)

## Strategy Performance (Rolling Validation 2022-2024)

{md_table}

## Key Observations
1. **Best Strategy**: `Threshold_P0.4` (EV > 1.1) achieved the highest ROI (0.81), suggesting that high-confidence predictions are more robust.
2. **Win vs Place**: Place bets generally showed higher hit rates (20-30%) compared to Win bets (5-7%), but ROI remained in the 0.73-0.75 range.
3. **Kelly Criterion**: Fixed fraction Kelly showed significant drawdown and lower ROI, likely due to probability over-estimation.

## Next Steps
- Consider **Isotonic Regression** (Phase G) to see if it improves calibration further.
- Implement **Skip Rules** (Phase H) based on confidence metrics to prune low-ROI bets and push ROI over 100%.
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    generate_report("reports/simulations/results/v13_e1_summary.csv", "reports/simulations/v13_e1_roi_report.md")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

REPORT_DIR = "reports/jra/wf_incremental"
MODES = ['full', 'odds_only', 'no_odds', 'residual']

import glob

def load_results():
    results = {}
    month_dir = os.path.join(REPORT_DIR, "monthly")
    
    for mode in MODES:
        # 1. Try single file first
        path = f"{REPORT_DIR}/results_{mode}.parquet"
        if mode == 'full' and not os.path.exists(path):
            path = f"{REPORT_DIR}/results_pipeline.parquet"
            
        if os.path.exists(path):
            print(f"Loading {mode} from {path}")
            df = pd.read_parquet(path)
        else:
            # 2. Try loading monthly chunks
            pattern = os.path.join(month_dir, f"results_{mode}_2025-*.parquet")
            files = glob.glob(pattern)
            if files:
                print(f"Loading {mode} from {len(files)} chunks in {month_dir}")
                dfs = [pd.read_parquet(f) for f in files]
                df = pd.concat(dfs, ignore_index=True)
            else:
                print(f"Warning: {mode} results not found at {path} or chunks")
                continue

        # Ensure EV column
        if 'ev' not in df.columns:
            if 'calib_prob' in df.columns and 'odds' in df.columns:
                df['ev'] = df['calib_prob'] * df['odds']
            elif 'pred_prob' in df.columns and 'odds' in df.columns:
                 # Fallback for old pipeline format which might use 'pred' as raw score
                 # But in standard pipeline 'pred' is raw score.
                 # Let's assume calib_prob is essential for EV.
                 pass
            else:
                print(f"Warning: {mode} missing calib_prob or odds")
        results[mode] = df
            
    return results

def analyze_monthly(df, mode):
    # Assuming 'date' or 'race_id' can determine month.
    # race_id in JRA is year dependent? Yes 2025...
    # But result df might simply have monthly chunks appended.
    # We don't have date column in minimal result df?
    # run_jra_pipeline_backtest saves: race_id, horse_number, pred, odds, rank
    # Actually we need date or at least month to aggregate.
    # The script aggregates monthly but saves 'all_res' without date column explicitly?
    # Wait, 'test_df' has 'date' column?
    # In run_pipeline: `keep_cols = ['race_id', 'horse_number', 'pred', 'odds', 'rank']`
    # Warning: Date is missing!
    # But race_id structure: YYYY... ? No, JRA race_id is 2025...
    # We can parse race_id: 202545010101 -> 2025 (Year) ... Month is not obvious.
    # But we processed sequentially.
    # If we want monthly stats, we should have saved date.
    # Assuming row order is chronological.
    
    # Just Global Stats for now if date missing
    stats = []
    
    # Betting Sweep
    for th in [1.0, 1.1, 1.2, 1.5, 2.0]:
        target = df[df['ev'] >= th]
        n_bets = len(target)
        hits = target[target['rank'] == 1]
        n_hits = len(hits)
        
        ret = (hits['odds'] * 100).sum()
        inv = n_bets * 100
        profit = ret - inv
        roi = ret / inv * 100 if inv > 0 else 0.0
        hit_rate = n_hits / n_bets * 100 if n_bets > 0 else 0.0
        
        stats.append({
            'mode': mode,
            'threshold': th,
            'bets': n_bets,
            'hits': n_hits,
            'hit_rate': hit_rate,
            'roi': roi,
            'profit': profit
        })
    return pd.DataFrame(stats)

def main():
    results = load_results()
    all_stats = []
    
    print("\n=== Analysis Report ===")
    for mode, df in results.items():
        stats = analyze_monthly(df, mode)
        all_stats.append(stats)
        print(f"\nMode: {mode}")
        print(stats.to_markdown(index=False, floatfmt=".2f"))
        
    if all_stats:
        final_df = pd.concat(all_stats, ignore_index=True)
        final_df.to_csv(f"{REPORT_DIR}/phase20_sweep_comparison.csv", index=False)
        print(f"\nSaved comparison to {REPORT_DIR}/phase20_sweep_comparison.csv")

def plot_comparison(df, filename="phase20_comparison.png"):
    plt.figure(figsize=(12, 6))
    
    markers = {'full': 'o', 'odds_only': 's', 'no_odds': '^', 'residual': 'D'}
    colors = {'full': 'black', 'odds_only': 'blue', 'no_odds': 'green', 'residual': 'red'}
    
    for mode in df['mode'].unique():
        data = df[df['mode'] == mode]
        plt.plot(data['bets'], data['roi'], marker=markers.get(mode, 'o'), 
                 color=colors.get(mode, 'gray'), label=mode, linestyle='--')
        
        # Annotate thresholds
        for _, row in data.iterrows():
            plt.annotate(f"{row['threshold']}", (row['bets'], row['roi']), 
                         textcoords="offset points", xytext=(0,5), ha='center')

    plt.axhline(100, color='red', linestyle=':', alpha=0.5, label='Break-even')
    plt.xlabel('Number of Bets')
    plt.ylabel('ROI (%)')
    plt.title('Phase 20: ROI vs Bets Trade-off by Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, filename))
    print(f"Saved plot to {os.path.join(REPORT_DIR, filename)}")

if __name__ == "__main__":
    import traceback
    try:
        main()
        # Re-load the saved CSV to plot (or pass dataframe from main but main structure is rigid now)
        csv_path = f"{REPORT_DIR}/phase20_sweep_comparison.csv"
        if os.path.exists(csv_path):
             df = pd.read_csv(csv_path)
             plot_comparison(df)
    except Exception:
        traceback.print_exc()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


REPORT_DIR = "reports/jra/wf_incremental"
MODES = ['odds_only', 'residual', 'residual_opt', 'residual_direct']

import glob

def load_results():
    results = {}
    month_dir = os.path.join(REPORT_DIR, "monthly")
    
    for mode in MODES:
        # Load logic same as old script but focused on monthly chunks
        pattern = os.path.join(month_dir, f"results_{mode}_2025-*.parquet")
        files = glob.glob(pattern)
        if files:
            print(f"Loading {mode} from {len(files)} chunks in {month_dir}")
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
            results[mode] = df
        else:
            print(f"Warning: {mode} results not found at {pattern}")
            
    return results

def calculate_metrics(df, threshold=1.0):
    if 'ev' not in df.columns:
        if 'calib_prob' in df.columns and 'odds' in df.columns:
            df['ev'] = df['calib_prob'] * df['odds']
        else:
            return None

    target = df[df['ev'] >= threshold]
    n_bets = len(target)
    if n_bets == 0:
        return {'bets': 0, 'roi': 0, 'profit': 0, 'hit_rate': 0}
        
    hits = target[target['rank'] == 1]
    ret = (hits['odds'] * 100).sum()
    inv = n_bets * 100
    profit = ret - inv
    roi = ret / inv * 100
    hit_rate = len(hits) / n_bets * 100
    
    return {'bets': n_bets, 'roi': roi, 'profit': profit, 'hit_rate': hit_rate}

def analyze_alpha(results):
    if 'odds_only' not in results:
        return
        
    base_metrics = calculate_metrics(results['odds_only'])
    print("\n=== Alpha Analysis (vs Odds-Only) ===")
    print(f"Base Odds-Only ROI: {base_metrics['roi']:.2f}% (Bets: {base_metrics['bets']})")
    
    rows = []
    for mode in results:
        if mode == 'odds_only': continue
        m = calculate_metrics(results[mode])
        alpha = m['roi'] - base_metrics['roi']
        print(f"Mode: {mode:15s} ROI: {m['roi']:.2f}% (Alpha: {alpha:+.2f}%) Bets: {m['bets']}")
        rows.append({'mode': mode, 'roi': m['roi'], 'alpha': alpha, 'bets': m['bets']})
        
    return pd.DataFrame(rows)

def plot_frontier(results, filename="phase21_frontier.png"):
    plt.figure(figsize=(12, 8))
    
    # Sweep EV thresholds
    thresholds = np.arange(0.8, 2.5, 0.1)
    
    colors = {'odds_only': 'gray', 'residual': 'black', 'residual_opt': 'blue', 'residual_direct': 'red'}
    
    for mode, df in results.items():
        bets = []
        rois = []
        labels = []
        
        # Ensure EV
        if 'ev' not in df.columns:
            if 'calib_prob' in df.columns: df['ev'] = df['calib_prob'] * df['odds']
            else: continue
            
        for th in thresholds:
            m = calculate_metrics(df, th)
            if m and m['bets'] > 10: # Minimum sample
                bets.append(m['bets'])
                rois.append(m['roi'])
                labels.append(f"{th:.1f}")
                
        if bets:
            plt.plot(bets, rois, marker='o', label=mode, color=colors.get(mode, 'gray'))
            # Annotate only some points
            for i, (b, r, l) in enumerate(zip(bets, rois, labels)):
                if i % 2 == 0:
                    plt.annotate(l, (b, r), xytext=(0, 5), textcoords='offset points', fontsize=8)

    plt.axhline(100, color='green', linestyle=':', label='Break-even')
    plt.xlabel('Number of Bets')
    plt.ylabel('ROI (%)')
    plt.title('Phase 21: Efficiency Frontier (EV Threshold Sweep)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, filename))
    print(f"Saved frontier plot to {filename}")

def main():
    results = load_results()
    if not results:
        print("No results loaded.")
        return

    # 1. Standard Comparison (EV >= 1.0)
    print("\n=== Standard Comparison (EV >= 1.0) ===")
    summary = []
    for mode in results:
        m = calculate_metrics(results[mode], threshold=1.0)
        m['mode'] = mode
        summary.append(m)
    
    sum_df = pd.DataFrame(summary)
    print(sum_df.to_markdown(index=False, floatfmt=".2f"))
    
    # 2. Alpha Analysis
    analyze_alpha(results)
    
    # 3. Frontier Plot
    plot_frontier(results)
    
    # Save Summary
    sum_df.to_csv(f"{REPORT_DIR}/phase21_summary.csv", index=False)

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()


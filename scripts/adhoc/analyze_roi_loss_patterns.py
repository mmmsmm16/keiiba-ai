
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_odds_bands(df, ticket_type, odds_col, score_col, p_th, ev_th):
    print(f"--- Analysis: {ticket_type} (p>={p_th}, ev>={ev_th}) ---")
    
    # Filter by Rule
    bets = df[
        (df[score_col] >= p_th) &
        (df[score_col] * df[odds_col] >= ev_th)
    ].copy()
    
    if bets.empty:
        print("No bets found.")
        return

    # Define Bands
    match ticket_type:
        case "Win":
            bins = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 999.0]
            labels = ["1-3", "3-5", "5-10", "10-20", "20-50", "50-100", "100+"]
        case "Umaren":
             bins = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 999.0]
             labels = ["1-5", "5-10", "10-20", "20-50", "50-100", "100+"]
        case "Place":
             bins = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 999.0]
             labels = ["1-1.5", "1.5-2", "2-3", "3-5", "5-10", "10+"]
    
    bets['band'] = pd.cut(bets[odds_col], bins=bins, labels=labels, right=False)
    
    summary = bets.groupby('band', observed=False).agg(
        Count=('date', 'count'),
        Return=('win_payoff', 'sum'),
        Cost=('win_payoff', lambda x: len(x) * 100),
        AvgProb=(score_col, 'mean'),
        AvgOdds=(odds_col, 'mean')
    )
    
    summary['ROI'] = summary['Return'] / summary['Cost']
    summary['HitRate'] = (summary['Return'] > 0) / summary['Count'] # Strict hit rate approximation (payout > 0)
    
    print(summary[['Count', 'ROI', 'HitRate', 'AvgProb', 'AvgOdds']])
    print("\n")

def main():
    DATA_PATH = "reports/simulations/v24_m5_roi_dataset_2022_2024.parquet"
    if not os.path.exists(DATA_PATH): return
    
    df = pd.read_parquet(DATA_PATH)
    test_df = df[df['year'] == 2024].copy()
    
    # Simulate Unshrunk + Odds Cap
    # Rule: p_win >= 0.10, ev >= 2.0, win_odds < 20.0
    
    # 1. Calculate Unshrunk EV
    test_df['ev_win'] = test_df['p_win'] * test_df['win_odds']
    
    # 2. Filter with Cap
    print("--- Analysis: Win (p>=0.1, ev>=2.0, Odds < 20.0) ---")
    bets = test_df[
        (test_df['p_win'] >= 0.10) &
        (test_df['ev_win'] >= 2.0) &
        (test_df['win_odds'] < 20.0)
    ].copy()
    
    # 3. Analyze Buckets
    analyze_odds_bands(test_df, "Win", "win_odds", "p_win", 0.10, 2.0)
    
    # 4. Summary of Capped Strategy
    if not bets.empty:
        count = len(bets)
        ret = bets['win_payoff'].sum()
        cost = count * 100
        roi = ret / cost
        print(f"\nTotal Capped Strategy (Odds<20): Count={count}, Cost={cost}, Return={ret}, ROI={roi:.3f}")
    else:
        print("No bets found.")

if __name__ == "__main__":
    main()

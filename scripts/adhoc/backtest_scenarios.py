import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Matplotlib Japanese setting (Optional, falling back to English labels if needed)
# plt.rcParams['font.family'] = 'Meiryo'

def backtest_scenarios(csv_path):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # 1. Preprocessing
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'race_id'])
    
    # Filter only WIN bets
    df = df[df['type'] == 'WIN'].copy()
    
    # Extract Venue
    def extract_venue(rid):
        rid_str = str(rid)
        if len(rid_str) == 12: # Standard 12 digit (YYYYMMDDPPHH) -> PP is index 8,9? No: 202501050601 -> 06 is index 8,9
            # JRA-VAN/PC-KEIBA format often: YYYYMMDDjjRR (12 digits) -> jj=venue
            # Wait, simulate_kelly writes race_id.
            # Example: 202501050601
            # 0-3: 2025
            # 4-7: 0105 (MMDD) ?? No usually YYYY is 4 chars.
            # Let's assume standard format: YYYY(4) + Venue(2) + ...
            # Actually simplest check is the `venue_code` generated in previous script or just try parsing.
            # 202501050601 -> 06 is venue? 
            # 2025(4) + 01(2) + 05(2) + 06(2) + 01(2) => 12 digits.
            return rid_str[8:10] # jj
        return "Unknown"

    VENUE_MAP = {
        '01': 'Sapporo', '02': 'Hakodate', '03': 'Fukushima', '04': 'Niigata', '05': 'Tokyo', 
        '06': 'Nakayama', '07': 'Chukyo', '08': 'Kyoto', '09': 'Hanshin', '10': 'Kokura'
    }

    df['venue_code'] = df['race_id'].astype(str).str[8:10] # Correct index for YYYYMMDDjjRR format?
    # Wait, usually YYYYMMDDjjRR.
    # 2025 01 05 06 01 -> 12 chars.
    # 0123 45 67 89 01
    # YYYY MM DD jj RR
    # So index 8:10 is venue.
    
    df['venue_name'] = df['venue_code'].map(VENUE_MAP).fillna('Unknown')
    
    # Define Filter Functions
    def filter_scenario(row, scenario):
        # Baseline
        if scenario == 'Baseline':
            return True
            
        # Plan A: Odds >= 5.0, Prob >= 0.08
        if scenario == 'Plan A':
            if row['odds'] < 5.0: return False
            if row['prob'] < 0.08: return False
            return True
            
        # Plan B: Plan A + Exclude Niigata(04), Fukushima(03), Hanshin(09), Kokura(10)
        if scenario == 'Plan B':
            # Base Plan A
            if row['odds'] < 5.0: return False
            if row['prob'] < 0.08: return False
            
            # Additional Venue Filter
            # Exclude 03, 04, 09, 10
            if row['venue_code'] in ['03', '04', '09', '10']: return False
            return True
        
        # Plan C: Sweet Spot (Odds 10.0 - 50.0, Min Prob 0.03)
        if scenario == 'Plan C':
            if row['odds'] < 10.0: return False
            if row['odds'] > 50.0: return False
            if row['prob'] < 0.03: return False
            return True
            
        return False

    # Simulate Flat Betting for each scenario
    scenarios = ['Baseline', 'Plan A', 'Plan B', 'Plan C']
    results = {}
    curves = pd.DataFrame(index=df['date'].unique()).sort_index() # For plot
    curves.index.name = 'date'

    bet_unit = 100

    print("\n--- Simulation Results (Flat Bet 100yen) ---")
    
    for sc in scenarios:
        # Apply filter
        mask = df.apply(lambda r: filter_scenario(r, sc), axis=1)
        sub_df = df[mask].copy()
        
        # Calculate stats
        sub_df['payout'] = sub_df.apply(lambda r: r['odds']*bet_unit if r['return'] > 0 else 0, axis=1)
        sub_df['profit'] = sub_df['payout'] - bet_unit
        
        # Aggregate
        total_bets = len(sub_df)
        hits = (sub_df['payout'] > 0).sum()
        hit_rate = (hits / total_bets * 100) if total_bets > 0 else 0
        total_profit = sub_df['profit'].sum()
        roi = ((total_profit + total_bets*bet_unit) / (total_bets*bet_unit) * 100) if total_bets > 0 else 0
        
        # Max Drawdown
        sub_df = sub_df.sort_values('date')
        if not sub_df.empty:
            # Need strict daily cumsum
            # Group by date first to handle multiple bets per day
            daily_profit = sub_df.groupby('date')['profit'].sum()
            daily_cum = daily_profit.cumsum()
            
            # Calculate drawdown
            # Running Max
            running_max = daily_cum.cummax()
            drawdown = daily_cum - running_max
            max_dd = drawdown.min()
            
            # Store curve for visualization (reindexed to full dates)
            # We need full timeline
            curves[sc] = daily_profit
            
        else:
            max_dd = 0
            curves[sc] = 0

        print(f"[{sc}]")
        print(f"  Bets: {total_bets} | Hits: {hits}")
        print(f"  WinRate: {hit_rate:.1f}%")
        print(f"  ROI: {roi:.1f}%")
        print(f"  Profit: {int(total_profit)} JPY")
        print(f"  MaxDD: {int(max_dd)} JPY")
        print("-" * 30)

    # Visualization
    # Fill NA with 0 (days with no bets)
    curves = curves.fillna(0).cumsum()
    
    plt.figure(figsize=(12, 6))
    for sc in scenarios:
        plt.plot(curves.index, curves[sc], label=sc)
        
    plt.title('Cumulative Profit by Strategy (Flat Betting)')
    plt.xlabel('Date')
    plt.ylabel('Profit (JPY)')
    plt.grid(True)
    plt.legend()
    plt.axhline(0, color='black', linestyle='-')
    
    plot_path = os.path.join(os.path.dirname(csv_path), 'scenario_comparison.png')
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "reports/jra/daily/sim_kelly_2025.csv"
        
    backtest_scenarios(csv_path)

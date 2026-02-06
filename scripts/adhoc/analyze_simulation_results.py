import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set font for Japanese support if available, else standard
# plt.rcParams['font.family'] = 'Meiryo' 

def analyze_simulation_results(csv_path):
    print(f"Loading results from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # Basic Preprocessing
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df['ev'] = df['prob'] * df['odds']
    
    # Filter for WIN bets for generic analysis (unless UMA requested, but usually separate)
    # Task requires pure predictive power -> WIN is best proxy.
    df_win = df[df['type'] == 'WIN'].copy()
    
    if df_win.empty:
        print("No WIN bets found in the log.")
        return

    print(f"Loaded {len(df_win)} WIN bet records.")
    
    # --- Task A: Flat Betting Verification ---
    print("\n=== Task A: Flat Betting Verification (100 yen per bet) ===")
    
    # Logic: For every bet made by Kelly, simulate if we bet flat 100 yen.
    # Note: This limits analysis to "Bets that Kelly accepted". 
    # If Kelly rejected a bet (fraction <= 0), it's not here. 
    # But sim log only has accepted bets anyway.
    
    df_win['flat_bet'] = 100
    df_win['flat_return'] = df_win.apply(lambda row: 100 * row['odds'] if row['return'] > 0 else 0, axis=1)
    df_win['flat_profit'] = df_win['flat_return'] - df_win['flat_bet']
    
    total_flat_profit = df_win['flat_profit'].sum()
    total_flat_roi = (df_win['flat_return'].sum() / df_win['flat_bet'].sum()) * 100
    win_rate = (df_win['flat_return'] > 0).mean() * 100
    
    print(f"Total Bets: {len(df_win)}")
    print(f"Flat Profit: {total_flat_profit} JPY")
    print(f"Flat ROI:    {total_flat_roi:.2f}%")
    print(f"Win Rate:    {win_rate:.2f}%")
    
    # Monthly Flat ROI
    monthly_flat = df_win.groupby('month').agg({
        'flat_bet': 'sum',
        'flat_return': 'sum',
        'flat_profit': 'sum'
    })
    monthly_flat['roi'] = (monthly_flat['flat_return'] / monthly_flat['flat_bet']) * 100
    print("\n-- Monthly Flat Betting Performance --")
    print(monthly_flat[['flat_bet', 'flat_profit', 'roi']])
    
    # Visualization: Cumulative Profit
    df_win = df_win.sort_values('date')
    df_win['cum_flat_profit'] = df_win['flat_profit'].cumsum()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_win['date'], df_win['cum_flat_profit'], label='Flat Betting (100yen)')
    plt.title('Cumulative Profit (Flat Betting)')
    plt.xlabel('Date')
    plt.ylabel('Profit (JPY)')
    plt.grid(True)
    plt.legend()
    plt.savefig('reports/jra/daily/flat_bet_profit.png')
    print("Saved plot: reports/jra/daily/flat_bet_profit.png")
    
    # --- Task B: Odds Band Analysis ---
    print("\n=== Task B: Odds Band Analysis ===")
    
    bins = [1.0, 1.9, 4.9, 9.9, 19.9, 49.9, np.inf]
    labels = ['1.0-1.9', '2.0-4.9', '5.0-9.9', '10.0-19.9', '20.0-49.9', '50.0+']
    
    df_win['odds_band'] = pd.cut(df_win['odds'], bins=bins, labels=labels, right=True)
    
    band_stats = df_win.groupby('odds_band', observed=False).agg({
        'flat_bet': 'count',
        'flat_return': lambda x: (x > 0).sum(), # Hit count
        'flat_profit': 'sum',
        'prob': 'mean' # Avg Pred Prob
    }).rename(columns={'flat_bet': 'count', 'flat_return': 'hits'})
    
    band_stats['win_rate'] = (band_stats['hits'] / band_stats['count']) * 100
    band_stats['roi'] = ((band_stats['flat_profit'] + band_stats['count']*100) / (band_stats['count']*100)) * 100
    band_stats['avg_odds'] = df_win.groupby('odds_band', observed=False)['odds'].mean()
    
    # Expected Win Rate based on Model Prob
    band_stats['exp_win_rate'] = band_stats['prob'] * 100
    
    print(band_stats[['count', 'win_rate', 'exp_win_rate', 'roi']])
    
    # Visualization: ROI by Band
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=band_stats.index, y=band_stats['roi'])
    plt.axhline(100, color='red', linestyle='--')
    plt.title('ROI by Odds Band')
    plt.ylabel('ROI (%)')
    # Add count labels
    for i, p in enumerate(ax.patches):
        if i < len(band_stats):
            count = band_stats.iloc[i]['count']
            ax.annotate(f"n={count}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.savefig('reports/jra/daily/odds_band_roi.png')
    print("Saved plot: reports/jra/daily/odds_band_roi.png")

    # --- Task C: Venue Analysis ---
    print("\n=== Task C: Venue Analysis ===")
    # Extract Venue Code from Race ID (Assuming YYYYTTDDRR format? Or PC-KEIBA YYYYMMDDjjRR?)
    # Based on JRA Van standard: Key is YYYYMMDDPPHH
    # Usually RaceID in this system is `202501050601` -> 06 is venue?
    # Let's infer.
    
    # Try parsing
    # Typically: YYYY(4) + Venue(2) + Kai(2) + Day(2) + Race(2)
    # Venue is chars 4-6 (0-indexed: 4,5)
    
    def extract_venue(rid):
        rid_str = str(rid)
        if len(rid_str) == 12: # Standard 12 digit
            return rid_str[4:6] # Index 4,5
        return "Unknown"

    VENUE_MAP = {
        '01': 'Sapporo', '02': 'Hakodate', '03': 'Fukushima', '04': 'Niigata', '05': 'Tokyo', 
        '06': 'Nakayama', '07': 'Chukyo', '08': 'Kyoto', '09': 'Hanshin', '10': 'Kokura'
    }

    df_win['venue_code'] = df_win['race_id'].apply(extract_venue)
    df_win['venue_name'] = df_win['venue_code'].map(VENUE_MAP).fillna(df_win['venue_code'])
    
    venue_stats = df_win.groupby('venue_name').agg({
        'flat_bet': 'count',
        'flat_profit': 'sum'
    })
    venue_stats['roi'] = ((venue_stats['flat_profit'] + venue_stats['flat_bet']*100) / (venue_stats['flat_bet']*100)) * 100
    venue_stats = venue_stats.sort_values('roi', ascending=False)
    
    print(venue_stats)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=venue_stats.index, y=venue_stats['roi'])
    plt.axhline(100, color='red', linestyle='--')
    plt.title('ROI by Venue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/jra/daily/venue_roi.png')
    print("Saved plot: reports/jra/daily/venue_roi.png")
    
    # --- Conclusion/Bottleneck Identification (Placeholder) ---
    print("\n=== Analysis Summary ===")
    print("1. Flat Betting ROI vs Kelly ROI: Comparison to check if Kelly sizing is hurting.")
    print("2. Odds Band with lowest ROI: Identify if favorites (low odds) or longshots are underperforming.")
    print("   - If low odds ROI < 80%: Model is overconfident on favorites.")
    print("   - If high odds ROI < 80%: Model is hallucinating value in dark horses.")
    print("3. Problematic Venues: Check if specific tracks drag down performance.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "reports/jra/daily/sim_kelly_2025.csv"
    
    analyze_simulation_results(csv_path)

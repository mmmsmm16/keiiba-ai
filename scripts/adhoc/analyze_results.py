
import pandas as pd
import os

try:
    df = pd.read_csv('reports/backtest/backtest_2025_v23_results.csv')
    print("Columns:", df.columns.tolist())
    
    # Map columns if necessary
    # Assuming: bet, return
    if 'bet' in df.columns:
         total_cost = df['bet'].sum()
         total_return = df['return'].sum()
    elif 'bet_amount' in df.columns:
         total_cost = df['bet_amount'].sum()
         total_return = df['return_amount'].sum()
    else:
         print("Unknown columns")
         exit()
         
    total_races = df['race_id'].nunique()
    total_bets = len(df)
    
    
    profit = total_return - total_cost
    roi = (total_return / total_cost * 100) if total_cost > 0 else 0
    hit_rate = (df['return'] > 0).mean() * 100
    
    # Calculate per bet type stats (Not available in CSV)
    # print("\n=== Best Performing Strategies ===")
    
    print("\n=== Overall Results (2025 JRA) ===")
    print(f"Races: {total_races}")
    print(f"Bets: {total_bets}")
    print(f"Total Cost: {total_cost:,.0f}")
    print(f"Total Return: {total_return:,.0f}")
    print(f"Profit: {profit:,.0f}")
    print(f"ROI: {roi:.2f}%")
    print(f"Hit Rate: {hit_rate:.2f}%")

except Exception as e:
    print(f"Error reading results: {e}")

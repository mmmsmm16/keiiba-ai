import pandas as pd
try:
    df = pd.read_csv('reports/simulation_2025_autopredict.csv')
    total_bet = df['bet'].sum()
    total_ret = df['return'].sum()
    roi = total_ret / total_bet * 100 if total_bet > 0 else 0
    profit = total_ret - total_bet
    print(f"Total Bets: {len(df)}")
    print(f"Total Invest: {total_bet:,.0f} JPY")
    print(f"Total Return: {total_ret:,.0f} JPY")
    print(f"Total Profit: {profit:+,.0f} JPY")
    print(f"ROI: {roi:.2f}%")
except Exception as e:
    print(f"Error: {e}")

# Quick script to check odds structure
import pandas as pd

print("=== T-30 Odds Structure ===")
df30 = pd.read_parquet('data/odds_snapshots/2025/odds_T-30.parquet', filters=[('ticket_type', '=', 'win')])
print('Columns:', df30.columns.tolist())
print('Shape:', df30.shape)
print('\nSample:')
print(df30.head(3))

print("\n=== T-10 Odds Structure ===")
df10 = pd.read_parquet('data/odds_snapshots/2025/odds_T-10.parquet', filters=[('ticket_type', '=', 'win')])
print('Columns:', df10.columns.tolist())
print('Shape:', df10.shape)
print('\nSample:')
print(df10.head(3))

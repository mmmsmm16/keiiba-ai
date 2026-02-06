import pandas as pd
import pickle

# Parquet
df = pd.read_parquet('data/processed/preprocessed_data_v10_leakfix.parquet')
print('=== Parquet Data ===')
print(f'Shape: {df.shape}')
print(f'Date range: {df["date"].min()} ~ {df["date"].max()}')
print(f'Total columns: {len(df.columns)}')

# Pickle
with open('data/processed/lgbm_datasets_v10_leakfix.pkl', 'rb') as f:
    datasets = pickle.load(f)

print('\n=== LGBM Dataset ===')
print(f'Train X shape: {datasets["train"]["X"].shape}')
print(f'Train y shape: {datasets["train"]["y"].shape}')
print(f'Total features: {len(datasets["train"]["X"].columns)}')

print('\n=== First 50 Features ===')
for i, col in enumerate(datasets['train']['X'].columns[:50]):
    print(f'{i+1:3}. {col}')

print(f'\n... (Total {len(datasets["train"]["X"].columns)} features)')

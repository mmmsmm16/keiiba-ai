import pandas as pd

df = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
print(f'Total columns: {len(df.columns)}')

meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 'is_win', 'is_top2', 'is_top3', 'year']
feature_cols = [c for c in df.columns if c not in meta_cols]
print(f'Feature columns: {len(feature_cols)}')

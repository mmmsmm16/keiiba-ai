"""Debug script to analyze v11 data issues"""
import pandas as pd
import numpy as np

df = pd.read_parquet('/workspace/data/processed/preprocessed_data_v11.parquet')

# Check lag1_last_3f statistics
print('=== lag1_last_3f Statistics ===')
print(f'Total rows: {len(df):,}')
print(f'Zero count: {(df["lag1_last_3f"] == 0).sum():,}')
print(f'Zero rate: {(df["lag1_last_3f"] == 0).mean():.1%}')
print(f'NaN count: {df["lag1_last_3f"].isna().sum():,}')
print(f'Value at 35.0 (neutral): {(df["lag1_last_3f"] == 35.0).sum():,}')
print(f'Unique values (sample):')
print(df["lag1_last_3f"].value_counts().head(10))

# Check mean_last_3f_5
print('\n=== mean_last_3f_5 Statistics ===')
print(f'Zero rate: {(df["mean_last_3f_5"] == 0).mean():.1%}')
print(f'Value at 35.0: {(df["mean_last_3f_5"] == 35.0).sum():,}')
print(f'Unique values (sample):')
print(df["mean_last_3f_5"].value_counts().head(10))

# Check lag1_rank shift logic with a sample
print('\n=== Shift Validation Sample ===')
sample_horse = df.groupby('horse_id').size().idxmax()
horse_df = df[df['horse_id'] == sample_horse].sort_values(['date', 'race_id'])[['horse_id', 'date', 'race_id', 'rank', 'lag1_rank']].head(10)
print(horse_df.to_string())

# Check sire_id unknown inflation
print('\n=== sire_id Unknown Inflation ===')
print(f'sire_id value counts:')
print(df['sire_id'].value_counts().head(10))
print(f'\nUnknown rate: {(df["sire_id"].astype(str).str.lower() == "unknown").mean():.1%}')

# Check sire_id_n_races for unknown
unknown_mask = df['sire_id'].astype(str).str.lower() == 'unknown'
if unknown_mask.any():
    print(f'sire_id_n_races (unknown rows) max: {df.loc[unknown_mask, "sire_id_n_races"].max():.0f}')
    print(f'sire_id_n_races (normal rows) 99.9%: {df.loc[~unknown_mask, "sire_id_n_races"].quantile(0.999):.0f}')

# Check trend_* columns
print('\n=== trend_* Columns ===')
trend_cols = [c for c in df.columns if c.startswith('trend_')]
print(f'trend columns: {trend_cols}')
for tc in trend_cols:
    print(f'  {tc}: unique={df[tc].nunique()}, mean={df[tc].mean():.3f}')

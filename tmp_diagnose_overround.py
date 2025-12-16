"""Quick script to diagnose overround extreme values"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')
from utils.race_filter import filter_jra_only

# Load and filter
df = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
df['year'] = df['race_id'].astype(str).str[:4].astype(int)
df = df[df['year'] == 2024]
df = filter_jra_only(df)

# Calculate overround
df['raw_prob'] = 1.0 / df['odds'].replace(0, np.nan)
race_overround = df.groupby('race_id')['raw_prob'].sum().reset_index()
race_overround.columns = ['race_id', 'overround']

# Find extreme races
extreme = race_overround[race_overround['overround'] < 1.0]
print(f'Races with overround < 1.0: {len(extreme)}')

for _, row in extreme.head(5).iterrows():
    race_id = row['race_id']
    race_df = df[df['race_id'] == race_id]
    print(f"Race {race_id}: overround={row['overround']:.4f}, runners={len(race_df)}")
    print(f"  odds: {race_df['odds'].tolist()}")

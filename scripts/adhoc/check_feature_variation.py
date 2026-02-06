import sys
sys.path.insert(0, '/workspace')

import pandas as pd
import numpy as np

# キャッシュを確認
cache_file = '/workspace/data/features_t2/temp_merge_current.parquet'
df = pd.read_parquet(cache_file)

print(f"Cache total records: {len(df)}")
print(f"\n=== Date Range in Cache ===")
print(f"Min date: {df['date'].min()}")
print(f"Max date: {df['date'].max()}")

# 2026年のデータがあるか確認
df_2026 = df[df['date'].dt.year == 2026]
print(f"\n2026 records in cache: {len(df_2026)}")

# race_id の例を確認
print(f"\nSample race_ids (last 5):")
print(df['race_id'].tail(5).tolist())

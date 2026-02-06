import pandas as pd
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
print(list(df.columns))


import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')

# Filter for numeric
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# Calculate correlations with 'rank'
print("calculating correlations with rank...")
corrs = df[numeric_cols].corrwith(df['rank'])

# Show top correlated features (absolute value)
print("\nTop Correlated Features with Rank:")
print(corrs.abs().sort_values(ascending=False).head(30))

# Also check column names containing 'corner' or 'pace' or 'agari'
print("\nPotential Leak Columns (Name Check):")
leak_keywords = ['corner', 'pace', 'agari', 'pos', 'time', 'margin']
for c in df.columns:
    if any(k in c for k in leak_keywords):
        print(c)

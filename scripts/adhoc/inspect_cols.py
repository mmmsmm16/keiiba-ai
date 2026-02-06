
import pandas as pd
import os

try:
    df = pd.read_parquet("data/temp_q1/year_2023.parquet")
    cols = sorted(df.columns.tolist())
    print("Columns:", cols)
    
    # Check for keywords
    print("--- Keywords ---")
    for c in cols:
        if 'run' in c or 'leg' in c or 'style' in c or 'kyaku' in c or 'corner' in c:
            print(f"Found: {c}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")

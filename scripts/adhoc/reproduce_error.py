import pandas as pd
import numpy as np

# Mocking the dataframe content causing the issue
data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
df = pd.DataFrame(data)

print("=== Reproduction ===")
try:
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        # This is the line causing the error
        print(f"{i+1:3}. {col:35} ({dtype:15})")
except TypeError as e:
    print(f"Caught expected error: {e}")

print("\n=== Fix ===")
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    # This should be the fix
    print(f"{i+1:3}. {col:35} ({str(dtype):15})")

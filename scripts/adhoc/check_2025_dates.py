
import pandas as pd
import os

def main():
    path = "data/temp_t2/T2_predictions_2025_only.parquet"
    if not os.path.exists(path):
        print("File not found.")
        return
        
    df = pd.read_parquet(path)
    if 'date' not in df.columns:
        print("Date column missing.")
        return
        
    df['date'] = pd.to_datetime(df['date'])
    print(f"Min Date: {df['date'].min()}")
    print(f"Max Date: {df['date'].max()}")
    print("\nMonth Counts:")
    print(df['date'].dt.month.value_counts().sort_index())

if __name__ == "__main__":
    main()

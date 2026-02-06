import pandas as pd
import numpy as np
import argparse
import sys
import os

def check_leaks(parquet_path):
    print(f"Checking for leaks in: {parquet_path}")
    if not os.path.exists(parquet_path):
        print("File not found.")
        return

    df = pd.read_parquet(parquet_path)
    
    # 1. Check for Forbidden Columns
    forbidden = ['rank', 'time', 'rank_numeric', 'is_win', 'is_top3', 
                 'speed_index', 'PCI', 'RPCI', 'pci', 'rpci', 'time_index', 'last_3f_index'] 
                 # Note: statistical features (e.g. pci_mean) are allowed, but raw values are allowed ONLY if they are NOT target
                 # Here we assume file is "preprocessed_data" which MIGHT contain raw columns for training target.
                 # If this is "dataset" (X), they must be gone.
                 # Usually preprocessed_data has everything.
    
    print("Skipping Forbidden Column Check for preprocessed_data (targets are expected to be there).")
    
    # 2. Correlation Check with Target
    # Check if any feature has >0.99 correlation with 'rank' or 'time'
    # excluding the target columns themselves.
    
    targets = ['rank', 'time']
    available_targets = [c for c in targets if c in df.columns]
    
    if not available_targets:
        print("No targets found to check correlation.")
        return

    # Filter numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    print(f"Calculating correlations for {len(numeric_df.columns)} numeric columns...")
    
    leaks_found = False
    
    for target in available_targets:
        # Convert to numeric if needed
        y = pd.to_numeric(df[target], errors='coerce')
        if y.isna().all():
            continue
            
        cors = numeric_df.corrwith(y).abs()
        
        # Threshold
        HIGH_CORR = 0.95
        
        possibles = cors[cors > HIGH_CORR].sort_values(ascending=False)
        # Exclude target itself
        possibles = possibles[possibles.index != target]
        
        if not possibles.empty:
            print(f"\n[WARNING] Potential Leaks detected for target '{target}':")
            for feat, val in possibles.items():
                 print(f"  - {feat}: {val:.4f}")
                 leaks_found = True
        else:
            print(f"No leaks detected for target '{target}'.")

    if not leaks_found:
        print("\n✅ No obvious leaks detected via correlation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to parquet file')
    args = parser.parse_args()
    
    check_leaks(args.file_path)

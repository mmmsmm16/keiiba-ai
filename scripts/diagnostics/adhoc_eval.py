
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import os
import sys

# Need to load metadata for segments
# We can load one of the feature files to get race metadata
# We use the raw year file for 2024 as it definitely contains metadata
METADATA_SOURCE = "data/temp_m4/year_2024.parquet"

def load_metadata():
    if not os.path.exists(METADATA_SOURCE):
        print(f"Metadata source {METADATA_SOURCE} not found. Segmentation unavailable.")
        return None
    
    df = pd.read_parquet(METADATA_SOURCE)
    
    # We need race_id, distance, n_horses
    # Check columns
    cols = df.columns
    req = ['race_id']
    if 'distance' in cols: req.append('distance')
    else: print("Warnings: distance not found in raw data")
    
    if 'n_horses' in cols: req.append('n_horses')
    else:
        # Try to calculate if missing (raw might not have n_horses if it's jra-van raw)
        # JRA-VAN raw usually has 'toso' -> n_horses? Or we count rows.
        print("Warnings: n_horses not found, calculating from group size")
        # will handle below
    
    meta = df[req].copy()
    
    # If n_horses missing, calculate
    if 'n_horses' not in meta.columns:
        meta['n_horses'] = meta.groupby('race_id')['race_id'].transform('count')
        
    # Deduplicate: One row per race
    meta = meta.drop_duplicates('race_id')
    return meta

def compute_metrics(df, label="All"):
    recall_5_list = []    # Standard Recall@5
    race_hit_5_list = []  # Race-Hit@5 (Any Hit)
    precision_5_list = [] # Precision@5
    ndcg_5_list = []
    
    race_groups = df.groupby('race_id')
    
    for race_id, group in race_groups:
        if len(group) == 0: continue
        
        # Sort by predicted score descending
        group_sorted = group.sort_values('y_pred', ascending=False)
        
        top5 = group_sorted.head(5)
        n_relevant = group['y_true'].astype(bool).sum()
        if n_relevant == 0:
            continue
            
        n_found = top5['y_true'].astype(bool).sum()
        
        # Standard Recall: Coverage of relevant items
        recall_5 = n_found / n_relevant
        recall_5_list.append(recall_5)
        
        # Race-Hit: Did we find AT LEAST ONE?
        race_hit_5 = 1.0 if n_found > 0 else 0.0
        race_hit_5_list.append(race_hit_5)
        
        # Precision: Density of relevant items in Top 5
        precision_5 = n_found / 5.0
        precision_5_list.append(precision_5)
        
        # NDCG@5
        y_true = group['y_true'].values.reshape(1, -1)
        y_score = group['y_pred'].values.reshape(1, -1)
        try:
            ndcg = ndcg_score(y_true, y_score, k=5)
            ndcg_5_list.append(ndcg)
        except:
            pass
            
    r5 = np.mean(recall_5_list) if recall_5_list else 0
    rh5 = np.mean(race_hit_5_list) if race_hit_5_list else 0
    p5 = np.mean(precision_5_list) if precision_5_list else 0
    n5 = np.mean(ndcg_5_list) if ndcg_5_list else 0
    
    return {
        "Label": label,
        "Rows": len(df),
        "Races": len(race_groups),
        "Recall@5": r5,
        "Race-Hit@5": rh5,
        "Precision@5": p5,
        "NDCG@5": n5
    }

def main():
    print("Eval Mode: ADHOC (Split Pipeline)")
    print("-" * 120)
    
    paths = {
        "M3 Baseline": "experiments/exp_m3_baseline_split/valid_preds.parquet",
        "M4-A (Core)": "experiments/exp_m4_a_adhoc/valid_preds.parquet",
        "M4-B (Class)": "experiments/exp_m4_b_adhoc/valid_preds.parquet",
        "M4-C (Segment)": "experiments/exp_m4_c_adhoc/valid_preds.parquet",
    }
    
    # Load metadata
    meta = load_metadata()
    if meta is None:
        print("Skipping segmentation (no metadata)")
    
    results = []
    
    for name, path in paths.items():
        if not os.path.exists(path):
            continue
            
        df = pd.read_parquet(path)
        
        # Merge metadata if available
        if meta is not None:
            df = pd.merge(df, meta, on='race_id', how='left')
        
        # 1. Overall
        res = compute_metrics(df, "Overall")
        res["Model"] = name
        results.append(res)
        
        # 2. Segments
        if meta is not None:
            # Small Field (<= 10)
            if 'n_horses' in df.columns:
                df_small = df[df['n_horses'] <= 10]
                if len(df_small) > 0:
                    res_small = compute_metrics(df_small, "Small Field (<=10)")
                    res_small["Model"] = name
                    results.append(res_small)
            
            # Mile (1400 <= d <= 1800)
            if 'distance' in df.columns:
                # distance might be string or int? usually int in base_attributes
                # Ensure numeric
                df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
                df_mile = df[(df['distance'] >= 1400) & (df['distance'] <= 1800)]
                if len(df_mile) > 0:
                    res_mile = compute_metrics(df_mile, "Mile (14-1800m)")
                    res_mile["Model"] = name
                    results.append(res_mile)

    # Print Table
    # Pivot to show Model as Row, Segment as columns? Or just flat list?
    # Flat list sorted by Model/Label
    
    headers = ["Model", "Segment", "Race-Hit@5", "Precision@5", "Recall@5", "NDCG@5", "Races"]
    print(f"{headers[0]:<20} | {headers[1]:<20} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<10} | {headers[5]:<10} | {headers[6]:<6}")
    print("-" * 120)
    
    for r in results:
        print(f"{r['Model']:<20} | {r['Label']:<20} | {r['Race-Hit@5']:.4f}     | {r['Precision@5']:.4f}     | {r['Recall@5']:.4f}     | {r['NDCG@5']:.4f}     | {r['Races']:<6}")

if __name__ == "__main__":
    main()

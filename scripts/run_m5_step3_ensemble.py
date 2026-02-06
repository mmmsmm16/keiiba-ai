
import sys
import os
import logging
import pandas as pd
import numpy as np
import json
from sklearn.metrics import ndcg_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXP_BASE = "experiments/exp_m5_ensemble_components"
METADATA_SOURCE = "data/temp_m5/year_2024.parquet"

def load_predictions():
    targets = ['win', 'top2', 'top3']
    dfs = {}
    for t in targets:
        path = f"{EXP_BASE}/valid_preds_{t}.parquet"
        if not os.path.exists(path):
            logger.error(f"Missing prediction file: {path}")
            continue
        df = pd.read_parquet(path)
        # Rename y_prob to p_{target}
        df = df.rename(columns={'y_prob': f'p_{t}'})
        # Keep minimal cols
        if not dfs:
            dfs['base'] = df[['race_id', 'horse_number', 'y_true']].copy() # y_true might differ per target binary definition
            # Actually we want the RAW RANK for evaluation (Hit calculation)
            # valid_preds usually has y_true as binary.
            # We need the actual rank to confirm Hit@5.
            # But wait, Race-Hit@5 checks if Top3 horses are in Top5 predictions.
            # So binary y_true (IsTop3) is what we need.
            # M4 evaluation used "IsTop3" as ground truth.
            # So let's load p_top3's y_true as the ground truth "IsTop3".
            pass
        
        dfs[t] = df[['race_id', 'horse_number', f'p_{t}']]
    
    if 'top3' not in dfs:
        logger.error("Top3 predictions missing, cannot evaluate Reference.")
        return None

    # Merge
    # Start with base (using top3's structure)
    base = pd.read_parquet(f"{EXP_BASE}/valid_preds_top3.parquet")[['race_id', 'horse_number', 'y_true']].rename(columns={'y_true': 'is_top3'})
    
    for t in targets:
        if t in dfs:
            base = pd.merge(base, dfs[t], on=['race_id', 'horse_number'], how='left')
            base[f'p_{t}'] = base[f'p_{t}'].fillna(0.0)
            
    return base

def load_metadata():
    if not os.path.exists(METADATA_SOURCE):
        logger.warning(f"Metadata source {METADATA_SOURCE} not found. Segmentation unavailable.")
        return None
    
    df = pd.read_parquet(METADATA_SOURCE)
    req = ['race_id']
    if 'distance' in df.columns: req.append('distance')
    if 'n_horses' in df.columns: req.append('n_horses')
    
    meta = df[req].drop_duplicates('race_id')
    
    # If n_horses missing, calc
    if 'n_horses' not in meta.columns:
         # Rough estimation from raw data group
         counts = df.groupby('race_id')['race_id'].count().rename('n_horses_calc')
         meta = pd.merge(meta, counts, on='race_id', how='left')
         meta['n_horses'] = meta['n_horses_calc']
         
    return meta

def calc_metrics(df_in, score_col, label="Model"):
    # Group by race_id
    # We need efficient implementation
    
    # Sort by race_id to ensure grouping
    df = df_in.sort_values('race_id')
    
    race_ids = df['race_id'].unique()
    
    ndcg_list = []
    hit_list = []
    precision_list = []
    recall_list = []
    
    # Vectorized or simple loop
    # Simple loop is safer for ranking metrics
    # To speed up: groupby
    
    # Determine groups
    groups = df.groupby('race_id', sort=False).size().values
    
    y_true_all = df['is_top3'].values
    y_score_all = df[score_col].values
    
    curr = 0
    for size in groups:
        y_t = y_true_all[curr : curr + size]
        y_s = y_score_all[curr : curr + size]
        curr += size
        
        if size < 2: continue
        if np.sum(y_t) == 0: continue
        
        # NDCG@5
        ndcg_list.append(ndcg_score([y_t], [y_s], k=5))
        
        # Rankings
        top_k_idx = np.argsort(y_s)[::-1][:5]
        
        hits = np.sum(y_t[top_k_idx])
        total_relevant = np.sum(y_t)
        
        # Race-Hit (Any Hit)
        hit_list.append(1.0 if hits > 0 else 0.0)
        
        # Precision@5 (Hits / 5)
        precision_list.append(hits / 5.0)
        
        # Recall@5 (Hits / Total Relevant)
        # Note: Previous reports might have used "Race-Hit" as "Recall".
        # But M4 report clearly separated "Race-Hit" and "Recall".
        # M3 Recall@5 was 0.6061? 
        # If Race-Hit is 0.9392 (very high), then Recall 0.6061 is plausible (we find ~1.8 out of 3 horses).
        recall_list.append(hits / total_relevant if total_relevant > 0 else 0.0)

    res = {
        'Model': label,
        'Race-Hit@5': np.mean(hit_list),
        'Precision@5': np.mean(precision_list),
        'Recall@5': np.mean(recall_list),
        'NDCG@5': np.mean(ndcg_list)
    }
    return res

def main():
    logger.info("Evaluating M5 Ensemble...")
    
    df = load_predictions()
    if df is None: return
    
    meta = load_metadata()
    if meta is not None:
        df = pd.merge(df, meta, on='race_id', how='left')
    
    # 1. Define Ensemble Scores
    # p_win, p_top2, p_top3
    
    # A: Single Models
    strategies = [
        {'name': 'M5-Win', 'col': 'p_win'},
        {'name': 'M5-Top2', 'col': 'p_top2'},
        {'name': 'M5-Top3', 'col': 'p_top3'},
    ]
    
    # B: Simple Average
    df['score_avg'] = (df['p_win'] + df['p_top2'] + df['p_top3']) / 3.0
    strategies.append({'name': 'Ensemble-Avg', 'col': 'score_avg'})
    
    # C: Weighted (Win Heavy)
    df['score_w1'] = 0.5 * df['p_win'] + 0.3 * df['p_top2'] + 0.2 * df['p_top3']
    strategies.append({'name': 'Ensemble-W1', 'col': 'score_w1'})
    
    # D: Weighted (Balanced)
    df['score_w2'] = 0.4 * df['p_win'] + 0.4 * df['p_top2'] + 0.2 * df['p_top3']
    strategies.append({'name': 'Ensemble-W2', 'col': 'score_w2'})
    
    # E: Top2 Heavy
    df['score_top2'] = 0.2 * df['p_win'] + 0.6 * df['p_top2'] + 0.2 * df['p_top3']
    strategies.append({'name': 'Ensemble-Top2', 'col': 'score_top2'})

    results = []
    
    # Evaluate Overall
    print("\n=== Overall Evaluation (2024) ===")
    for s in strategies:
        res = calc_metrics(df, s['col'], s['name'])
        results.append(res)
        
    res_df = pd.DataFrame(results)
    print(res_df.to_markdown(index=False, floatfmt=".4f"))
    
    # Save Results
    res_df.to_csv(f"{EXP_BASE}/ensemble_metrics_overall.csv", index=False)
    
    # Evaluate Segments if metadata available
    if 'n_horses' in df.columns and 'distance' in df.columns:
        print("\n=== Small Field (<= 10) ===")
        small_df = df[df['n_horses'] <= 10].copy()
        res_small = []
        for s in strategies:
            res = calc_metrics(small_df, s['col'], s['name'])
            res_small.append(res)
        print(pd.DataFrame(res_small).to_markdown(index=False, floatfmt=".4f"))

        print("\n=== Mile (1400-1800) ===")
        mile_df = df[(df['distance'] >= 1400) & (df['distance'] <= 1800)].copy()
        res_mile = []
        for s in strategies:
            res = calc_metrics(mile_df, s['col'], s['name'])
            res_mile.append(res)
        print(pd.DataFrame(res_mile).to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    main()

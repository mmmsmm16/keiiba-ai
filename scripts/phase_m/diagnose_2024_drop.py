import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_ndcg(y_true, y_score, k=5):
    """
    Compute NDCG@k for a single race.
    y_true: scalar 1 for winner, 0 for others (binary relevance) OR rank.
    Usually for racing, relevance = 1/rank or binary.
    User request: "Recall@5, NDCG@5 (Most important)"
    We will use binary relevance: 1 if rank<=3 else 0? Or 1 if rank==1?
    Standard for this project: 1 if win, or relevance based on rank (1/rank).
    Let's assume Relevance = 1 for 1st, 1/2 for 2nd, 1/3 for 3rd.
    """
    # Create dataframe
    df = pd.DataFrame({'true': y_true, 'score': y_score})
    df = df.sort_values('score', ascending=False)
    
    # Relevance: 1/rank if rank <= 3 else 0?
    # Let's stick to simple binary relevance for "Winning" or "Top3" if not specified.
    # But usually NDCG implies graded relevance.
    # Let's use relevance = 1/true_rank for top 5, else 0.
    relevance = 1.0 / df['true']
    relevance = relevance.where(df['true'] <= 18, 0) # Safety
    
    # IDCG
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:k])])
    
    # DCG
    dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance.iloc[:k])])
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def compute_recall_at_k(y_true, y_score, k=5):
    """
    Fraction of relevant items in top K.
    For horse racing, typically "Did the winner land in Top K predicted?"
    y_true: rank
    """
    df = pd.DataFrame({'rank': y_true, 'score': y_score})
    top_k = df.sort_values('score', ascending=False).head(k)
    # Check if 1st place horse is in top k
    return 1.0 if (top_k['rank'] == 1).any() else 0.0

def get_metrics_for_group(g):
    """
    Compute metrics for a group of races.
    g: DataFrame containing multiple races
    """
    # AUC, LogLoss, Brier are global metrics (classification)
    # NDCG, Recall are per-race metrics (ranking)
    
    # 1. Classification Metrics (Target: Rank=1)
    y_true_binary = (g['rank'] == 1).astype(int)
    # Check if we have both 0s and 1s
    if y_true_binary.nunique() < 2:
        auc = np.nan
        ll = np.nan
        bs = np.nan
    else:
        auc = roc_auc_score(y_true_binary, g['prob'])
        ll = log_loss(y_true_binary, g['prob'])
        bs = brier_score_loss(y_true_binary, g['prob'])
    
    # 2. Ranking Metrics
    # Aggregated by mean of per-race metrics
    
    # We need to group by race_id inside this chunk
    # Optimization: Vectorized approach or group-apply
    
    def per_race_metrics(race_df):
        return pd.Series({
            'ndcg_5': compute_ndcg(race_df['rank'], race_df['prob'], k=5),
            'recall_5': compute_recall_at_k(race_df['rank'], race_df['prob'], k=5)
        })
    
    # Filter valid races (must have rank=1)
    # valid_races = g.groupby('race_id')['rank'].min() == 1
    # race_metrics = g[g['race_id'].isin(valid_races[valid_races].index)].groupby('race_id').apply(per_race_metrics)
    
    # For speed on large segments, we might take a sample or assume data integrity
    race_metrics = g.groupby('race_id').apply(per_race_metrics)
    
    return pd.Series({
        'AUC': auc,
        'LogLoss': ll,
        'Brier': bs,
        'NDCG@5': race_metrics['ndcg_5'].mean(),
        'Recall@5': race_metrics['recall_5'].mean(),
        'Count': len(g['race_id'].unique())
    })

def psi(score_initial, score_new, num_bins=10, bucket_type='bins'):
    """
    Calculate PSI (Population Stability Index)
    """
    eps = 1e-4
    
    if bucket_type == 'bins':
        breakpoints = np.linspace(0, 100, num_bins+1)
        # Use percentiles to determine bins from initial distribution
        try:
            bins = np.percentile(score_initial, breakpoints)
        except:
             bins = np.linspace(score_initial.min(), score_initial.max(), num_bins+1)
        bins[0] = -np.inf
        bins[-1] = np.inf
    else:
        # Fixed bins logic could go here
        bins = np.linspace(score_initial.min(), score_initial.max(), num_bins+1)
        
    initial_counts = np.histogram(score_initial, bins)[0]
    new_counts = np.histogram(score_new, bins)[0]
    
    initial_dist = initial_counts / len(score_initial)
    new_dist = new_counts / len(score_new)
    
    psi_val = np.sum((initial_dist - new_dist) * np.log((initial_dist + eps) / (new_dist + eps)))
    return psi_val

def main():
    DATA_PATH = 'data/derived/preprocessed_with_prob_v12.parquet'
    OUTPUT_DIR = 'reports/model_diagnostics'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter years
    df = df[df['date'].dt.year.isin([2022, 2023, 2024])].copy()
    if len(df) == 0:
        logger.error("No data found for 2022-2024.")
        return
        
    logger.info(f"Data Loaded: {len(df)} rows. Years: {df['date'].dt.year.unique()}")
    
    # --- M1-1 Segment Analysis ---
    logger.info("Starting Segment Analysis...")
    
    # Define Segment Functions
    def get_dist_cat(d):
        if d < 1400: return 'Sprint (<1400)'
        elif d < 1800: return 'Mile (1400-1799)'
        elif d < 2200: return 'Intermediate (1800-2199)' # Adjusted usually 1800-2000/2200
        else: return 'Long (>=2200)'

    df['seg_dist'] = df['distance'].apply(get_dist_cat)
    df['seg_entries'] = pd.cut(df['n_horses'], bins=[0, 10, 14, 99], labels=['Small (<=10)', 'Medium (11-14)', 'Large (15+)'])
    
    # Venue Grouping
    major_venues = ['Tokyo', 'Nakayama', 'Kyoto', 'Hanshin', 'Chukyo'] # Depending on venue naming
    # Venue name mapping might be needed if IDs are used. 
    # Inspect showed 'venue' is often ID or name. Assuming ID 1-10 (JRA).
    # 01: Sapporo, 02: Hakodate, 03: Fukushima, 04: Niigata, 05: Tokyo, 06: Nakayama, 07: Chukyo, 08: Kyoto, 09: Hanshin, 10: Kokura
    venue_map = {5: 'Tokyo', 6: 'Nakayama', 8: 'Kyoto', 9: 'Hanshin', 7: 'Chukyo'}
    df['seg_venue'] = df['venue'].map(venue_map).fillna('Local/Other') # Need to verify venue column type
    
    # Surface
    # 1: Turf, 2: Dirt (Standard mapping)
    surface_map = {1: 'Turf', 2: 'Dirt'}
    df['seg_surface'] = df['surface'].map(surface_map).fillna('Other')

    # Frame
    def get_frame_seg(f):
        if f <= 4: return 'Inner (1-4)'
        elif f <= 6: return 'Mid (5-6)'
        else: return 'Outer (7-8)'
    df['seg_frame'] = df['frame_number'].apply(get_frame_seg)
    
    # Track Condition (state)
    # 1: Good, 2: Yielding?, 3: Soft?, 4: Heavy?
    # Standard JRA: 1=Ryou(Good), 2=YayaOmo, 3=Omo, 4=Furyou
    state_map = {1: 'Good', 2: 'Slightly Heavy', 3: 'Heavy', 4: 'Bad'}
    df['seg_state'] = df['state'].map(state_map).fillna('Unknown')

    segments = {
        'Distance': 'seg_dist',
        'Field Size': 'seg_entries',
        'Venue': 'seg_venue',
        'Surface': 'seg_surface',
        'Frame': 'seg_frame',
        'Condition': 'seg_state'
    }
    
    all_metrics = []
    
    # Year Split
    df['year'] = df['date'].dt.year
    
    metrics_cols = ['NDCG@5', 'Recall@5', 'AUC', 'LogLoss']
    
    logger.info("Calculating metrics per segment...")
    
    for seg_name, seg_col in segments.items():
        if seg_col not in df.columns:
            logger.warning(f"Skipping {seg_name}, col {seg_col} not found")
            continue
            
        # Group by Year + Segment
        grouped = df.sort_values('date').groupby(['year', seg_col])
        
        # This is slow if applied directly.
        # Faster to compute race-level metrics first then aggregate?
        # But for AUC/LogLoss we need the whole set.
        
        # Let's iterate to show progress
        for (y, s_val), g in grouped:
            if len(g) < 100: continue # Skip small samples
            
            m = get_metrics_for_group(g)
            m['Year'] = y
            m['Segment_Type'] = seg_name
            m['Segment_Value'] = s_val
            all_metrics.append(m)
            
    results_df = pd.DataFrame(all_metrics)
    results_df.to_parquet(os.path.join(OUTPUT_DIR, '2024_segment_metrics.parquet'))
    
    # Analysis: Find biggest drops in 2024 vs 2022-2023 avg
    pivot = results_df.pivot_table(index=['Segment_Type', 'Segment_Value'], columns='Year', values=metrics_cols)
    
    # Calculate drops
    report_md = "# 2024 Diagnostics Report\n\n"
    report_md += "## M1-1. Segment Analysis\n\n"
    report_md += "| Segment Type | Segment Value | Metric | 2024 | '22-'23 Avg | Delta |\n"
    report_md += "|---|---|---|---|---|---|\n"
    
    deltas = []
    
    if 2024 in pivot.columns.get_level_values(1):
        for idx in pivot.index:
            s_type, s_val = idx
            for metric in metrics_cols:
                val_2024 = pivot.loc[idx, (metric, 2024)]
                val_prev = pivot.loc[idx, (metric, [2022, 2023])].mean()
                
                delta = val_2024 - val_prev
                # For Loss/Brier, positive delta is BAD. For others negative is BAD.
                is_bad = delta > 0 if metric in ['LogLoss', 'Brier'] else delta < 0
                
                if pd.notna(delta) and is_bad:
                    deltas.append({
                        'Type': s_type, 'Value': s_val, 'Metric': metric,
                        '2024': val_2024, 'Prev': val_prev, 'Delta': delta,
                        'AbsDelta': abs(delta)
                    })
    
    deltas_df = pd.DataFrame(deltas).sort_values('AbsDelta', ascending=False)
    
    for i, row in deltas_df.head(20).iterrows():
        report_md += f"| {row['Type']} | {row['Value']} | {row['Metric']} | {row['2024']:.4f} | {row['Prev']:.4f} | {row['Delta']:.4f} |\n"
        
    # --- M1-2 Shift Analysis ---
    logger.info("Starting Shift Analysis...")
    report_md += "\n## M1-2. Distribution Shift (PSI)\n\n"
    
    # Identify numerical features
    # Exclude IDs, dates, target
    exclude = ['race_id', 'date', 'title', 'year', 'month', 'day', 'prob', 'rank', 'horse_name', 'jockey_id', 'trainer_id', 'owner_id', 'horse_id', 'sire_id', 'mare_id', 'venue', 'race_number', 'fold_year', 'model_version']
    num_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in num_cols if c not in exclude and not c.startswith('seg_')]
    
    # 2024 vs 2022+2023
    df_prev = df[df['year'].isin([2022, 2023])]
    df_2024 = df[df['year'] == 2024]
    
    psi_scores = []
    for col in feature_cols:
        if df_prev[col].nunique() < 5: continue # Skip categorical/ordinal with few values for PSI for now
        
        try:
            val = psi(df_prev[col].dropna(), df_2024[col].dropna())
            psi_scores.append({'Feature': col, 'PSI': val})
        except:
            pass
            
    psi_df = pd.DataFrame(psi_scores).sort_values('PSI', ascending=False)
    
    report_md += "| Feature | PSI |\n|---|---|\n"
    for i, row in psi_df.head(20).iterrows():
        name = row['Feature']
        val = row['PSI']
        alert = "⚠️" if val > 0.1 else "" 
        report_md += f"| {name} | {val:.4f} {alert} |\n"
        
    with open(os.path.join(OUTPUT_DIR, '2024_shift_report.md'), 'w', encoding='utf-8') as f:
        f.write(report_md)
        
    logger.info("Diagnostics Completed. Reports saved.")

if __name__ == "__main__":
    main()

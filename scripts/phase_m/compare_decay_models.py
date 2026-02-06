import pandas as pd
import numpy as np
import os
import argparse
import logging
import pickle
import glob
import yaml
import sys
from sklearn.metrics import roc_auc_score, log_loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_ndcg(y_true, y_score, k=5):
    # Sort by score desc
    df = pd.DataFrame({'rank': y_true, 'score': y_score})
    df_sorted = df.sort_values('score', ascending=False)
    
    # Calculate IDCG (Ideal DCG based on actual ranks)
    # Relevance: 1/rank for rank <= 18
    relevance = 1.0 / df['rank']
    relevance = relevance.where(df['rank'] <= 18, 0)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:k])])
    
    if idcg == 0: return 0.0

    # Calculate DCG (Based on predicted order)
    # Re-calculate relevance for sorted rows
    sorted_relevance = 1.0 / df_sorted['rank']
    sorted_relevance = sorted_relevance.where(df_sorted['rank'] <= 18, 0)
    
    dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevance.iloc[:k])])
    
    return dcg / idcg

def get_metrics_for_df(df_input, target='top3'):
    def per_race(g):
        return pd.Series({
            'ndcg_5': compute_ndcg(g['rank'], g['prob'], k=5),
            'recall_5': compute_recall_at_k(g['rank'], g['prob'], k=5, target=target)
        })
    race_metrics = df_input.groupby('race_id').apply(per_race)
    
    if target == 'win':
        y_bin = (df_input['rank'] == 1).astype(int)
    elif target == 'top2':
        y_bin = (df_input['rank'] <= 2).astype(int)
    else: # top3
        y_bin = (df_input['rank'] <= 3).astype(int)
        
    auc = roc_auc_score(y_bin, df_input['prob']) if y_bin.nunique() > 1 else np.nan
    ll = log_loss(y_bin, df_input['prob']) if y_bin.nunique() > 1 else np.nan
    return {
        'AUC': auc,
        'LogLoss': ll,
        'NDCG@5': race_metrics['ndcg_5'].mean(),
        'Recall@5': race_metrics['recall_5'].mean(),
        'Count': len(df_input['race_id'].unique())
    }

def compute_recall_at_k(y_true, y_score, k=5, target='top3'):
    df = pd.DataFrame({'rank': y_true, 'score': y_score})
    top_k = df.sort_values('score', ascending=False).head(k)
    
    threshold = 3
    if target == 'win': threshold = 1
    elif target == 'top2': threshold = 2
        
    return 1.0 if (top_k['rank'] <= threshold).any() else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, default='models/experiments')
    parser.add_argument('--pattern', type=str, default='m3_top3_*')
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--target', type=str, default='top3', choices=['win', 'top2', 'top3'])
    args = parser.parse_args()
    
    exp_dirs = glob.glob(os.path.join(args.models_dir, args.pattern))
    if not exp_dirs: return
    
    first_exp_config = os.path.join(exp_dirs[0], 'config.yaml')
    with open(first_exp_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        feature_blocks = config.get('features', [])

    # Load only 2024 data is sufficient IF we use model Dump for categories
    # Load data
    # [Fix] Load history from 2015 to support rolling features (M4-A/B/C)
    # Then slice for target year after feature loading
    start_date = "2015-01-01"
    end_date = f"{args.year}-12-31"

    loader = JraVanDataLoader()
    logger.info(f"Loading data from {start_date} to {end_date} for feature consistency...")
    raw_df = loader.load(history_start_date=start_date, end_date=end_date, jra_only=True, skip_odds=True)
    
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    pipeline = FeaturePipeline(cache_dir="data/features")
    df = pipeline.load_features(clean_df, feature_blocks)
    
    # Metadata Merge
    if 'race_id' not in df.columns: df['race_id'] = clean_df['race_id']
    if 'horse_number' not in df.columns: df['horse_number'] = clean_df['horse_number']
    
    cols_to_merge = ['rank', 'n_horses', 'distance', 'date']
    merged_cols = [c for c in cols_to_merge if c in clean_df.columns and c not in df.columns]
    if merged_cols:
        target_source = clean_df[['race_id', 'horse_number'] + merged_cols].drop_duplicates()
        df = pd.merge(df, target_source, on=['race_id', 'horse_number'], how='left')

    if 'n_horses' not in df.columns:
        df['n_horses'] = df.groupby('race_id')['horse_number'].transform('count')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Segments
    def get_dist_cat(d):
        if d < 1400: return 'Sprint'
        elif d < 1800: return 'Mile' 
        elif d < 2200: return 'Intermediate'
        else: return 'Long'
    df['seg_dist'] = df['distance'].apply(get_dist_cat)
    df['seg_entries'] = pd.cut(df['n_horses'], bins=[0, 10, 14, 99], labels=['Small', 'Medium', 'Large'])

    results = []
    
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        model_path = os.path.join(exp_dir, 'model.pkl')
        if not os.path.exists(model_path): continue
        
        logger.info(f"Evaluating {exp_name}...")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Prepare X
            if hasattr(model, 'feature_name'):
                model_features = model.feature_name()
                X = df[model_features].copy()
            else:
                drop = ['race_id', 'horse_id', 'date', 'target', 'year', 'rank', 'prob', 'seg_dist', 'seg_entries', 'weight']
                X = df.drop(columns=[c for c in drop if c in df.columns], axis=1)

            # Convert to codes for robust prediction (Numpy fallback)
            # This relies on the fact that we loaded full history, so cat codes should match training
            # We must handle both 'object' and 'category'
            for col in X.columns:
                if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
                    X[col] = X[col].astype('category').cat.codes
            
            # Predict on numpy array to bypass LightGBM categorical metadata check
            probs = model.predict(X.values)
            
            df_pred = df.copy()
            df_pred['prob'] = probs
            
            m_overall = get_metrics_for_df(df_pred, target=args.target) # Evaluate on all (2024 full)
            
            # Filter for segments (if needed, but m_overall includes all)
            # We want metrics on 2024 (which df is)
            
            m_small = get_metrics_for_df(df_pred[df_pred['seg_entries'] == 'Small'], target=args.target)
            m_mile = get_metrics_for_df(df_pred[df_pred['seg_dist'] == 'Mile'], target=args.target)
            
            results.append({
                'Exp': exp_name,
                'Overall_Recall': m_overall['Recall@5'], 'Overall_NDCG': m_overall['NDCG@5'],
                'Small_Recall': m_small['Recall@5'], 'Small_NDCG': m_small['NDCG@5'],
                'Mile_Recall': m_mile['Recall@5'], 'Mile_NDCG': m_mile['NDCG@5'],
                'Overall_AUC': m_overall['AUC']
            })
        except Exception as e:
            logger.error(f"Failed {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        logger.error("No results!")
        return

    res_df = pd.DataFrame(results).sort_values('Overall_Recall', ascending=False)
    print("\n=== M3 Comparison Results (2024) ===\n")
    print(res_df.to_markdown(index=False, floatfmt=".4f"))
    
    output_file = f'reports/model_diagnostics/m3_{args.target}_results.md'
    with open(output_file, 'w') as f:
        f.write(f"# M3 {args.target.capitalize()} Time-Decay Experiment Results\n\n" + res_df.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    main()

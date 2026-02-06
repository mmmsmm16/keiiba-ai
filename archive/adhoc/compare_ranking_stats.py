import os
import sys
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))     # Project Root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))      # src directory

from src.model.ensemble import EnsembleModel
from src.inference.preprocessor import InferencePreprocessor
from src.inference.loader import InferenceDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_v5_predictions():
    path = "reports/predictions_v5_2025.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"v5 predictions not found at {path}")
    df = pd.read_csv(path)
    # Ensure race_id is string
    df['race_id'] = df['race_id'].astype(str)
    return df

def generate_v4_predictions(race_ids):
    output_path = "reports/predictions_v4_2025.csv"
    
    if os.path.exists(output_path):
        logger.info(f"Loading existing v4 predictions from {output_path}...")
        return pd.read_csv(output_path)
    
    logger.info("Generating v4 predictions...")
    
    # Load Model
    model = EnsembleModel()
    model_path = "models/ensemble_v4_2025.pkl"
    if not os.path.exists(model_path):
         # Fallback try
         model_path = "models/ensemble_v4.pkl"
    
    logger.info(f"Loading v4 model from {model_path}")
    model.load_model(model_path)
    
    loader = InferenceDataLoader()
    preprocessor = InferencePreprocessor()
    
    # Process in batches
    batch_size = 100
    all_preds = []
    
    # Split race_ids into batches
    race_id_list = list(race_ids)
    batches = [race_id_list[i:i + batch_size] for i in range(0, len(race_id_list), batch_size)]
    
    for batch in tqdm(batches, desc="Evaluating v4"):
        try:
            # Load Data
            raw_df = loader.load(race_ids=batch)
            if raw_df.empty: continue
            
            # Preprocess
            X, ids = preprocessor.preprocess(raw_df)
            
            # Predict
            scores = model.predict(X)
            
            # Create Result DF
            batch_result = ids.copy()
            batch_result['score'] = scores
            
            # If 'rank' column exists in raw_df, we need to preserve it
            # But preprocessor separates X and ids. 'rank' might be in X or dropped?
            # ids usually has index info. We need to merge 'rank' back if possible.
            # InferenceDataLoader loads 'rank' if available (it does for jvd_se).
            # But Preprocessor 'preprocess' returns 'ids' which usually only has minimal columns.
            # Let's check if 'rank' is in raw_df and merge.
            
            if 'rank' in raw_df.columns:
                 # Raw DF has rank. We need to map it back to the rows.
                 # raw_df and X/ids should have same index if preprocessor doesn't drop rows.
                 # Preprocessor MIGHT drop rows (e.g. missing combined odds).
                 # This is tricky. 
                 # Best way: X, ids = preprocess(raw_df). ids has the original index?
                 # Assuming ids aligns with X.
                 # Let's trying to merge 'rank' from raw_df using index.
                 
                 # Ensure unique index
                 pass

            all_preds.append(batch_result)
            
        except Exception as e:
            logger.error(f"Error in batch: {e}")
            continue

    if not all_preds:
        return pd.DataFrame()
        
    df_pred = pd.concat(all_preds, ignore_index=True)
    
    # Rank within race
    df_pred['pred_rank'] = df_pred.groupby('race_id')['score'].rank(ascending=False, method='min')
    
    # We really need actual 'rank' for evaluation.
    # The 'loader.load' fetches from DB. If it's 2025 past data, 'jvd_se' should have rank.
    # But 'ids' returned by preprocessor might not contain 'rank'.
    # We must fetch actual results separately or ensure they are carried over.
    # Let's assume we can fetch actuals using 'jvd_se' again or just use the v5 dataframe's meta-data if it has actual rank.
    
    # Let's save what we have.
    df_pred.to_csv(output_path, index=False)
    return df_pred

def calculate_metrics(df, model_name):
    # Ensure we have 'rank' (actual) and 'pred_rank' (predicted)
    if 'rank' not in df.columns:
        # Try to merge rank from v5 df if missing? or fail.
        # For now assume it exists or is merged before calling this.
        return None

    stats = []
    
    # Filter valid ranks (exclude NaN, 0, 99 etc if any)
    valid_df = df[df['rank'].notna() & (df['rank'] > 0)].copy()
    
    for r in range(1, 19): # Rank 1 to 18
        # Filter horses predicted as Rank 'r'
        target_df = valid_df[valid_df['pred_rank'] == r]
        count = len(target_df)
        
        if count == 0:
            stats.append({
                'Model': model_name, 'Pred Rank': r, 'Count': 0,
                'Win Rate': 0, 'Ren (Top 2)': 0, 'Fuku (Top 3)': 0
            })
            continue
            
        # Win: actual rank == 1
        wins = len(target_df[target_df['rank'] == 1])
        # Ren: actual rank <= 2
        rens = len(target_df[target_df['rank'] <= 2])
        # Fuku: actual rank <= 3
        fukus = len(target_df[target_df['rank'] <= 3])
        
        stats.append({
            'Model': model_name,
            'Pred Rank': r,
            'Count': count,
            'Win Rate': wins / count,
            'Ren (Top 2)': rens / count,
            'Fuku (Top 3)': fukus / count
        })
        
    return pd.DataFrame(stats)

def main():
    # 1. Load v5
    v5_df = load_v5_predictions()
    
    # Calculate v5 Pred Rank if not exists
    if 'pred_rank' not in v5_df.columns:
        v5_df['pred_rank'] = v5_df.groupby('race_id')['score'].rank(ascending=False, method='min')
    
    race_ids = v5_df['race_id'].unique()
    logger.info(f"Target Races: {len(race_ids)}")

    # 2. Get v4
    v4_df = generate_v4_predictions(race_ids)
    
    # 3. Merge Actual Rank (Critical step)
    # v5 csv usually has 'rank' column if it was generated by evaluate script.
    # Let's check v5_df columns in runtime.
    # If v4_df is generated newly, it might miss 'rank'.
    # We can merge 'rank' from v5_df to v4_df using (race_id, horse_number).
    
    if 'rank' in v5_df.columns:
        actuals = v5_df[['race_id', 'horse_number', 'rank']].copy()
        # Merge to v4
        v4_df = pd.merge(v4_df, actuals, on=['race_id', 'horse_number'], how='left', suffixes=('', '_actual'))
        # If v4 had 'rank', merge might create conflict. Handle suffixes.
        if 'rank_actual' in v4_df.columns:
             v4_df['rank'] = v4_df['rank'].fillna(v4_df['rank_actual'])
             v4_df = v4_df.drop(columns=['rank_actual'])
    
    # 4. Calculate Stats
    df_stats_v5 = calculate_metrics(v5_df, 'v5 (JRA-Only)')
    df_stats_v4 = calculate_metrics(v4_df, 'v4 (Full)')
    
    # 5. Display
    print("\n=== 📊 Prediction Accuracy by Rank (JRA 2025) ===")
    
    for rank in range(1, 6): # Show Top 5 Ranks
        row_v5 = df_stats_v5[df_stats_v5['Pred Rank'] == rank].iloc[0]
        row_v4 = df_stats_v4[df_stats_v4['Pred Rank'] == rank].iloc[0]
        
        print(f"\n[Rank {rank}]")
        print(f"  v4: Win={row_v4['Win Rate']:.1%} | Ren={row_v4['Ren (Top 2)']:.1%} | Fuku={row_v4['Fuku (Top 3)']:.1%}")
        print(f"  v5: Win={row_v5['Win Rate']:.1%} | Ren={row_v5['Ren (Top 2)']:.1%} | Fuku={row_v5['Fuku (Top 3)']:.1%}")
        
    print("\n--- Detailed Table ---")
    combined = pd.concat([df_stats_v4, df_stats_v5])
    print(combined.pivot(index='Pred Rank', columns='Model', values=['Win Rate', 'Ren (Top 2)', 'Fuku (Top 3)']))

if __name__ == "__main__":
    main()

import sys
import os
import pandas as pd
import numpy as np

# Adjust path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from inference.preprocessor import InferencePreprocessor

def main():
    # 1. Load Training Features
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    datasets = pd.read_pickle(dataset_path)
    train_cols = datasets['train']['X'].columns.tolist()
    print(f"Training Features ({len(train_cols)}):")
    print(train_cols)
    
    # 2. Generate Inference Features
    preprocessor = InferencePreprocessor()
    
    # Mock New DF
    # Needs columns: date, race_id, horse_id, jockey_id, trainer_id, sire_id, etc.
    # And numeric columns for simple features.
    
    # Load a small sample from history to use as new_df
    history_path = os.path.join(os.path.dirname(__file__), '../../data/processed/preprocessed_data.parquet')
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found.")
        return
        
    history_df = pd.read_parquet(history_path)
    # Pick a recent row
    new_df = history_df.tail(10).copy()
    # Reset some cols to simulate raw state? 
    # Actually InferencePreprocessor handles raw-like df.
    # But usually new_df comes from loader.
    # Let's hope history_df enough to pass basic checks.
    # We might need to ensure 'venue' etc are present.
    
    # Run preprocess
    try:
        X_inf, _ = preprocessor.preprocess(new_df, history_df=history_df)
        inf_cols = X_inf.columns.tolist()
        print(f"\nInference Features ({len(inf_cols)}):")
        print(inf_cols)
        
        # 3. Compare
        train_set = set(train_cols)
        inf_set = set(inf_cols)
        
        missing = train_set - inf_set
        extra = inf_set - train_set
        
        print(f"\nMissing in Inference ({len(missing)}):")
        print(missing)
        
        print(f"\nExtra in Inference ({len(extra)}):")
        print(extra)
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

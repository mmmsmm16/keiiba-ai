
import os
import sys
import pandas as pd
import pickle
import logging

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.dataset import DatasetSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Regenerating LGBM Datasets with Weights...")
    
    input_path = 'data/processed/preprocessed_data.parquet'
    output_path = 'data/processed/lgbm_datasets.pkl'
    
    if not os.path.exists(input_path):
        logger.error(f"Input not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded DataFrame: {len(df)} rows")
    
    splitter = DatasetSplitter()
    datasets = splitter.split_and_create_dataset(df)
    
    with open(output_path, 'wb') as f:
        pickle.dump(datasets, f)
        
    logger.info(f"Saved datasets to: {output_path}")

if __name__ == "__main__":
    main()

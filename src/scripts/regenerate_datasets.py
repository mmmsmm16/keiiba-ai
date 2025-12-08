
import os
import sys
import argparse
import pickle
import pandas as pd
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.dataset import DatasetSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Regenerate LGBM Datasets')
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data.parquet')
    parser.add_argument('--output', type=str, default='data/processed/lgbm_datasets.pkl')
    parser.add_argument('--valid_year', type=int, default=2024, help='Year to use as Validation Set')
    args = parser.parse_args()
    
    logger.info(f"Regenerating LGBM Datasets (Valid Year: {args.valid_year})...")
    
    # Check Input
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
        
    # Load Data
    logger.info("Loading preprocessed data...")
    df = pd.read_parquet(args.input)
    if 'race_id' not in df.columns: df = df.reset_index()
    
    # Split
    splitter = DatasetSplitter()
    datasets = splitter.split_and_create_dataset(df, valid_year=args.valid_year)
    
    # Save
    logger.info(f"Saving datasets to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(datasets, f)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()

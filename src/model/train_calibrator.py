
import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import logging
from sklearn.isotonic import IsotonicRegression

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.lgbm import KeibaLGBM
from model.catboost_model import KeibaCatBoost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds='clip')

    def fit(self, scores, labels):
        self.ir.fit(scores, labels)

    def predict(self, scores):
        return self.ir.predict(scores)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.ir, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.ir = pickle.load(f)

def load_datasets():
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    return datasets['train'], datasets['valid']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='lgbm')
    parser.add_argument('--model_version', type=str, required=True)
    args = parser.parse_args()
    
    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    model_path = os.path.join(model_dir, f'{args.model_type}_{args.model_version}.pkl')
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    # Load Model
    if args.model_type == 'lgbm':
        model = KeibaLGBM()
    else:
        model = KeibaCatBoost()
    model.load_model(model_path)
    
    # Load Data (Valid Set)
    _, valid_set = load_datasets()
    
    # Predict
    logger.info("Predicting scores on validation set...")
    scores = model.predict(valid_set['X'])
    labels = valid_set['y']
    
    # Binary Target for Calibration (Rank 1 = 1, else 0? Or Top 3?)
    # Usually we calibrate for Win Probability (Rank 1).
    # But our betting strategy uses "calibrated_prob" for EV check.
    # If EV check is for "Winning", we should calibrate to Win.
    # Current strategy checks: (prob * odds) > 1.2. This implies "prob" is Win Probability.
    # So we fit Isotonic on (Score, IsWin).
    
    # Target in dataset is 3(1st), 2(2nd), 1(3rd), 0.
    binary_target = (labels == 3).astype(int)
    
    logger.info("Training Calibrator (Score -> Win Probability)...")
    calibrator = ProbabilityCalibrator()
    calibrator.fit(scores, binary_target)
    
    save_path = os.path.join(model_dir, f'calibrator_{args.model_version}.pkl')
    calibrator.save(save_path)
    logger.info(f"Saved calibrator to {save_path}")

if __name__ == "__main__":
    main()

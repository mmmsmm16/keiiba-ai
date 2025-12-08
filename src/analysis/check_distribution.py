
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import pickle
from scipy.special import softmax

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.lgbm import KeibaLGBM
from model.evaluate_betting_roi import calculate_race_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--model_version', type=str, default='v5_weighted')
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data.parquet')
    args = parser.parse_args()

    model_dir = 'models'
    
    # Load Data
    df = pd.read_parquet(args.input)
    if 'race_id' not in df.columns: df = df.reset_index()
    valid_df = df[df['year'] == args.year].copy()
    
    # Load Model
    model = KeibaLGBM()
    model.load_model(os.path.join(model_dir, f'lgbm_{args.model_version}.pkl'))
    
    feature_cols = getattr(model.model, 'feature_name', lambda: [])()
    if not feature_cols:
         X_valid = valid_df.select_dtypes(include=[np.number])
    else:
         for c in feature_cols: 
              if c not in valid_df.columns: valid_df[c] = 0
         X_valid = valid_df[feature_cols]

    valid_df['score'] = model.predict(X_valid)
    valid_df['prob'] = valid_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    
    # Calibrator
    calib_path = os.path.join(model_dir, f'calibrator_{args.model_version}.pkl')
    if os.path.exists(calib_path):
        from model.calibration import ProbabilityCalibrator
        calibrator = ProbabilityCalibrator()
        calibrator.load(calib_path)
        valid_df['calibrated_prob'] = calibrator.predict(valid_df['prob'].values)
    else:
        valid_df['calibrated_prob'] = valid_df['prob']

    # Betting features
    clean_valid = valid_df[['race_id', 'odds', 'horse_number', 'calibrated_prob', 'score', 'date']].copy()
    race_feats = calculate_race_features(clean_valid)
    
    features = ['entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses']
    
    # Betting Model
    path_win = os.path.join(model_dir, f'betting_model_{args.model_version}_win.pkl')
    path_place = os.path.join(model_dir, f'betting_model_{args.model_version}_place.pkl')
    
    with open(path_win, 'rb') as f: model_win = pickle.load(f)
    with open(path_place, 'rb') as f: model_place = pickle.load(f)
    
    race_feats['conf_win'] = model_win.predict(race_feats[features])
    race_feats['conf_place'] = model_place.predict(race_feats[features])
    
    print("=== Distributions ===")
    print("Calibrated Prob (Top 1):")
    top1_probs = valid_df.loc[valid_df.groupby('race_id')['score'].idxmax(), 'calibrated_prob']
    print(top1_probs.describe())
    
    print("\nConf Win:")
    print(race_feats['conf_win'].describe())
    
    print("\nConf Place:")
    print(race_feats['conf_place'].describe())
    
    print("\nEV Check (Prob * Odds > 1.2) Count:")
    valid_df['ev_val'] = valid_df['calibrated_prob'] * valid_df['odds']
    top1_ev = valid_df.loc[valid_df.groupby('race_id')['score'].idxmax(), 'ev_val']
    print((top1_ev > 1.2).value_counts())

if __name__ == "__main__":
    main()

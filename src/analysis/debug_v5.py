
import os
import sys
import pandas as pd
import numpy as np
import pickle
from scipy.special import softmax

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.lgbm import KeibaLGBM
from model.evaluate import load_payout_data
from model.betting_strategy import BettingOptimizer
from model.evaluate_betting_roi import calculate_race_features

def main():
    print("DEBUG: Starting Pipeline")
    year = 2024
    model_version = 'v5_weighted'
    
    # 1. Load Data
    print("DEBUG: Loading Data...")
    df = pd.read_parquet('data/processed/preprocessed_data.parquet')
    if 'race_id' not in df.columns: df = df.reset_index()
    valid_df = df[df['year'] == year].copy()
    print(f"DEBUG: Valid DF Size: {len(valid_df)}")
    
    # 2. Model
    print("DEBUG: Loading Model...")
    model = KeibaLGBM()
    model.load_model(f'models/lgbm_{model_version}.pkl')
    
    feature_cols = getattr(model.model, 'feature_name', lambda: [])()
    if not feature_cols:
         X_valid = valid_df.select_dtypes(include=[np.number])
    else:
         for c in feature_cols: 
              if c not in valid_df.columns: valid_df[c] = 0
         X_valid = valid_df[feature_cols]

    print("DEBUG: Predicting...")
    valid_df['score'] = model.predict(X_valid)
    valid_df['prob'] = valid_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    
    # Calibrator
    calib_path = f'models/calibrator_{model_version}.pkl'
    if os.path.exists(calib_path):
        from model.calibration import ProbabilityCalibrator
        calibrator = ProbabilityCalibrator()
        calibrator.load(calib_path)
        valid_df['calibrated_prob'] = calibrator.predict(valid_df['prob'].values)
    else:
        valid_df['calibrated_prob'] = valid_df['prob']
        
    print(f"DEBUG: Calibrated Prob Mean: {valid_df['calibrated_prob'].mean()}")

    # Betting Models
    print("DEBUG: Loading Betting Models...")
    with open(f'models/betting_model_{model_version}_win.pkl', 'rb') as f: model_win = pickle.load(f)
    with open(f'models/betting_model_{model_version}_place.pkl', 'rb') as f: model_place = pickle.load(f)
    
    clean_valid = valid_df[['race_id', 'odds', 'horse_number', 'calibrated_prob', 'score', 'date']].copy()
    race_feats = calculate_race_features(clean_valid)
    
    features = ['entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses']
    race_feats['conf_win'] = model_win.predict(race_feats[features])
    race_feats['conf_place'] = model_place.predict(race_feats[features])
    
    conf_map = race_feats.set_index('race_id')[['conf_win', 'conf_place']].to_dict('index')
    print(f"DEBUG: Conf Map Size: {len(conf_map)}")
    
    # Optimizer / Payouts
    print("DEBUG: Loading Payouts...")
    payout_df = load_payout_data(years=[year])
    print(f"DEBUG: Payout DF Size: {len(payout_df)}")
    optimizer = BettingOptimizer(clean_valid, payout_df)
    print(f"DEBUG: Payout Map Size: {len(optimizer.payout_map)}")
    
    # Simulation Logic Check
    print("DEBUG: Running Simulation Logic Loop...")
    processed_count = 0
    skipped_payout = 0
    skipped_horses = 0
    
    for race_id, group in clean_valid.groupby('race_id'):
        sorted_horses = group.sort_values('score', ascending=False)
        if len(sorted_horses) < 6: 
            skipped_horses += 1
            continue
            
        payouts = optimizer.payout_map.get(race_id, {})
        if not payouts: 
            skipped_payout += 1
            continue
            
        processed_count += 1
        
    print(f"DEBUG: Processed: {processed_count}, Skipped Horses: {skipped_horses}, Skipped Payout: {skipped_payout}")

if __name__ == "__main__":
    main()

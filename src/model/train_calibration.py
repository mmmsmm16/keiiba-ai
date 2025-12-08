
import os
import sys
import pickle
import pandas as pd
import numpy as np
import argparse
import logging
from scipy.special import softmax

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.lgbm import KeibaLGBM
from model.ensemble import EnsembleModel
from model.calibration import ProbabilityCalibrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Probability Calibrator')
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data.parquet')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--base_model', type=str, default='lgbm', choices=['lgbm', 'ensemble'])
    parser.add_argument('--year', type=int, default=2023, help='Validation Year to fit calibrator')
    args = parser.parse_args()

    # 1. Load Validation Data
    logger.info(f"Loading data... (Using year {args.year} for calibration fit)")
    df = pd.read_parquet(args.input)
    
    # Validation set (e.g., 2023) to fit calibrator without overfitting to Train (<=2022) or Test (2024)
    # Ideally, we should use the same validation set as used during model training, or a hold-out set.
    val_df = df[df['year'] == args.year].copy()
    
    if val_df.empty:
        logger.error(f"No validation data found for year {args.year}")
        return

    # 2. Load Base Model and Predict using RAW Score (before Softmax)
    logger.info(f"Loading base model: {args.base_model}")
    model = None
    if args.base_model == 'lgbm':
        model = KeibaLGBM()
        model_path = os.path.join(args.model_dir, 'lgbm.pkl')
        model.load_model(model_path)
    elif args.base_model == 'ensemble':
        model = EnsembleModel()
        model_path = os.path.join(args.model_dir, 'ensemble_model.pkl')
        model.load_model(model_path)
    
    # Predict
    # Need features
    if args.base_model == 'lgbm':
        # Get features from model or dataset
        # Assuming we can just use numeric features + standard categorical handling if pipeline matches
        # Best to load features list from saved artifact if possible, but fallback to naive
        feature_cols = None
        if hasattr(model.model, 'feature_name'):
            feature_cols = model.model.feature_name()
    
    # Just grab numeric cols if feature list unknown (RISKY but standard fallback in this project)
    if not feature_cols:
        # Try loading from dataset pickle metadata if available
        pass 
        # Fallback
        X_val = val_df.select_dtypes(include=[np.number])
        # exclude meta
        exclude = ['rank', 'horse_number', 'year', 'month', 'day', 'race_number', 'odds', 'popularity', 'return_amount']
        X_val = X_val[[c for c in X_val.columns if c not in exclude]]
    else:
        # Fill missing
        for c in feature_cols:
             if c not in val_df.columns: val_df[c] = 0
        X_val = val_df[feature_cols]

    logger.info("Predicting raw scores...")
    raw_scores = model.predict(X_val)
    val_df['raw_score'] = raw_scores
    
    # 3. Prepare Calibration Targets
    # Softmax within race to get "Model Probability" (Current uncalibrated prob)
    # Actually, Isotonic Regression works best on the RAW SCORE (Logit) -> Probability mapping
    # OR Softmax Prob -> True Probability mapping.
    # Since we use Softmax for final output, let's calibrate the Softmax Output? 
    # NO. Isotonic Regression is monotonic. Softmax is monotonic w.r.t raw score (if others fixed).
    # But Softmax depends on other horses.
    # Strategy:
    # A) Global Calibration: Raw Score -> Win Prob (Independent of race members? No, rank learning...)
    # B) Softmax Calibration: Softmax Prob -> Win Prob.
    # B is safer for "Probability correction".
    
    logger.info("Calculating Softmax probabilities...")
    val_df['prob'] = val_df.groupby('race_id')['raw_score'].transform(lambda x: softmax(x))
    
    # Target: Win (Rank=1)
    y_true = (val_df['rank'] == 1).astype(int).values
    X_prob = val_df['prob'].values
    
    # 4. Fit Calibrator
    logger.info("Fitting Isotonic Regression (Softmax Prob -> True Win Prob)...")
    calibrator = ProbabilityCalibrator()
    calibrator.fit(X_prob, y_true)
    
    # 5. Save
    save_path = os.path.join(args.model_dir, 'calibrator.pkl')
    calibrator.save(save_path)
    
    # 6. Evaluation on Validation Set (Self-check)
    calibrated_prob = calibrator.predict(X_prob)
    
    # Binning check
    df_res = pd.DataFrame({'prob': X_prob, 'calib_prob': calibrated_prob, 'win': y_true})
    df_res['bin'] = pd.cut(df_res['prob'], bins=np.linspace(0, 1, 11))
    grouped = df_res.groupby('bin')[['win', 'calib_prob', 'prob']].mean()
    print("\n--- Calibration Check (Validation Data) ---")
    print(grouped)
    
if __name__ == "__main__":
    main()

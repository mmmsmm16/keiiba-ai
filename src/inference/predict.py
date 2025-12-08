
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import pickle
from datetime import datetime
from scipy.special import softmax

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.lgbm import KeibaLGBM
from model.catboost_model import KeibaCatBoost
from model.tabnet_model import KeibaTabNet
from model.ensemble import EnsembleModel
from inference.preprocessor import InferencePreprocessor
from inference.loader import InferenceDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument('--date', type=str, required=True, help='Target Date (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, default='ensemble', choices=['lgbm', 'catboost', 'tabnet', 'ensemble'])
    parser.add_argument('--version', type=str, default='v1')
    args = parser.parse_args()
    
    # 1. Load Data
    loader = InferenceDataLoader()
    # 開催日を判定 (指定日が開催日でない場合、直近の過去開催日を取得したりするロジックが必要だが、ここでは指定日必須とする)
    target_date = args.date.replace('-', '') # YYYYMMDD
    
    logger.info(f"Loading data for date: {target_date}")
    raw_df = loader.load_race_data(target_date)
    
    if raw_df.empty:
        logger.warning(f"No race data found for {target_date}")
        return

    # 2. Preprocess
    logger.info("Preprocessing...")
    preprocessor = InferencePreprocessor()
    # 過去データ(学習用)を使って特徴量エンジニアリング
    # 注意: ここで過去データを全ロードすると重いので、キャッシュを使うか、あるいは推論に必要な統計量だけロードする仕組みが推奨される
    # 現状の実装: load_data() で train_data をロードしている
    preprocessor.load_data() 
    
    processed_df = preprocessor.process(raw_df)
    
    if processed_df.empty:
        logger.error("Preprocessing failed (empty result).")
        return

    # 3. Load Model
    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    model = None
    
    logger.info(f"Loading Model: {args.model} ({args.version})")
    try:
        if args.model == 'ensemble':
            model = EnsembleModel()
            path = os.path.join(model_dir, f'ensemble_{args.version}.pkl')
            if not os.path.exists(path): path = os.path.join(model_dir, 'ensemble_model.pkl')
            model.load_model(path)
        elif args.model == 'lgbm':
            model = KeibaLGBM()
            path = os.path.join(model_dir, f'lgbm_{args.version}.pkl')
            if not os.path.exists(path): path = os.path.join(model_dir, 'lgbm.pkl')
            model.load_model(path)
        elif args.model == 'catboost':
            model = KeibaCatBoost()
            path = os.path.join(model_dir, f'catboost_{args.version}.pkl')
            if not os.path.exists(path): path = os.path.join(model_dir, 'catboost.pkl')
            model.load_model(path)
        elif args.model == 'tabnet':
            model = KeibaTabNet()
            path = os.path.join(model_dir, f'tabnet_{args.version}.zip')
            if not os.path.exists(path): path = os.path.join(model_dir, 'tabnet.zip')
            model.load_model(path.replace('.zip', '.pkl'))
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return

    # 4. Load Calibrator (New for Phase 13)
    logger.info("Loading Probability Calibrator...")
    calibrator = None
    calib_path = os.path.join(model_dir, 'calibrator.pkl')
    if os.path.exists(calib_path):
        from model.calibration import ProbabilityCalibrator
        calibrator = ProbabilityCalibrator()
        try:
            calibrator.load(calib_path)
        except Exception as e:
            logger.warning(f"Failed to load calibrator: {e}")
            calibrator = None
    else:
        logger.warning("Calibrator not found. Proceeding with raw probabilities.")

    # 5. Predict
    logger.info("Predicting...")
    # Feature selection
    feature_cols = None
    if args.model == 'lgbm' and hasattr(model.model, 'feature_name'):
        feature_cols = model.model.feature_name()
    elif args.model == 'catboost' and hasattr(model.model, 'feature_names_'):
        feature_cols = model.model.feature_names_
    
    # Fallback to dataset metadata
    if feature_cols is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                datasets = pickle.load(f)
            if datasets['train']['X'] is not None:
                feature_cols = datasets['train']['X'].columns.tolist()

    if feature_cols:
        missing = set(feature_cols) - set(processed_df.columns)
        for c in missing: processed_df[c] = 0
        X_pred = processed_df[feature_cols]
    else:
        # Fallback risky
        X_pred = processed_df.select_dtypes(include=[np.number])
    
    scores = model.predict(X_pred)
    processed_df['score'] = scores
    
    # Softmax Prob
    processed_df['prob'] = processed_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    
    # Calibrated Prob
    if calibrator:
        processed_df['calibrated_prob'] = calibrator.predict(processed_df['prob'].values)
        ev_prob = processed_df['calibrated_prob']
    else:
        processed_df['calibrated_prob'] = processed_df['prob']
        ev_prob = processed_df['prob']
        
    # Expected Value
    if 'odds' in processed_df.columns:
        processed_df['expected_value'] = ev_prob * processed_df['odds'].fillna(0)
    else:
        processed_df['expected_value'] = 0

    # Save Results
    output_dir = os.path.join(os.path.dirname(__file__), '../../experiments/predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, f"{target_date}_{args.model}.csv")
    
    # Output columns
    out_cols = [
        'race_id', 'horse_number', 'horse_name', 
        'score', 'prob', 'calibrated_prob', 'expected_value', 
        'odds', 'popularity', 'rank' # rank might be missing for future
    ]
    # Filter existing
    out_cols = [c for c in out_cols if c in processed_df.columns]
    
    processed_df[out_cols].to_csv(save_path, index=False)
    logger.info(f"Predictions saved to {save_path}")
    
    # Also print top picks
    print("\n--- Top Picks (EV Top 5) ---")
    picks = processed_df.sort_values('expected_value', ascending=False).head(5)
    print(picks[['race_id', 'horse_number', 'horse_name', 'calibrated_prob', 'odds', 'expected_value']])

if __name__ == "__main__":
    main()

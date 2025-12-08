
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import pickle
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from scipy.stats import entropy

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.lgbm import KeibaLGBM
from model.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_race_features(df):
    """
    レース単位の統計量特徴量を計算
    calibrated_prob があればそちらを優先使用する
    """
    race_feats = []
    
    prob_col = 'calibrated_prob' if 'calibrated_prob' in df.columns else 'prob'
    logger.info(f"Using '{prob_col}' for Feature Generation.")

    for race_id, group in df.groupby('race_id'):
        probs = group[prob_col].values
        odds = group['odds'].fillna(0).values
        
        ent = entropy(probs)
        odds_std = np.std(odds)
        max_prob = np.max(probs)
        sorted_probs = sorted(probs, reverse=True)
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
        n_horses = len(group)
        
        race_feats.append({
            'race_id': race_id,
            'entropy': ent,
            'odds_std': odds_std,
            'max_prob': max_prob,
            'confidence_gap': gap,
            'n_horses': n_horses
        })
    df_res = pd.DataFrame(race_feats)
    if df_res.empty:
        return pd.DataFrame(columns=['race_id', 'entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses'])
    return df_res

def create_targets(df):
    """
    正解ラベルを作成: 
    1. Win: モデルの1位指名が1着だったか
    2. Place: モデルの1位指名が3着以内だったか
    """
    targets = []
    
    for race_id, group in df.groupby('race_id'):
        if group['score'].isna().all(): continue
        
        # Best Score Horse
        best_horse = group.loc[group['score'].idxmax()]
        
        if pd.isna(best_horse['rank']):
            continue
            
        rank = int(best_horse['rank'])
        is_win = 1 if rank == 1 else 0
        is_place = 1 if rank <= 3 else 0
        
        targets.append({
            'race_id': race_id, 
            'target_win': is_win,
            'target_place': is_place
        })
            
    df_res = pd.DataFrame(targets)
    if df_res.empty:
        return pd.DataFrame(columns=['race_id', 'target_win', 'target_place'])
    return df_res

def train_and_save(train_data, valid_data, target_col, output_name, model_dir):
    """
    指定されたターゲットでモデルを学習・保存
    """
    features = ['entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses']
    
    logger.info(f"Training Model for Target: {target_col}")
    logger.info(f"Positive Ratio (Train): {train_data[target_col].mean():.4f}")
    
    lgb_train = lgb.Dataset(train_data[features], train_data[target_col])
    lgb_eval = lgb.Dataset(valid_data[features], valid_data[target_col], reference=lgb_train)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'is_unbalance': True,
    }
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
    
    bst = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_eval],
                    callbacks=callbacks)
    
    save_path = os.path.join(model_dir, output_name)
    with open(save_path, 'wb') as f:
        pickle.dump(bst, f)
    logger.info(f"Saved model to {save_path}")
    return bst

def main():
    parser = argparse.ArgumentParser(description='Train Betting Confidence Models (Win & Place)')
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data.parquet')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--base_model', type=str, default='lgbm')
    parser.add_argument('--model_version', type=str, default=None, help='Specific model version (e.g. v5_weighted)')
    parser.add_argument('--year_train', type=str, default='2019,2020,2021,2022,2023') # Expanded train years
    parser.add_argument('--year_valid', type=str, default='2024') 
    args = parser.parse_args()

    # Data Loading
    logger.info("Loading Data...")
    df = pd.read_parquet(args.input)
    if 'race_id' not in df.columns: df = df.reset_index()

    train_years = [int(y) for y in args.year_train.split(',')]
    valid_years = [int(y) for y in args.year_valid.split(',')]
    
    # Load Base Model
    logger.info(f"Loading Base Model: {args.base_model} {args.model_version or ''}")
    model = None
    model_path = None

    if args.base_model == 'lgbm':
        model = KeibaLGBM()
        if args.model_version:
             p = os.path.join(args.model_dir, f'lgbm_{args.model_version}.pkl')
             if os.path.exists(p): model_path = p
        
        if not model_path:
             model_path = os.path.join(args.model_dir, 'lgbm.pkl')
             
        model.load_model(model_path)
        
    elif args.base_model == 'catboost': # Added catboost support
        from model.catboost_model import KeibaCatBoost
        model = KeibaCatBoost()
        if args.model_version:
             p = os.path.join(args.model_dir, f'catboost_{args.model_version}.pkl')
             if os.path.exists(p): model_path = p
        
        if not model_path:
             model_path = os.path.join(args.model_dir, 'catboost.pkl')
        model.load_model(model_path)

    elif args.base_model == 'ensemble':
        model = EnsembleModel()
        model.load_model(os.path.join(args.model_dir, 'ensemble_model.pkl'))
        
    # Load Calibrator
    logger.info("Loading Calibrator...")
    calibrator = None
    calib_name = f'calibrator_{args.model_version}.pkl' if args.model_version else 'calibrator.pkl'
    calib_path = os.path.join(args.model_dir, calib_name)
    
    if os.path.exists(calib_path):
        from model.calibration import ProbabilityCalibrator
        calibrator = ProbabilityCalibrator()
        calibrator.load(calib_path)
        logger.info(f"Loaded {calib_path}")
    else:
        # Fallback
        if os.path.exists(os.path.join(args.model_dir, 'calibrator.pkl')):
            from model.calibration import ProbabilityCalibrator
            calibrator = ProbabilityCalibrator()
            calibrator.load(os.path.join(args.model_dir, 'calibrator.pkl'))
            logger.info("Loaded default calibrator.pkl (versioned one not found)")
    
    # Predict Scores
    feature_cols = None
    if args.base_model == 'lgbm' and hasattr(model.model, 'feature_name'):
        feature_cols = model.model.feature_name()
    
    if not feature_cols:
        X = df.select_dtypes(include=[np.number])
    else:
        for c in feature_cols: 
             if c not in df.columns: df[c] = 0
        X = df[feature_cols]

    logger.info("Predicting Base Scores...")
    df['score'] = model.predict(X)
    df['prob'] = df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    
    if calibrator:
        df['calibrated_prob'] = calibrator.predict(df['prob'].values)
        
    # Split
    train_df = df[df['year'].isin(train_years)].copy()
    valid_df = df[df['year'].isin(valid_years)].copy()
    
    if 'rank' not in train_df.columns:
        logger.error("Rank column missing from data!")
        return

    logger.info(f"Train Rows: {len(train_df)}, Valid Rows: {len(valid_df)}")
    
    # Create Targets
    logger.info("Creating Targets (Win & Place)...")
    target_train = create_targets(train_df)
    target_valid = create_targets(valid_df)
    
    if target_train.empty:
        logger.error("Target Train is empty.")
        return
        
    # Create Features
    logger.info("Creating Race Features...")
    feat_train = calculate_race_features(train_df)
    feat_valid = calculate_race_features(valid_df)
    
    # Merge
    train_data = pd.merge(feat_train, target_train, on='race_id')
    valid_data = pd.merge(feat_valid, target_valid, on='race_id')
    
    logger.info(f"Final Train Samples: {len(train_data)}")
    
    # Determine Output Names
    suffix = f"_{args.model_version}" if args.model_version else ""
    
    # Train Win Model
    train_and_save(train_data, valid_data, 'target_win', f'betting_model{suffix}_win.pkl', args.model_dir)
    
    # Train Place Model
    train_and_save(train_data, valid_data, 'target_place', f'betting_model{suffix}_place.pkl', args.model_dir)

if __name__ == "__main__":
    main()

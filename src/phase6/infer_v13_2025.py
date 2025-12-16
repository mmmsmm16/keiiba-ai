"""
V13 Market Residual Inference for 2025 Holdout
学習済みv13モデルを使って2025データに対して推論を実行

Usage:
    docker compose exec app python src/phase6/infer_v13_2025.py
"""

import sys
import os
import argparse
import logging
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.special import logit
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.race_filter import filter_jra_only

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safe logit transform"""
    p = np.clip(p, eps, 1 - eps)
    return logit(p)


def softmax_per_race(logits: np.ndarray, race_ids: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax per race (numerically stable)"""
    df = pd.DataFrame({'race_id': race_ids, 'logit': logits / temperature})
    
    # Subtract max for numerical stability
    df['logit_shifted'] = df.groupby('race_id')['logit'].transform(lambda x: x - x.max())
    df['exp'] = np.exp(df['logit_shifted'])
    df['softmax'] = df.groupby('race_id')['exp'].transform(lambda x: x / x.sum())
    
    return df['softmax'].values


def normalize_prob_per_race(probs: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Normalize probabilities per race to sum to 1"""
    df = pd.DataFrame({'race_id': race_ids, 'prob': probs})
    df['prob_norm'] = df.groupby('race_id')['prob'].transform(lambda x: x / x.sum())
    return df['prob_norm'].values


def load_feature_columns(dataset_path: str) -> List[str]:
    """Load feature column names from lgbm_datasets.pkl"""
    if not os.path.exists(dataset_path):
        return []
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data.get('feature_cols', [])


def prepare_features_for_inference(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare features for inference (same as training)"""
    df = df.copy()
    
    # Create p_market from odds
    if 'p_market' not in df.columns:
        df['p_market_raw'] = 1.0 / df['odds'].replace(0, np.nan)
        df['p_market'] = df.groupby('race_id')['p_market_raw'].transform(lambda x: x / x.sum())
    
    # Add market logit as a feature
    df['baseline_logit'] = safe_logit(df['p_market'].values)
    
    # Filter available feature columns
    available_features = [c for c in feature_cols if c in df.columns]
    
    # Exclude leak columns
    exclude = ['rank', 'time', 'raw_time', 'prize_money', 'prob', 'p_market',
               'prob_model_raw', 'prob_model_norm', 'is_winner', 'finish_time',
               'race_id', 'horse_id', 'horse_key', 'date', 'year', 'fold_year',
               'model_version', 'odds', 'p_market_raw', 'overround']
    
    safe_features = [c for c in available_features if c not in exclude and not c.startswith('prob_')]
    
    return df, safe_features


def load_models(model_dir: str) -> List[lgb.Booster]:
    """Load all fold models from directory"""
    models = []
    
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.txt')])
    logger.info(f"Found {len(model_files)} model files: {model_files}")
    
    for f in model_files:
        path = os.path.join(model_dir, f)
        model = lgb.Booster(model_file=path)
        models.append(model)
        logger.info(f"Loaded model: {f}")
    
    return models


def infer_2025(
    df: pd.DataFrame,
    models: List[lgb.Booster],
    feature_cols: List[str]
) -> pd.DataFrame:
    """Run inference on 2025 data using ensemble of fold models"""
    
    # Prepare features
    df, safe_features = prepare_features_for_inference(df, feature_cols)
    
    # Get feature names from first model (all models should have same features)
    model_features = models[0].feature_name()
    logger.info(f"Model expects {len(model_features)} features")
    
    # Check feature availability
    missing = [f for f in model_features if f not in df.columns]
    if missing:
        logger.warning(f"Missing {len(missing)} features: {missing[:10]}...")
        # Fill missing with 0
        for f in missing:
            df[f] = 0
    
    # Use model's feature list
    train_features = model_features
    logger.info(f"Inference with {len(train_features)} features, {len(df):,} rows")
    
    X = df[train_features].values
    
    # Ensemble prediction (average across models)
    preds = []
    for i, model in enumerate(models):
        pred = model.predict(X, num_iteration=model.best_iteration)
        preds.append(pred)
        logger.info(f"Model {i+1}: mean pred = {pred.mean():.4f}")
    
    # Average predictions
    pred_avg = np.mean(preds, axis=0)
    
    # Create result DataFrame
    result_df = df[['race_id', 'horse_id', 'date', 'odds']].copy()
    result_df['year'] = 2025
    result_df['p_market'] = df['p_market'].values
    result_df['prob_residual_raw'] = pred_avg
    
    # Store score_logit (from ensemble output)
    result_df['score_logit'] = safe_logit(pred_avg)
    
    # baseline_logit for delta calculation
    baseline = df['baseline_logit'].values
    result_df['delta_logit'] = result_df['score_logit'] - baseline
    
    # Method A: Normalize per race
    result_df['prob_residual_norm'] = normalize_prob_per_race(
        result_df['prob_residual_raw'].values,
        result_df['race_id'].values
    )
    
    # Method B: Softmax per race
    result_df['prob_residual_softmax'] = softmax_per_race(
        result_df['score_logit'].values,
        result_df['race_id'].values
    )
    
    result_df['model_version'] = 'v13_market_residual'
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description="V13 Model Inference for 2025")
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data_v11.parquet')
    parser.add_argument('--model_dir', type=str, default='models/v13_market_residual')
    parser.add_argument('--dataset_pkl', type=str, default='models/lgbm_datasets.pkl')
    parser.add_argument('--output', type=str, default='data/predictions/v13_market_residual_2025_infer.parquet')
    parser.add_argument('--jra_only', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    # Extract year
    df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    # Filter 2025 only
    df = df[df['year'] == 2025].copy()
    logger.info(f"2025 data: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # JRA filter
    if args.jra_only:
        df = filter_jra_only(df)
        logger.info(f"After JRA filter: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Load feature columns
    feature_cols = load_feature_columns(args.dataset_pkl)
    if not feature_cols:
        # Fallback: use all numeric columns
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.warning(f"Using fallback features: {len(feature_cols)} columns")
    
    # Load models
    models = load_models(args.model_dir)
    if not models:
        raise ValueError(f"No models found in {args.model_dir}")
    
    # Run inference
    result_df = infer_2025(df, models, feature_cols)
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result_df.to_parquet(args.output, index=False)
    logger.info(f"Saved predictions to {args.output}")
    logger.info(f"Output: {len(result_df):,} rows, columns: {result_df.columns.tolist()}")
    
    # Summary stats
    logger.info("Prediction summary:")
    logger.info(f"  prob_residual_softmax: mean={result_df['prob_residual_softmax'].mean():.4f}, std={result_df['prob_residual_softmax'].std():.4f}")
    logger.info(f"  prob_residual_norm: mean={result_df['prob_residual_norm'].mean():.4f}, std={result_df['prob_residual_norm'].std():.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

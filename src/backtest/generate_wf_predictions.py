"""
Phase 5 Prerequisite: Generate Walk-Forward OOF Predictions
年単位Walk-ForwardでOOF予測（prob）を生成し保存

Usage (in container):
    docker compose exec app python src/backtest/generate_wf_predictions.py --suffix _v11 --model lgbm
    docker compose exec app python src/backtest/generate_wf_predictions.py --suffix _v11 --model lgbm --eval_years 2024

Output:
    data/predictions/v12_wf.parquet
    data/derived/preprocessed_with_prob_v12.parquet
"""

import sys
import os
import argparse
import logging
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.calibration import ProbabilityCalibrator
from model.lgbm import KeibaLGBM
from utils.period_guard import validate_period

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WFPredictionGenerator:
    """Walk-Forward OOF予測生成器"""
    
    # Default WF Folds (expanding window)
    DEFAULT_FOLDS = [
        {
            'name': 'Fold_2021',
            'eval_year': 2021,
            'model_train_end': 2019,  # Train on 2014-2019
            'calib_year': 2020,
        },
        {
            'name': 'Fold_2022',
            'eval_year': 2022,
            'model_train_end': 2020,
            'calib_year': 2021,
        },
        {
            'name': 'Fold_2023',
            'eval_year': 2023,
            'model_train_end': 2021,
            'calib_year': 2022,
        },
        {
            'name': 'Fold_2024',
            'eval_year': 2024,
            'model_train_end': 2022,
            'calib_year': 2023,
        },
    ]
    
    def __init__(
        self,
        data_path: str,
        dataset_info_path: str,
        model_type: str = 'lgbm',
        model_params: dict = None,
        train_start_year: int = 2014
    ):
        self.data_path = data_path
        self.dataset_info_path = dataset_info_path
        self.model_type = model_type
        self.model_params = model_params or {}
        self.train_start_year = train_start_year
        
        self.df = None
        self.feature_cols = None
        self.predictions = []
    
    def load_data(self):
        """データとfeature_colsをロード"""
        logger.info(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        
        # year列を確認・追加
        if 'year' not in self.df.columns:
            if 'race_date' in self.df.columns:
                self.df['year'] = pd.to_datetime(self.df['race_date']).dt.year
            elif 'race_id' in self.df.columns:
                self.df['year'] = self.df['race_id'].astype(str).str[:4].astype(int)
        
        logger.info(f"Data loaded: {len(self.df):,} rows, years: {sorted(self.df['year'].unique())}")
        
        # Feature columns from dataset_info
        logger.info(f"Loading features from {self.dataset_info_path}...")
        with open(self.dataset_info_path, 'rb') as f:
            datasets = pickle.load(f)
            self.feature_cols = datasets['train']['X'].columns.tolist()
            del datasets
        
        logger.info(f"Features: {len(self.feature_cols)}")
    
    def _prepare_set(self, mask):
        """データセット準備"""
        df_subset = self.df.loc[mask].copy()
        df_subset = df_subset.sort_values('race_id')
        
        X = df_subset[self.feature_cols]
        y = (df_subset['rank'] == 1).astype(int) if 'rank' in df_subset.columns else pd.Series(0, index=df_subset.index)
        group_counts = df_subset.groupby('race_id', sort=False).size().tolist()
        
        return X, y, group_counts, df_subset
    
    def _train_model(self, X_train, y_train, g_train, X_valid, y_valid, g_valid):
        """モデル学習"""
        train_set = {'X': X_train, 'y': y_train, 'group': g_train}
        valid_set = {'X': X_valid, 'y': y_valid, 'group': g_valid}
        
        if self.model_type == 'lgbm':
            from model.lgbm import KeibaLGBM
            model = KeibaLGBM(params=self.model_params)
            model.train(train_set, valid_set)
            return model
        elif self.model_type == 'catboost':
            from model.catboost_model import KeibaCatBoost
            model = KeibaCatBoost(params=self.model_params)
            model.train(train_set, valid_set)
            return model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def generate_predictions(self, eval_years: list = None):
        """WF予測を生成"""
        folds = self.DEFAULT_FOLDS
        
        # eval_yearsが指定されていればフィルタ
        if eval_years:
            folds = [f for f in folds if f['eval_year'] in eval_years]
        
        logger.info(f"Generating predictions for {len(folds)} folds...")
        
        for fold in folds:
            eval_year = fold['eval_year']
            
            # 2025はブロック（Holdout）
            try:
                validate_period(eval_year, eval_year, allow_holdout=False)
            except ValueError as e:
                logger.warning(f"Skipping {eval_year}: {e}")
                continue
            
            logger.info("="*60)
            logger.info(f"Generating predictions for {fold['name']} (Eval: {eval_year})")
            
            # Masks
            train_mask = (self.df['year'] >= self.train_start_year) & (self.df['year'] <= fold['model_train_end'])
            calib_mask = self.df['year'] == fold['calib_year']
            eval_mask = self.df['year'] == eval_year
            
            if not train_mask.any() or not calib_mask.any() or not eval_mask.any():
                logger.warning(f"Skipping {fold['name']}: insufficient data")
                continue
            
            # Prepare datasets
            X_train, y_train, g_train, _ = self._prepare_set(train_mask)
            X_calib, y_calib, g_calib, _ = self._prepare_set(calib_mask)
            X_eval, y_eval, g_eval, df_eval = self._prepare_set(eval_mask)
            
            logger.info(f"  Train: {len(X_train):,} rows, Calib: {len(X_calib):,}, Eval: {len(X_eval):,}")
            
            # Train model
            logger.info("  Training model...")
            model = self._train_model(X_train, y_train, g_train, X_calib, y_calib, g_calib)
            
            # Get raw scores
            calib_scores = model.predict(X_calib)
            eval_scores = model.predict(X_eval)
            
            # Fit calibrator on calib set
            logger.info("  Fitting calibrator...")
            calibrator = ProbabilityCalibrator()
            calibrator.fit(calib_scores, y_calib)
            
            # Predict probabilities
            eval_probs = calibrator.predict(eval_scores)
            
            # Collect predictions
            pred_df = pd.DataFrame({
                'race_id': df_eval['race_id'].values,
                'horse_id': df_eval['horse_id'].values if 'horse_id' in df_eval.columns else df_eval.index.values,
                'umaban': df_eval['umaban'].values if 'umaban' in df_eval.columns else None,
                'year': eval_year,
                'fold_year': eval_year,
                'prob': eval_probs,
                'raw_score': eval_scores,
                'model_version': 'v12'
            })
            
            self.predictions.append(pred_df)
            logger.info(f"  Generated {len(pred_df):,} predictions")
        
        # Concatenate all predictions
        if self.predictions:
            all_preds = pd.concat(self.predictions, ignore_index=True)
            logger.info(f"Total predictions: {len(all_preds):,}")
            return all_preds
        else:
            logger.warning("No predictions generated")
            return pd.DataFrame()
    
    def save_predictions(self, predictions: pd.DataFrame, output_path: str):
        """予測を保存"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions.to_parquet(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    
    def merge_with_preprocessed(
        self,
        predictions: pd.DataFrame,
        output_path: str
    ):
        """元データに予測をマージして保存"""
        logger.info("Merging predictions with preprocessed data...")
        
        # 予測データの結合キーを準備
        merge_keys = ['race_id']
        if 'horse_id' in predictions.columns and 'horse_id' in self.df.columns:
            merge_keys.append('horse_id')
        elif 'umaban' in predictions.columns and 'umaban' in self.df.columns:
            merge_keys.append('umaban')
        
        # 予測を結合（必要なカラムだけ）
        pred_cols = merge_keys + ['prob', 'raw_score', 'fold_year', 'model_version']
        pred_subset = predictions[[c for c in pred_cols if c in predictions.columns]].copy()
        
        # 型を揃える
        for key in merge_keys:
            if key in self.df.columns and key in pred_subset.columns:
                pred_subset[key] = pred_subset[key].astype(self.df[key].dtype)
        
        # マージ
        merged = self.df.merge(pred_subset, on=merge_keys, how='left')
        
        # カバレッジ確認
        prob_coverage = merged['prob'].notna().mean() * 100
        logger.info(f"Probability coverage after merge: {prob_coverage:.2f}%")
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.to_parquet(output_path, index=False)
        logger.info(f"Merged data saved to {output_path} ({len(merged):,} rows)")
        
        return merged


def main():
    parser = argparse.ArgumentParser(
        description="Generate Walk-Forward OOF Predictions for Phase 5+"
    )
    parser.add_argument('--suffix', type=str, default='_v11', help='Dataset suffix')
    parser.add_argument('--model', type=str, default='lgbm', choices=['lgbm', 'catboost'])
    parser.add_argument(
        '--eval_years',
        type=int,
        nargs='+',
        default=[2021, 2022, 2023, 2024],
        help='Years to generate predictions for'
    )
    parser.add_argument('--output_dir', type=str, default='data/predictions')
    parser.add_argument('--derived_dir', type=str, default='data/derived')
    
    args = parser.parse_args()
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, f'data/processed/preprocessed_data{args.suffix}.parquet')
    dataset_info_path = os.path.join(base_dir, f'data/processed/lgbm_datasets{args.suffix}.pkl')
    
    # Load params
    params = {}
    params_path = os.path.join(base_dir, f'models/params/{args.model}_v1_best_params.json')
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        logger.info(f"Loaded params from {params_path}")
    
    # Generate
    generator = WFPredictionGenerator(
        data_path=data_path,
        dataset_info_path=dataset_info_path,
        model_type=args.model,
        model_params=params
    )
    generator.load_data()
    
    predictions = generator.generate_predictions(eval_years=args.eval_years)
    
    if len(predictions) > 0:
        # Save predictions
        pred_output = os.path.join(base_dir, args.output_dir, 'v12_wf.parquet')
        generator.save_predictions(predictions, pred_output)
        
        # Merge and save
        merged_output = os.path.join(base_dir, args.derived_dir, 'preprocessed_with_prob_v12.parquet')
        generator.merge_with_preprocessed(predictions, merged_output)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

import sys
import os
import pandas as pd
import numpy as np
import logging
import argparse
import pickle
import json
from datetime import datetime
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.calibration import ProbabilityCalibrator
from model.lgbm import KeibaLGBM
from model.catboost_model import KeibaCatBoost
from model.tabnet_model import KeibaTabNet
from betting.purchase_model import PurchaseModel
# from model.ensemble import EnsembleModel # Ensemble training might be complex in loop, focus on single models first or reuse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExpandingWindowValidator:
    def __init__(self, data_path, dataset_info_path, model_type='lgbm', model_params=None):
        self.data_path = data_path
        self.dataset_info_path = dataset_info_path
        self.model_type = model_type
        self.model_params = model_params or {}
        
        self.df = None
        self.feature_cols = None
        
        self.results = []

    def load_data(self):
        logger.info(f"Loading data from {self.data_path}...")
        try:
            self.df = pd.read_parquet(self.data_path)
            logger.info(f"Parquet loaded. Shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Failed to load parquet: {e}")
            raise

        # Load feature columns from dataset info (lgbm_datasets.pkl)
        logger.info(f"Loading feature metadata from {self.dataset_info_path}...")
        try:
            with open(self.dataset_info_path, 'rb') as f:
                # Optimized: simple load, might disable gc to speed up?
                datasets = pickle.load(f)
                if datasets['train']['X'] is not None:
                    self.feature_cols = datasets['train']['X'].columns.tolist()
                    logger.info(f"Features loaded: {len(self.feature_cols)}")
                    # Release memory
                    del datasets
                    import gc; gc.collect()
                else:
                    raise ValueError("Feature columns could not be loaded from dataset info.")
        except Exception as e:
            logger.error(f"Failed to load dataset info: {e}")
            raise
        
        logger.info(f"Data loading complete.")

    def run_cv(self):
        # Define Folds (Method A)
        folds = [
            # Fold 1
            {
                'name': 'Fold 1 (Eval 2021)',
                'train_years': [y for y in range(2014, 2021)], # 2014-2020 total train
                'model_train_years': [y for y in range(2014, 2020)], # 2014-2019
                'calib_years': [2020], # 2020
                'eval_years': [2021]
            },
            # Fold 2
            {
                'name': 'Fold 2 (Eval 2022)',
                'train_years': [y for y in range(2014, 2022)],
                'model_train_years': [y for y in range(2014, 2021)],
                'calib_years': [2021],
                'eval_years': [2022]
            },
            # Fold 3
            {
                'name': 'Fold 3 (Eval 2023)',
                'train_years': [y for y in range(2014, 2023)],
                'model_train_years': [y for y in range(2014, 2022)],
                'calib_years': [2022],
                'eval_years': [2023]
            }
        ]

        logger.info(f"Starting Expanding Window CV (Method A) with {len(folds)} folds...")
        
        for fold in folds:
            logger.info("="*60)
            logger.info(f"Running {fold['name']}")
            logger.info(f"  Model Train: {fold['model_train_years']}")
            logger.info(f"  Calib Fit:   {fold['calib_years']}")
            logger.info(f"  Evaluate:    {fold['eval_years']}")
            
            # Prepare Data (Sort by race_id is crucial for group creation)
            logger.info("  Preparing Datasets...")
            
            # Helper to prepare set
            def prepare_set(mask):
                df_subset = self.df.loc[mask].copy()
                # Ensure sorted by race_id for group calculation
                df_subset = df_subset.sort_values('race_id')
                
                X = df_subset[self.feature_cols]
                y = df_subset['rank'] == 1
                
                # Create group (counts per race_id)
                # Since it's sorted, we can use value_counts(sort=False) if we trust race_id order,
                # but it's safer to groupby.
                # LightGBM expects group to be list of counts in order of appearance.
                group_counts = df_subset.groupby('race_id', sort=False).size().tolist()
                
                return X, y, group_counts, df_subset['race_id'], df_subset['odds'].fillna(0)

            # Define masks
            train_mask = self.df['year'].isin(fold['model_train_years'])
            calib_mask = self.df['year'].isin(fold['calib_years'])
            eval_mask = self.df['year'].isin(fold['eval_years'])

            X_train, y_train, g_train, _, _ = prepare_set(train_mask)
            X_calib, y_calib, g_calib, _, _ = prepare_set(calib_mask)
            X_eval, y_eval, g_eval, eval_race_ids, eval_odds = prepare_set(eval_mask)
            
            # Train Base Model
            logger.info("  Training Base Model...")
            # We need to pass groups to _train_model
            model = self._train_model(X_train, y_train, g_train, X_calib, y_calib, g_calib) 
            
            # Get Raw Scores
            calib_scores = model.predict(X_calib)
            eval_scores = model.predict(X_eval)
            
            # Fit Calibrator
            logger.info("  Fitting Probability Calibrator...")
            calibrator = ProbabilityCalibrator()
            calibrator.fit(calib_scores, y_calib)
            
            # Predict Probabilities
            eval_probs = calibrator.predict(eval_scores)
            
            # Calculate Metrics
            metrics = self._calculate_metrics(y_eval, eval_probs, eval_scores)
            
            # Simple ROI Check (Top 1 by Prob)
            roi_metrics = self._calculate_roi(eval_race_ids, eval_probs, eval_odds, y_eval)
            metrics.update(roi_metrics)
            
            logger.info(f"  Result: LogLoss={metrics['log_loss']:.5f}, Brier={metrics['brier_score']:.5f}, AUC={metrics['auc']:.5f}, ROI={metrics['roi']:.2f}%")
            
            self.results.append({
                'fold': fold['name'],
                'metrics': metrics
            })
            
        self._print_summary()

    def _train_model(self, X_train, y_train, g_train, X_valid, y_valid, g_valid):
        # Wrap datasets
        train_set = {'X': X_train, 'y': y_train, 'group': g_train}
        valid_set = {'X': X_valid, 'y': y_valid, 'group': g_valid}
        
        if self.model_type == 'lgbm':
            model = KeibaLGBM(params=self.model_params)
            model.train(train_set, valid_set)
            return model
        elif self.model_type == 'catboost':
            model = KeibaCatBoost(params=self.model_params)
            model.train(train_set, valid_set)
            return model
        elif self.model_type == 'tabnet':
             model = KeibaTabNet(params=self.model_params)
             model.train(train_set, valid_set)
             return model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _calculate_metrics(self, y_true, y_prob, y_score):
        ll = log_loss(y_true, y_prob)
        bs = brier_score_loss(y_true, y_prob)
        try:
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = 0.5
            
        return {
            'log_loss': ll,
            'brier_score': bs,
            'auc': auc
        }

    def _calculate_roi(self, race_ids, probs, odds, y_true):
        # Create temp dataframe
        df = pd.DataFrame({
            'race_id': race_ids.values,
            'prob': probs,
            'odds': odds.values,
            'hit': y_true.values
        })
        
        pm = PurchaseModel()
        
        # 1. Market Prob & EV
        df = pm.calculate_market_probability(df)
        df = pm.calculate_expected_value(df)
        
        metrics = {}
        
        # Strategy A: Naive Top 1 (Existing)
        picks_naive = df.loc[df.groupby('race_id')['prob'].idxmax()]
        bet_sum = len(picks_naive) * 100
        return_sum = picks_naive[picks_naive['hit'] == 1]['odds'].sum() * 100
        metrics['roi_naive'] = return_sum / bet_sum * 100 if bet_sum > 0 else 0
        
        # Strategy B: Kelly (Fraction=0.1, Bankroll=10000, Max=5%)
        # Note: In backtest, we sum returns. 
        # For simplicity here, we assume fixed bankroll per race batch or just sum amounts.
        # Ideally, bankroll updates, but for simple WF metric, we sum absolute amounts.
        df_kelly = pm.apply_betting_strategy(df, strategy_name='kelly', bankroll=10000, fraction=0.1)
        bet_sum_kelly = df_kelly['bet_amount'].sum()
        return_sum_kelly = (df_kelly[df_kelly['hit'] == 1]['bet_amount'] * df_kelly[df_kelly['hit'] == 1]['odds']).sum()
        metrics['roi_kelly'] = return_sum_kelly / bet_sum_kelly * 100 if bet_sum_kelly > 0 else 0
        metrics['bets_kelly'] = (df_kelly['bet_amount'] > 0).sum()
        
        # Strategy C: Flat Bet on EV > 0
        df_flat = pm.apply_betting_strategy(df, strategy_name='flat', threshold=0.0, bet_amount=100)
        bet_sum_flat = df_flat['bet_amount'].sum()
        return_sum_flat = (df_flat[df_flat['hit'] == 1]['bet_amount'] * df_flat[df_flat['hit'] == 1]['odds']).sum()
        metrics['roi_ev_flat'] = return_sum_flat / bet_sum_flat * 100 if bet_sum_flat > 0 else 0
        metrics['bets_ev_flat'] = (df_flat['bet_amount'] > 0).sum()
        
        # Use Naive for the main logger for now, but return all
        metrics['roi'] = metrics['roi_naive']
        
        return metrics

    def _print_summary(self):
        logger.info("="*60)
        logger.info("Walk-Forward Validation Summary")
        logger.info("="*60)
        
        df_res = pd.DataFrame([
            {
                'Fold': r['fold'],
                'LogLoss': r['metrics']['log_loss'],
                'Brier': r['metrics']['brier_score'],
                'AUC': r['metrics']['auc'],
                'ROI(Naive)': r['metrics']['roi_naive'],
                'ROI(Kelly)': r['metrics']['roi_kelly'],
                'ROI(EV>0)': r['metrics']['roi_ev_flat']
            } for r in self.results
        ])
        
        # Calculate Mean
        mean_ll = df_res['LogLoss'].mean()
        mean_roi = df_res['ROI(Naive)'].mean()
        mean_kelly = df_res['ROI(Kelly)'].mean()
        
        print(df_res.to_string(index=False))
        print("-" * 60)
        print(f"Mean LogLoss:   {mean_ll:.5f}")
        print(f"Mean ROI(Naive): {mean_roi:.2f}%")
        print(f"Mean ROI(Kelly): {mean_kelly:.2f}%")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Validation (Expanding Window)')
    parser.add_argument('--model', type=str, default='lgbm', choices=['lgbm', 'catboost', 'tabnet'], help='Model type')
    parser.add_argument('--suffix', type=str, default='', help='Dataset suffix')
    parser.add_argument('--params_version', type=str, default='v1', help='Parameter version to load (if available)')
    args = parser.parse_args()
    
    # Paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, f'../../data/processed/preprocessed_data{args.suffix}.parquet')
    dataset_info_path = os.path.join(base_dir, f'../../data/processed/lgbm_datasets{args.suffix}.pkl')
    
    # Load Params (Optional)
    params = {}
    param_path = os.path.join(base_dir, f'../../models/params/{args.model}_{args.params_version}_best_params.json')
    if os.path.exists(param_path):
        with open(param_path, 'r') as f:
            params = json.load(f)
        logger.info(f"Loaded params from {param_path}")
    
    validator = ExpandingWindowValidator(data_path, dataset_info_path, model_type=args.model, model_params=params)
    validator.load_data()
    validator.run_cv()

if __name__ == "__main__":
    main()

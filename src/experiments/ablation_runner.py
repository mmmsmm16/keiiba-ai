import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
import json
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.validation import ExpandingWindowValidator
from model.lgbm import KeibaLGBM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AblationRunner:
    def __init__(self, data_path, dataset_info_path, params_path=None):
        self.data_path = data_path
        self.dataset_info_path = dataset_info_path
        self.params = {}
        if params_path and os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.params = json.load(f)
        
        self.full_features = self._load_feature_list()
        
    def _load_feature_list(self):
        with open(self.dataset_info_path, 'rb') as f:
            data = pickle.load(f)
        return data['train']['X'].columns.tolist()

    def get_feature_subset(self, drop_groups=None):
        """
        指定したグループの特徴量を削除したリストを返す
        """
        if not drop_groups:
            return self.full_features
            
        features = self.full_features.copy()
        
        # Simple string matching for groups (customize as needed)
        drop_list = []
        for feat in features:
            for group in drop_groups:
                # Group logic (consistent with list_features.py)
                is_hit = False
                if group == 'embedding' and '_emb_' in feat: is_hit = True
                if group == 'bloodline' and ('sire' in feat or 'blood' in feat or 'bms' in feat): is_hit = True
                if group == 'human' and ('jockey' in feat or 'trainer' in feat): is_hit = True
                if group == 'speed' and ('speed' in feat or 'time' in feat or '3f' in feat): is_hit = True
                if group == 'realtime' and 'trend' in feat: is_hit = True
                
                if is_hit:
                    drop_list.append(feat)
                    break
        
        final_features = [f for f in features if f not in drop_list]
        logger.info(f"Dropped {len(drop_list)} features (Group: {drop_groups}). Remaining: {len(final_features)}")
        return final_features

    def run_screening(self, drop_groups=None):
        """
        Screening Mode: Fixed Valid 2024
        Train: 2014-2023, Valid: 2024
        """
        logger.info(f"Running Ablation Screening (Fixed Valid 2024). Drop: {drop_groups}")
        features = self.get_feature_subset(drop_groups)
        
        # Load Data
        df = pd.read_parquet(self.data_path)
        
        # Split (Screening definition)
        # Train: ~2023 (including 2023)
        # Valid: 2024
        train_mask = df['year'].between(2014, 2023)
        valid_mask = df['year'] == 2024
        
        if valid_mask.sum() == 0:
            logger.warning("No data for 2024. Using 2023 as valid for testing.")
            train_mask = df['year'].between(2014, 2022)
            valid_mask = df['year'] == 2023
            
        X_train = df.loc[train_mask, features]
        y_train = (df.loc[train_mask, 'rank'] == 1)
        
        X_valid = df.loc[valid_mask, features]
        y_valid = (df.loc[valid_mask, 'rank'] == 1)
        
        # Training (Screening uses LightGBM regression/ranking with early stopping)
        # Prepare datasets
        # Note: No 'group' for simple screening unless we strictly want ranking. 
        # Using binary classification (logloss) for speed/proxy is often fine for screening.
        # But to match baseline, let's use the same model class.
        
        # For simple screening, we might skip full ranking setup if it's slow, 
        # but let's stick to consistency if possible.
        # Sort for group
        def prepare_ranking_set(df_subset, features):
            df_subset = df_subset.sort_values('race_id')
            X = df_subset[features]
            y = (df_subset['rank'] == 1)
            group = df_subset.groupby('race_id', sort=False).size().tolist()
            return X, y, group

        X_train, y_train, g_train = prepare_ranking_set(df.loc[train_mask], features)
        X_valid, y_valid, g_valid = prepare_ranking_set(df.loc[valid_mask], features)
        
        train_set = {'X': X_train, 'y': y_train, 'group': g_train}
        valid_set = {'X': X_valid, 'y': y_valid, 'group': g_valid}
        
        model = KeibaLGBM(params=self.params)
        model.train(train_set, valid_set)
        
        # Evaluation
        valid_scores = model.predict(X_valid)
        
        # Calc LogLoss
        from sklearn.metrics import log_loss, roc_auc_score
        ll = log_loss(y_valid, valid_scores) # Note: scores might need sigmoid if using raw prediction? 
        # KeibaLGBM predict returns raw scores for LambdaRank? No, predict returns... 
        # lgb.predict returns margins depending on objective.
        # LambdaRank predicts "scores" (not probability). We cannot calc LogLoss directly from Ranking scores without calibration.
        # BUT, KeibaLGBM might be configured for 'binary' if we want LogLoss?
        # Check KeibaLGBM params. Default is 'lambdarank'.
        
        # For LambdaRank, we should check ndcg or use a calibrator to get probs.
        # Let's use AUC for screening as it's scale-invariant.
        auc = roc_auc_score(y_valid, valid_scores)
        
        logger.info(f"Screening Result: AUC={auc:.5f}")
        return {'auc': auc}

    def run_verification(self, drop_groups=None):
        """
        Verification Mode: Walk-Forward (Reuse validation.py logic)
        """
        logger.info(f"Running Ablation Verification (Walk-Forward). Drop: {drop_groups}")
        features = self.get_feature_subset(drop_groups)
        
        # Monkey patch Validator's feature loader or pass features
        # ExpandingWindowValidator loads features from pickle. We need to override.
        validator = ExpandingWindowValidator(self.data_path, self.dataset_info_path, model_params=self.params)
        
        # Override features
        validator.feature_cols = features # This needs to be set AFTER load_data normally, but Validator loads in load_data.
        # We can subclass or modify validator logic.
        # Simpler: Load data, then overwrite cols
        validator.load_data()
        validator.feature_cols = features
        
        validator.run_cv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['screening', 'verification'], default='screening')
    parser.add_argument('--drop', nargs='+', help='Groups to drop (e.g. embedding speed)')
    parser.add_argument('--suffix', default='_v11')
    args = parser.parse_args()
    
    data_path = f'data/processed/preprocessed_data{args.suffix}.parquet'
    info_path = f'data/processed/lgbm_datasets{args.suffix}.pkl'
    params_path = 'models/params/lgbm_v1_best_params.json'
    
    runner = AblationRunner(data_path, info_path, params_path)
    
    if args.mode == 'screening':
        runner.run_screening(args.drop)
    else:
        runner.run_verification(args.drop)

if __name__ == "__main__":
    main()

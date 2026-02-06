import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from scipy.special import expit

logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/v13_market_residual'))
PARQUET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/preprocessed_data.parquet'))

class ModelWrapper:
    def __init__(self):
        self.models = self._load_models()
        self.features_cache = None
        
    def _load_models(self):
        models = []
        for fold in ['2022', '2023', '2024']:
            path = os.path.join(MODEL_DIR, f'v13_fold_{fold}.txt')
            if os.path.exists(path):
                models.append(lgb.Booster(model_file=path))
                logger.info(f"Loaded Model: {path}")
        if not models:
            logger.warning(f"No models found in {MODEL_DIR}. Mocking might be needed if files missing.")
        return models

    def _get_features(self, race_id: str) -> pd.DataFrame:
        if self.features_cache is None:
            if os.path.exists(PARQUET_PATH):
                logger.info(f"Loading Feature Parquet: {PARQUET_PATH}")
                self.features_cache = pd.read_parquet(PARQUET_PATH)
            else:
                logger.error(f"Feature Parquet not found: {PARQUET_PATH}")
                return pd.DataFrame()
        
        # Filter by race_id
        # race_id in parquet might be string or int? usually string.
        # Ensure string comparison
        df = self.features_cache[self.features_cache['race_id'].astype(str) == str(race_id)].copy()
        return df

    def predict(self, race_id: str, snapshot_odds: dict) -> dict:
        """
        Returns structure expected by StrategyEngine:
        {
            'metrics': {'p1': float, 'margin': float, 'entropy': float},
            'horses': [
                {'horse_number': int, 'horse_name': str, 'p_cal': float}, ...
            ]
        }
        """
        df = self._get_features(race_id)
        if df.empty:
            logger.warning(f"No features for race {race_id}")
            return {}

        if not self.models:
            return {}

        # 1. Inject Odds (for market prob calculation if needed, or feature)
        # Some models use odds as feature.
        # Map odds (tansho) to 'odds' column
        # snapshot_odds['tansho'] -> {h_num: odds}
        tansho_map = snapshot_odds.get('tansho', {})
        if tansho_map:
            df['odds'] = df['horse_number'].map(tansho_map)
            df['odds'].fillna(0, inplace=True) # Or some default?
            
            # Recalculate popularity?
            # df['popularity'] = df['odds'].rank(method='min')
            
        # 2. Predict
        feature_cols = self.models[0].feature_name()
        
        # Missing cols
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0
                
        X = df[feature_cols].fillna(0)
        
        preds = []
        for m in self.models:
            preds.append(m.predict(X))
        avg_pred = np.mean(preds, axis=0) # Logit
        
        # 3. Softmax (Prob Cal)
        # V13 methodology: prob = softmax(logit) (?)
        # Or prob = sigmoid(logit)?
        # auto_predict_v13.py uses:
        #   df['prob_residual_raw'] = expit(avg_pred)
        #   softmax_race...
        
        raw_prob = expit(avg_pred)
        
        # Softmax per race
        # exp(logit - max) / sum
        exp_vals = np.exp(avg_pred - np.max(avg_pred))
        p_cal = exp_vals / np.sum(exp_vals)
        
        df['p_cal'] = p_cal
        
        # 4. Calculate Metrics
        df_sorted = df.sort_values('p_cal', ascending=False)
        
        p_values = df_sorted['p_cal'].values
        p1 = p_values[0] if len(p_values) > 0 else 0
        p2 = p_values[1] if len(p_values) > 1 else 0
        margin = p1 - p2
        
        # Entropy
        # -sum(p * log(p))
        entropy = -np.sum(p_values * np.log(p_values + 1e-9))
        
        metrics = {
            'p1': float(p1),
            'margin': float(margin),
            'entropy': float(entropy)
        }
        
        # 5. Format Horses
        horses = []
        for _, row in df_sorted.iterrows():
            horses.append({
                'horse_number': int(row['horse_number']),
                'horse_name': str(row.get('horse_name', f"H{int(row['horse_number'])}")),
                'p_cal': float(row['p_cal'])
            })
            
        return {
            'metrics': metrics,
            'horses': horses
        }

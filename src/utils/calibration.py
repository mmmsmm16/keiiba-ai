"""
Calibration Utilities
校正手法のユーティリティ（Temperature / Isotonic / Full Beta）

Usage:
    from utils.calibration import FullBetaCalibrator, TemperatureScaling, IsotonicCalibrator
"""

import numpy as np
import pickle
import os
import logging
from typing import Tuple, Optional
from scipy.special import expit, logit
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """安全なlogit変換"""
    p = np.clip(p, eps, 1 - eps)
    return logit(p)


def safe_log(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """安全なlog変換"""
    p = np.clip(p, eps, 1.0)
    return np.log(p)


class TemperatureScaling:
    """温度スケーリング校正"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.optimal_T = 1.0
        self.is_fitted = False
    
    def fit(self, probs: np.ndarray, y_true: np.ndarray, 
            search_range: Tuple[float, float] = (0.1, 5.0)) -> 'TemperatureScaling':
        """最適温度を探索"""
        logits = safe_logit(probs)
        
        def nll(T):
            scaled_probs = expit(logits / T)
            scaled_probs = np.clip(scaled_probs, 1e-15, 1 - 1e-15)
            return log_loss(y_true, scaled_probs)
        
        result = minimize_scalar(nll, bounds=search_range, method='bounded')
        self.optimal_T = result.x
        self.is_fitted = True
        logger.info(f"Temperature fitted: T={self.optimal_T:.4f}")
        return self
    
    def predict(self, probs: np.ndarray) -> np.ndarray:
        """温度スケーリング適用"""
        logits = safe_logit(probs)
        return expit(logits / self.optimal_T)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'optimal_T': self.optimal_T}, f)
    
    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.optimal_T = data['optimal_T']
            self.is_fitted = True
        return True


class IsotonicCalibrator:
    """Isotonic Regression校正"""
    
    def __init__(self):
        self.iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        self.is_fitted = False
    
    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> 'IsotonicCalibrator':
        self.iso_reg.fit(probs, y_true)
        self.is_fitted = True
        logger.info("Isotonic calibrator fitted")
        return self
    
    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self.iso_reg.predict(probs)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.iso_reg, f)
    
    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            self.iso_reg = pickle.load(f)
            self.is_fitted = True
        return True


class FullBetaCalibrator:
    """
    Full Beta Calibration
    
    p' = sigmoid(a * log(p) + b * log(1-p) + c)
    
    Parameters:
        a: log(p) coefficient
        b: log(1-p) coefficient  
        c: intercept
    """
    
    def __init__(self, eps: float = 1e-6):
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
        self.eps = eps
        self.is_fitted = False
    
    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            bounds: Optional[Tuple] = None) -> 'FullBetaCalibrator':
        """
        最尤推定でパラメータを学習
        
        Args:
            probs: モデル確率 (0,1)
            y_true: 正解ラベル (0/1)
            bounds: パラメータ境界 ((a_lo,a_hi), (b_lo,b_hi), (c_lo,c_hi))
        """
        if bounds is None:
            bounds = [(0.01, 10.0), (0.01, 10.0), (-5.0, 5.0)]
        
        # クリップ
        p = np.clip(probs, self.eps, 1 - self.eps)
        log_p = np.log(p)
        log_1mp = np.log(1 - p)
        
        def nll(params):
            a, b, c = params
            logits = a * log_p + b * log_1mp + c
            calibrated = expit(logits)
            calibrated = np.clip(calibrated, 1e-15, 1 - 1e-15)
            return log_loss(y_true, calibrated)
        
        # 初期値
        x0 = [1.0, 1.0, 0.0]
        
        result = minimize(nll, x0, method='L-BFGS-B', bounds=bounds)
        
        self.a, self.b, self.c = result.x
        self.is_fitted = True
        
        logger.info(f"Full Beta fitted: a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}")
        return self
    
    def predict(self, probs: np.ndarray) -> np.ndarray:
        """校正適用"""
        p = np.clip(probs, self.eps, 1 - self.eps)
        log_p = np.log(p)
        log_1mp = np.log(1 - p)
        
        logits = self.a * log_p + self.b * log_1mp + self.c
        calibrated = expit(logits)
        
        return np.clip(calibrated, self.eps, 1 - self.eps)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'a': self.a, 'b': self.b, 'c': self.c}, f)
    
    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.a = data['a']
            self.b = data['b']
            self.c = data['c']
            self.is_fitted = True
        return True


class MarketBlend:
    """
    市場確率とモデル確率のブレンド
    
    p_blend = (1-λ) * p_market + λ * p_model
    """
    
    def __init__(self, lambda_: float = 0.5):
        self.lambda_ = lambda_
    
    def blend(self, p_market: np.ndarray, p_model: np.ndarray,
              normalize_per_race: bool = True,
              race_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        確率をブレンド
        
        Args:
            p_market: 市場確率
            p_model: モデル確率
            normalize_per_race: レース内正規化
            race_ids: レースID（正規化用）
        
        Returns:
            ブレンド確率
        """
        p_blend = (1 - self.lambda_) * p_market + self.lambda_ * p_model
        
        if normalize_per_race and race_ids is not None:
            import pandas as pd
            df = pd.DataFrame({'race_id': race_ids, 'p_blend': p_blend})
            race_sum = df.groupby('race_id')['p_blend'].transform('sum')
            p_blend = np.where(race_sum > 0, p_blend / race_sum, p_blend)
        
        return np.clip(p_blend, 1e-15, 1 - 1e-15)


def create_market_prob(odds: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """
    オッズから市場確率を作成（レース内正規化）
    
    Args:
        odds: 単勝オッズ
        race_ids: レースID
    
    Returns:
        正規化された市場確率
    """
    import pandas as pd
    
    df = pd.DataFrame({'race_id': race_ids, 'odds': odds})
    df['p_raw'] = 1.0 / df['odds'].replace(0, np.nan)
    df['p_market'] = df.groupby('race_id')['p_raw'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else np.nan
    )
    
    return df['p_market'].values


def normalize_prob_per_race(probs: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """
    確率をレース内正規化
    
    Args:
        probs: 確率
        race_ids: レースID
    
    Returns:
        正規化確率（sum=1 per race）
    """
    import pandas as pd
    
    df = pd.DataFrame({'race_id': race_ids, 'prob': probs})
    race_sum = df.groupby('race_id')['prob'].transform('sum')
    normalized = np.where(race_sum > 0, probs / race_sum, np.nan)
    
    return normalized

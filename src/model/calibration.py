
import numpy as np
import pickle
import os
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    """
    モデルの出力スコアを実際の確率に補正するクラス (Isotonic Regression)
    """
    def __init__(self):
        self.iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        self.is_fitted = False

    def fit(self, scores: np.ndarray, y_true: np.ndarray):
        """
        キャリブレーションモデルを学習する
        Args:
            scores (np.ndarray): モデルの予測スコア (1d array)
            y_true (np.ndarray): 正解ラベル (0 or 1)
        """
        logger.info(f"Fitting Calibrator with {len(scores)} samples...")
        self.iso_reg.fit(scores, y_true)
        self.is_fitted = True
        logger.info("Calibration model fitted.")

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        スコアを確率に変換する
        """
        if not self.is_fitted:
            logger.warning("Calibrator is not fitted yet. Returning raw scores (clipped 0-1).")
            return np.clip(scores, 0, 1) # Fallback if naive
        
        return self.iso_reg.predict(scores)

    def save(self, path):
        """モデルを保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.iso_reg, f)
        logger.info(f"Calibrator saved to {path}")

    def load(self, path):
        """モデルをロード"""
        if not os.path.exists(path):
            logger.error(f"Calibrator file not found: {path}")
            return False
            
        with open(path, 'rb') as f:
            self.iso_reg = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Calibrator loaded from {path}")
        return True

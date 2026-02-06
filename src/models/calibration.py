import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.special import logit, expit

logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    """
    確率校正（Calibration）を行うクラス。
    Platt Scaling (Logistic Regression) および Isotonic Regression をサポート。
    """
    def __init__(self, method='platt'):
        self.method = method
        self.model = None
        
    def fit(self, y_true, y_prob):
        """
        校正器を学習する。
        
        Args:
            y_true: 二値のターゲット (0 or 1)
            y_prob: モデルが出力した確率スコア (0~1)
        """
        # スコアをロジットに変換 (Platt Scaling用)
        # 0や1に近い値をクリップして計算安定性を高める
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        y_logit = logit(y_prob_clipped).reshape(-1, 1)
        
        if self.method == 'platt':
            # Logistic Regression on Logits
            logger.info("Fitting Platt Scaling calibrator...")
            self.model = LogisticRegression(C=1.0, solver='lbfgs')
            self.model.fit(y_logit, y_true)
        elif self.method == 'isotonic':
            logger.info("Fitting Isotonic Regression calibrator...")
            self.model = IsotonicRegression(out_of_bounds='clip')
            self.model.fit(y_prob, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
            
    def predict(self, y_prob):
        """
        確率を校正する。
        
        Args:
            y_prob: 元の確率スコア
            
        Returns:
            校正後の確率スコア
        """
        if self.model is None:
            logger.warning("Calibrator is not fitted yet. Returning raw probabilities.")
            return y_prob
            
        if self.method == 'platt':
            y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
            y_logit = logit(y_prob_clipped).reshape(-1, 1)
            return self.model.predict_proba(y_logit)[:, 1]
        elif self.method == 'isotonic':
            return self.model.predict(y_prob)
        else:
            return y_prob

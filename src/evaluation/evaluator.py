import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, y_true, y_prob, group_ids=None, odds=None, threshold=None, is_probability=True):
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true (array-like): Binary targets (0/1).
            y_prob (array-like): Predicted probabilities or scores.
            group_ids (array-like, optional): Race IDs for Top1/Group metrics.
            odds (array-like, optional): Odds for ROI calculation.
            threshold (float, optional): Threshold for binary classification metrics.
            is_probability (bool): If False, skip logloss/brier or normalize scores.
            
        Returns:
            dict: Dictionary containing metric names and values.
        """
        metrics = {}
        
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        
        # 1. Standard Metrics (only for probability predictions)
        if is_probability and y_prob.max() <= 1.0 and y_prob.min() >= 0.0:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['auc'] = np.nan
                
            try:
                metrics['logloss'] = log_loss(y_true, y_prob)
                metrics['brier'] = brier_score_loss(y_true, y_prob)
            except ValueError:
                metrics['logloss'] = np.nan
                metrics['brier'] = np.nan
        else:
            # For regression scores, calculate AUC but skip logloss/brier
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['auc'] = np.nan
            metrics['logloss'] = np.nan
            metrics['brier'] = np.nan
        
        # 2. Calibration (ECE) - only for probability predictions
        if is_probability and y_prob.max() <= 1.0:
            metrics['ece'] = self._expected_calibration_error(y_true, y_prob)
        else:
            metrics['ece'] = np.nan
        
        # 3. Group-wise Metrics (Top1 Hit Rate)
        if group_ids is not None:
            # Efficient Top1 Calculation
            top1_hit = self._calc_top1_hit_rate(y_true, y_prob, group_ids)
            metrics['top1_precision'] = top1_hit
            
        # 4. ROI (Flat bet on Top1)
        if group_ids is not None and odds is not None:
            roi, profit, ret = self._calc_top1_roi(y_true, y_prob, odds, group_ids)
            metrics['roi_top1_flat'] = roi
            metrics['return_top1_flat'] = ret
            
        return metrics

    def _expected_calibration_error(self, y_true, y_prob, n_bins=10):
        """
        Calculate Expected Calibration Error (ECE).
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        # Ensure numpy arrays
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

    def _calc_top1_hit_rate(self, y_true, y_prob, group_ids):
        """
        Calculate precision@1 (Top1 Hit Rate) per race.
        """
        df = pd.DataFrame({
            'y_true': y_true, 
            'y_prob': y_prob, 
            'race_id': group_ids
        })
        
        # Find index of max prob per race
        # Note: In case of ties, idxmax returns the first occurrence.
        # Ideally we should handle ties, but for continuous probs simple argmax is usually fine.
        idx_max = df.groupby('race_id')['y_prob'].idxmax()
        
        # Check success
        # Use .loc with the indices of max prob rows
        top1_wins = df.loc[idx_max, 'y_true'].sum()
        n_races = len(idx_max)
        
        return top1_wins / n_races if n_races > 0 else 0.0

    def _calc_top1_roi(self, y_true, y_prob, odds, group_ids):
        """
        Calculate ROI if we bet 1 unit on the Top1 horse in every race.
        """
        df = pd.DataFrame({
            'y_true': y_true, 
            'y_prob': y_prob, 
            'odds': odds,
            'race_id': group_ids
        })
        
        # Drop rows with NaN odds if any (safe guard)
        # However, for ROI calc, if we bet on a horse with missing odds, we assume return 0 or skip?
        # Usually valid data has odds. Assuming 0 for NaN odds.
        df['odds'] = df['odds'].fillna(0.0)
        
        # Identify Top1
        idx_max = df.groupby('race_id')['y_prob'].idxmax()
        
        total_bet = len(idx_max) * 100 # 100 yen per race
        
        # Calculate return
        # Return = 100 * odds IF y_true==1 ELSE 0
        hits = df.loc[idx_max]
        total_return = (hits['y_true'] * hits['odds'] * 100).sum()
        
        profit = total_return - total_bet
        roi = (total_return / total_bet) * 100 if total_bet > 0 else 0.0
        
        return roi, profit, total_return

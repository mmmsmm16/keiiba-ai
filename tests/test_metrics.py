
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Adjust path to import from scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming the functions are in scripts/phase_m/compare_decay_models.py
# We might need to extract them or import them. 
# For now, I will define the logic here or try to import if possible.
# Ideally refactor `compare_decay_models.py` to be importable, or move metrics to `src/utils/metrics.py`.
# Given the constraints, I will test the logic by copying the implementation or refactoring.
# Let's try to import from the script (it needs __init__.py or sys.path hack).

from scripts.phase_m.compare_decay_models import compute_ndcg, compute_recall_at_k

class TestMetrics(unittest.TestCase):
    def test_ndcg_ordering(self):
        """
        Verify that NDCG changes when the prediction order changes.
        """
        # Case 1: Perfect ordering
        # Rank 1 horse has highest score
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_score_perfect = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
        
        ndcg_perfect = compute_ndcg(y_true, y_score_perfect, k=5)
        
        # Case 2: Worst ordering
        # Rank 1 horse has lowest score
        y_score_worst = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        ndcg_worst = compute_ndcg(y_true, y_score_worst, k=5)
        
        print(f"NDCG Perfect: {ndcg_perfect}, NDCG Worst: {ndcg_worst}")
        
        self.assertGreater(ndcg_perfect, ndcg_worst)
        self.assertAlmostEqual(ndcg_perfect, 1.0, places=4) # Ideally 1.0 if IDCG is calculated correctly
        
    def test_recall_at_k(self):
        """
        Verify Recall@5 behavior.
        """
        # True ranks: 1, 2, 3, 4, 5...
        y_true = pd.Series([1, 6, 7, 8, 9, 10]) # Winner is index 0
        
        # Scenario A: Winner is in top 5 predictions
        y_score_hit = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
        recall_hit = compute_recall_at_k(y_true, y_score_hit, k=5, target='win')
        self.assertEqual(recall_hit, 1.0)
        
        # Scenario B: Winner is NOT in top 5
        y_score_miss = pd.Series([0.1, 0.8, 0.7, 0.6, 0.5, 0.9])
        recall_miss = compute_recall_at_k(y_true, y_score_miss, k=5, target='win')
        self.assertEqual(recall_miss, 0.0)

if __name__ == '__main__':
    unittest.main()

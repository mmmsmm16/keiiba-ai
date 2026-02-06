import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.scripts.auto_predict_v13 import V13Predictor

class TestV13PredictorGuard:
    def test_predict_race_raises_on_forbidden_columns(self):
        """
        Verify that predict_race raises an error if forbidden leak-prone columns
        are present in the input DataFrame and NOT explicitly handled/overwritten/dropped.
        OR, if the implementation decides to use an allowlist, check that it works.
        
        The requirement is: "Input features using allowlist... forbidden columns raise exception".
        """
        predictor = V13Predictor()
        predictor.models = [MagicMock()] # Mock model
        predictor.models[0].feature_name.return_value = ['authorized_feature']
        
        # DataFrame with FORBIDDEN column 'rank' (final rank)
        df = pd.DataFrame({
            'horse_number': [1, 2],
            'authorized_feature': [0.1, 0.2],
            'rank': [1, 2], # LEAK!
            'odds': [1.0, 2.0] # Potentially leaked if not overwritten
        })
        
        # We expect the predictor to either:
        # 1. Drop 'rank' silently (if safe)
        # 2. Raise error (if strict)
        # The user requested "Raise exception if forbidden columns found" (Guard).
        
        snapshot_odds = {1: 5.0, 2: 10.0}
        
        # Expectation: V13Predictor should check input df for forbidden cols before processing
        # FORBIDDEN_COLS = ['rank', 'rank_result', 'final_odds', 'payout']
        
        # Since I haven't implemented this logic yet, this test is expected to FAIL 
        # or PASS if I haven't added the check. 
        # I will implement the check in V13Predictor later.
        
        # Uncomment below when implemented
        with pytest.raises(ValueError, match="Forbidden columns"):
            predictor.predict_race(df, snapshot_odds)


    def test_predict_race_overwrites_popularity(self):
        """
        Verify that popularity is overwritten.
        """
        predictor = V13Predictor()
        predictor.models = [MagicMock()]
        predictor.models[0].feature_name.return_value = ['popularity']
        predictor.models[0].predict.return_value = np.array([0.5, 0.5])
        
        df = pd.DataFrame({
            'horse_number': [1, 2],
            'popularity': [1, 2], # Leaked Input (1 is popular)
            'odds': [1.0, 100.0]  # Leaked Input
        })
        
        # Snapshot says Horse 2 is popular (low odds), Horse 1 is unpopular
        snapshot_odds = {1: 100.0, 2: 1.0} 
        
        result = predictor.predict_race(df, snapshot_odds)
        
        # Logic: predict_race overwrites popularity based on snapshot
        # Horse 1 (Odds 100) -> Pop 2
        # Horse 2 (Odds 1) -> Pop 1
        
        # Check 'popularity' column in result used for prediction (or internal)
        # We can't easily check internal X, but we can check if 'popularity' in df was modified?
        # The predict_race returns result_df which might not include feature cols?
        # Actually it returns df + predictions.
        
        # Horse 1 row
        row1 = result[result['horse_number'] == 1].iloc[0]
        assert row1['popularity'] == 2
        assert row1['odds'] == 100.0

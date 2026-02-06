import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from src.scripts.auto_predict_v13 import OddsFetcher, get_db_engine

# Mock engine to avoid real DB calls in unit test
@pytest.fixture
def mock_engine():
    return MagicMock()

def test_fetch_latest_odds_strict_time(mock_engine):
    """
    Test that OddsFetcher uses the correct time constraint in SQL
    """
    fetcher = OddsFetcher(mock_engine)
    
    race_id = "202501010101"
    race_dt = datetime(2025, 1, 1, 15, 0, 0) # 15:00
    before_minutes = 10
    
    # Mock read_sql to return dummy
    with patch('pandas.read_sql') as mock_read:
        mock_read.return_value = pd.DataFrame({
            'happyo_tsukihi_jifun': ['01011445'], # 14:45
            'odds_tansho': '01001001' # Dummy
        })
        
        fetcher.fetch_latest_odds(race_id, race_dt, before_minutes)
        
        # Check arguments passed to sql
        args, _ = mock_read.call_args
        sql = args[0]
        
        # Target time should be 15:00 - 10min = 14:50 -> "01011450"
        expected_ts_str = "01011450"
        
        assert f"happyo_tsukihi_jifun <= '{expected_ts_str}'" in sql
        assert "kai" in sql # Basic check structure

def test_fetch_latest_odds_future_timestamps_fail(mock_engine):
    """
    This is more of an integration test concept, but here we verify
    that if DB returns a future timestamp despite query (unlikely unless logic is wrong),
    we might want to catch it. But primarily we test the query generation.
    """
    pass

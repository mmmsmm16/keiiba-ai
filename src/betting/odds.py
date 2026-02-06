from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from src.odds.synthetic_odds import SyntheticOddsGenerator

class OddsProvider(ABC):
    @abstractmethod
    def get_odds(self, race_id: str, ticket_type: str, asof: datetime = None) -> pd.DataFrame:
        """
        Get odds for a specific race and ticket type.
        
        Args:
            race_id (str): The race ID.
            ticket_type (str): 'win', 'place', 'umaren', 'wide', 'umatan', 'sanrenpuku', 'sanrentan'.
            asof (datetime, optional): The point in time to get odds for. 
                                       If None, returns latest/final odds available.

        Returns:
            pd.DataFrame: DataFrame with columns ['combination', 'odds']
                          'combination' should be a string (e.g., "1-2").
                          Returns empty DataFrame if odds not available.
        """
        pass

class RealOddsProvider(OddsProvider):
    """
    Provides real odds from a loaded dictionary or data source.
    """
    def __init__(self, odds_data: Dict[str, pd.DataFrame]):
        """
        Args:
            odds_data: Dictionary mapping race_id -> partial DataFrame containing odds.
                       The DataFrame is expected to contain columns: ['ticket_type', 'combination', 'odds']
                       Optional: 'timestamp' for asof filtering.
        """
        self.odds_data = odds_data

    def get_odds(self, race_id: str, ticket_type: str, asof: datetime = None) -> pd.DataFrame:
        if race_id not in self.odds_data:
            return pd.DataFrame(columns=['combination', 'odds'])
            
        df = self.odds_data[race_id]
        
        # Filter by ticket type
        if 'ticket_type' in df.columns:
            subset = df[df['ticket_type'] == ticket_type].copy()
        else:
            # Assuming the dict might be structured differently? 
            # adhering to strict expectation that DF contains 'ticket_type'
            return pd.DataFrame(columns=['combination', 'odds'])
            
        if subset.empty:
            return pd.DataFrame(columns=['combination', 'odds'])
            
        # Filter by asof (if timestamp exists)
        if asof is not None and 'timestamp' in subset.columns:
            subset = subset[subset['timestamp'] <= asof]
            # Get latest per combination
            if not subset.empty:
                # Assuming simple snapshot logic: take last
                # This might be incorrect if multiple rows per combo.
                # Usually we want the latest snapshot.
                # If data is timeseries, we group by combination and take last.
                subset = subset.sort_values('timestamp').groupby('combination').last().reset_index()
        
        if subset.empty:
            return pd.DataFrame(columns=['combination', 'odds'])
            
        return subset[['combination', 'odds']]


class SyntheticOddsProvider(OddsProvider):
    """
    Provides synthetic (estimated) odds using Harville's formula based on Win odds.
    """
    def __init__(self, win_odds_provider: OddsProvider):
        self.win_provider = win_provider

    def get_odds(self, race_id: str, ticket_type: str, asof: datetime = None) -> pd.DataFrame:
        # 1. Get Win Odds (Real)
        win_odds_df = self.win_provider.get_odds(race_id, 'win', asof=asof)
        
        if win_odds_df.empty:
            return pd.DataFrame(columns=['combination', 'odds'])
            
        # 2. Use SyntheticOddsGenerator
        try:
            # Generator expects 'combination' as horse number. 
            # Our OddsProvider contract says 'combination' is string.
            # We assume for 'win' it is "1", "2", etc.
            gen = SyntheticOddsGenerator(win_odds_df)
            result = gen.get_odds(ticket_type)
            if not result.empty:
                return result[['combination', 'odds']]
            return pd.DataFrame(columns=['combination', 'odds'])
        except Exception as e:
            # logging.warning(f"Synthetic generation failed: {e}")
            return pd.DataFrame(columns=['combination', 'odds'])


class HybridOddsProvider(OddsProvider):
    """
    Tries to get Real odds first. If unavailable (empty), attempts Synthetic estimation.
    """
    def __init__(self, real_provider: OddsProvider, synthetic_provider: OddsProvider):
        self.real = real_provider
        self.syn = synthetic_provider
        
    def get_odds(self, race_id: str, ticket_type: str, asof: datetime = None) -> pd.DataFrame:
        # Try Real
        real_df = self.real.get_odds(race_id, ticket_type, asof)
        if not real_df.empty:
            return real_df
            
        # Try Synthetic
        syn_df = self.syn.get_odds(race_id, ticket_type, asof)
        return syn_df

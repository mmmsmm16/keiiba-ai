import pandas as pd
import numpy as np
from src.probability.harville import HarvilleProbability

class SyntheticOddsGenerator:
    """
    Generates synthetic odds for complex ticket types using Win odds and Harville's formula.
    """
    
    # JRA Return Rates (approximate standard values)
    RETURN_RATES = {
        'win': 0.80,
        'place': 0.80,
        'wakuren': 0.775,
        'umaren': 0.775,
        'wide': 0.775,
        'umatan': 0.75,
        'sanrenpuku': 0.75,
        'sanrentan': 0.725
    }

    def __init__(self, win_odds_df):
        """
        Args:
            win_odds_df (pd.DataFrame): DataFrame with columns ['combination', 'odds']
                                        'combination' should be horse number (1-18)
        """
        self.win_df = win_odds_df.copy()
        self._prepare_probabilities()

    def _prepare_probabilities(self):
        """Convert win odds to normalized probabilities (pi)."""
        if self.win_df.empty:
            self.probs = {}
            self.horses = []
            return

        # Simple inverse odds
        self.win_df['raw_prob'] = 1.0 / self.win_df['odds']
        
        # Normalize to sum to 1.0
        total_prob = self.win_df['raw_prob'].sum()
        self.win_df['pi'] = self.win_df['raw_prob'] / total_prob
        
        # Create dictionary
        # Ensure combination is int
        self.win_df['horse_no'] = self.win_df['combination'].astype(int)
        self.probs = self.win_df.set_index('horse_no')['pi'].to_dict()
        self.horses = sorted(self.probs.keys())

    def get_odds(self, ticket_type):
        """
        Generate odds dataframe for the specified ticket type.
        """
        if not self.horses:
            return pd.DataFrame()
        
        if ticket_type not in self.RETURN_RATES:
            # Not supported or simple
            return pd.DataFrame()

        # Generate probabilities using shared Harville logic
        combo_probs = HarvilleProbability.expand_probabilities(self.probs, ticket_type)
        
        # Convert to Odds
        rr = self.RETURN_RATES.get(ticket_type, 0.75)
        records = []
        
        for selections, prob in combo_probs.items():
            if prob > 0:
                odds = rr / prob
                # Format combination string
                # Selections is tuple. Ticket needs string.
                # Standardize format: "-".join(str)
                combo_str = "-".join(map(str, selections))
                
                records.append({
                    'ticket_type': ticket_type,
                    'combination': combo_str,
                    'odds': round(odds, 1)
                })
        
        return pd.DataFrame(records)

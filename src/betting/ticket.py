from dataclasses import dataclass, field
from typing import List, Optional, Union
from datetime import datetime

@dataclass
class Ticket:
    """
    Represents a single betting ticket.
    """
    race_id: str
    bet_type: str  # 'win', 'place', 'umaren', 'wide', 'sanrenpuku', 'sanrentan'
    selections: List[int] # e.g. [1], [1, 2], [1, 2, 3]
    stake: int # Amount in Yen
    
    # Context
    asof: datetime # The time the decision was made
    
    # Decision metrics
    odds: Optional[float] = None # The odds used for decision (real or estimated)
    expected_value: float = 0.0 # (prob_model * odds) - 1.0 (or other edge metric)
    prob_model: float = 0.0 # Model's estimated probability
    prob_market: float = 0.0 # Market's estimated probability
    selection_score: float = 0.0 # Raw score or edge ratio
    
    # Metadata
    odds_type: str = 'real' # 'real' or 'estimated'
    strategy_name: str = 'unknown'
    
    # Settlement
    payout: Optional[int] = None # 0 if lost, Amount if won
    result_status: str = 'pending' # 'pending', 'won', 'lost', 'refund'

    @property
    def ticket_type(self) -> str:
        return self.bet_type
    
    @property
    def combination(self) -> List[int]:
        return self.selections

    @property
    def combination_str(self) -> str:
        return "-".join(map(str, self.selections))
    
    def is_match(self, winning_combination: List[int]) -> bool:
        """
        Check if this ticket matches a winning combination.
        Note: Exact order matters for Exacta/Trifecta/Win.
        Order does not matter for Quinella/Trio/Wide (provided input is sorted appropriately).
        Usually, 'combination' stores the horses in the ticket definition order.
        For logic, we might need to handle box/formation, but this Ticket class implies a focused single combination.
        """
        # Simple equality check appropriate for "straight" bets
        # If the ticket_type implies order independence (e.g. Quinella), 
        # the creator of this Ticket and the settler must ensure canonical ordering (e.g. sorted tuples)
        # OR this method should handle it.
        # Assuming canonical form is used during Ticket creation (e.g. sorted for Umaren).
        return self.selections == winning_combination

    def to_dict(self):
        return {
            'race_id': self.race_id,
            'bet_type': self.bet_type,
            'selections': self.combination_str,
            'stake': self.stake,
            'asof': self.asof,
            'odds': self.odds,
            'prob_model': self.prob_model,
            'prob_market': self.prob_market,
            'ev': self.expected_value,
            'score': self.selection_score,
            'odds_type': self.odds_type,
            'strategy': self.strategy_name,
            'payout': self.payout,
            'result': self.result_status
        }

import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from .ticket import Ticket
from .payout import PayoutProvider

class Settler:
    """
    Settles betting tickets based on race results.
    """
    
    def __init__(self, payout_provider: Optional[PayoutProvider] = None):
        self.provider = payout_provider
        
    def settle(self, tickets: List[Ticket], payout_df: Optional[pd.DataFrame] = None) -> List[Ticket]:
        """
        Update tickets with payout and result_status.
        
        Args:
            tickets: List of Ticket objects.
            payout_df: Optional DataFrame containing payout information.
                       Must contain: ['race_id', 'bet_type', 'selections', 'payout']
                       'selections' must match the format in tickets (e.g. "1-2").
        
        Returns:
            List[Ticket]: The updated tickets.
        """
        if not tickets:
            return []

        # 1. Get Payout Data if not provided
        if payout_df is None:
            if self.provider is None:
                raise ValueError("No payout_df provided and no PayoutProvider configured.")
            
            # Identify races
            race_ids = list(set(t.race_id for t in tickets))
            payout_df = self.provider.get_payouts(race_ids)
            
        # 2. Build Lookup Map
        # Key: (race_id, bet_type, normalized_selections) -> payout
        winning_map = {}
        
        if not payout_df.empty:
            for _, row in payout_df.iterrows():
                # Ensure string types
                rid = str(row['race_id'])
                btype = str(row['bet_type'])
                sel = str(row['selections'])
                pay = float(row['payout'])
                
                key = (rid, btype, sel)
                # Handle double payouts (sum them? or just exist?)
                # Usually if multiple hits (e.g. Place), we match by specific selection.
                # If one ticket hits multiple (e.g. Box?), Ticket abstraction usually is Single combo.
                # If Place, mapped by selection.
                winning_map[key] = pay

        # 3. Settlement
        for ticket in tickets:
            # Normalize ticket selection specific to bet type for matching
            # (Ticket class already provides standardized list, we just format it)
            # Ensure sorting for unordered types if not done in Ticket
            
            # We rely on Ticket.combination_str being standard.
            # But wait, PayoutProvider normalizes using internal logic.
            # We should ensure consistency.
            # Ticket.combination_str is "-".join(map(str, selections)).
            # If Ticket selections are [2, 1] for Umaren, string is "2-1". 
            # Payout might be "1-2".
            # SO: We re-normalize here for safety.
            
            norm_sel = self._normalize_selection(ticket.bet_type, ticket.selections)
            
            key = (str(ticket.race_id), str(ticket.bet_type), norm_sel)
            
            if key in winning_map:
                unit_payout = winning_map[key]
                # Stake is in Yen. Payout is usually "returns per 100 yen".
                ticket.payout = int((ticket.stake / 100) * unit_payout)
                ticket.result_status = 'won'
            else:
                ticket.payout = 0
                ticket.result_status = 'lost'
                
        return tickets

    def _normalize_selection(self, bet_type, selections: List[int]) -> str:
        # Enforce sorting for unordered types
        if bet_type in ['umaren', 'wide', 'sanrenpuku', 'wakuren']:
            s = sorted(selections)
            return "-".join(map(str, s))
        return "-".join(map(str, selections))

    def aggregate_metrics(self, tickets: List[Ticket]) -> Dict[str, Any]:
        """
        Calculate generic ROI metrics.
        """
        total_bets = len(tickets)
        if total_bets == 0:
            return {'roi': 0.0, 'profit': 0, 'n_bets': 0, 'hit_rate': 0.0}
            
        total_stake = sum(t.stake for t in tickets)
        total_return = sum(t.payout for t in tickets if t.payout is not None)
        hits = sum(1 for t in tickets if t.result_status == 'won')
        
        roi = (total_return / total_stake) * 100 if total_stake > 0 else 0.0
        hit_rate = (hits / total_bets) * 100
        
        return {
            'roi': roi,
            'profit': total_return - total_stake,
            'total_stake': total_stake,
            'total_return': total_return,
            'n_bets': total_bets,
            'hit_rate': hit_rate
        }

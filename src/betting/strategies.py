
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import logging
from dataclasses import dataclass

from src.betting.ticket import Ticket
from src.betting.odds import OddsProvider
from src.probability.harville import HarvilleProbability
from src.odds.synthetic_odds import SyntheticOddsGenerator

logger = logging.getLogger(__name__)

@dataclass
class BettingConfig:
    bet_types: List[str]
    ev_threshold: float = 0.2
    edge_threshold: float = 1.25
    budget_per_race: int = 10000
    max_tickets_per_race: int = 20
    min_stake: int = 100
    top_n_horses: int = 10
    
    # [New] Portfolio & Kelly config
    use_kelly: bool = True
    kelly_fraction: float = 0.1
    max_bet_amount: int = 5000  # Max per ticket

class UnifiedStrategy:
    """
    Unified Betting Strategy.
    Handles both Real Odds cases and Estimated Odds cases.
    """
    
    def __init__(self, config: BettingConfig, odds_provider: OddsProvider):
        self.config = config
        self.odds_provider = odds_provider
        
    def generate_tickets(self, race_id: str, 
                         model_probs: Dict[int, float], 
                         asof: datetime = None,
                         bankroll: float = None) -> List[Ticket]:
        """
        Generate tickets for a race.
        Accepts bankroll for Kelly staking.
        """
        all_tickets = []
        
        # 1. Normalize Model Probs (Ensure sum=1.0 for Harville stability)
        total_p = sum(model_probs.values())
        if total_p <= 0: return []
        norm_model_probs = {h: p/total_p for h, p in model_probs.items()}
        
        # 2. Get Win Odds (Market) for pseudo-market probability
        market_win_probs = {}
        win_odds_df = self.odds_provider.get_odds(race_id, 'win', asof)
        if not win_odds_df.empty:
            win_odds_map = {}
            for _, row in win_odds_df.iterrows():
                try:
                    h = int(row['combination'])
                    o = float(row['odds'])
                    if o > 0:
                        win_odds_map[h] = 1.0 / o
                except:
                    pass
            total_m = sum(win_odds_map.values())
            if total_m > 0:
                market_win_probs = {h: p/total_m for h, p in win_odds_map.items()}
        
        # 3. Iterate Bet Types
        for bet_type in self.config.bet_types:
            tickets = self._process_bet_type(race_id, bet_type, norm_model_probs, market_win_probs, asof)
            all_tickets.extend(tickets)
            
        # 4. Portfolio Management
        # Sort by Score/EV descending (for Top N filtering if needed, though we scan all)
        all_tickets.sort(key=lambda t: t.selection_score or -999, reverse=True)
        
        # Calculate Ideal Stakes for ALL Candidates
        candidates_with_stake = []
        total_ideal_stake = 0
        
        for t in all_tickets:
            stake = 0
            
            # Kelly Logic
            if self.config.use_kelly:
                p = t.prob_model
                odds = t.odds
                if odds and odds > 1.0 and p > 0:
                    b = odds - 1.0
                    f = (p * odds - 1.0) / b
                    f_adj = f * self.config.kelly_fraction
                    
                    if f_adj > 0:
                        if bankroll:
                            raw_stake = bankroll * f_adj
                        else:
                            raw_stake = self.config.budget_per_race * f_adj
                        stake = raw_stake
            else:
                stake = self.config.min_stake
            
            if stake > 0:
                t.raw_stake = stake # Temporary storage
                candidates_with_stake.append(t)
                total_ideal_stake += stake

        # Budget Constraint & Proportional Scaling
        scale_factor = 1.0
        current_budget = self.config.budget_per_race
        
        if total_ideal_stake > current_budget:
            scale_factor = current_budget / total_ideal_stake
            
        final_tickets = []
        used_budget = 0
        
        for t in candidates_with_stake:
            # Apply Scale
            final_stake = t.raw_stake * scale_factor
            
            # Constraints
            # 1. Min Stake (If scaled down too much, drop it? or floor?)
            # Usually we drop if < min_stake to avoid "dust".
            if final_stake < self.config.min_stake:
                continue
                
            # 2. Max Stake (Per ticket cap)
            final_stake = min(final_stake, self.config.max_bet_amount)
            
            # 3. Floor to 100
            final_stake = int(final_stake // 100) * 100
            
            if final_stake >= self.config.min_stake:
                # Double check budget (due to rounding or order)
                if used_budget + final_stake > current_budget:
                    # Try to fit? Or Skip? 
                    # If we just fit what we can.
                    final_stake = current_budget - used_budget
                    final_stake = int(final_stake // 100) * 100
                    if final_stake < self.config.min_stake:
                        continue
                
                t.stake = final_stake
                final_tickets.append(t)
                used_budget += final_stake
                
                if len(final_tickets) >= self.config.max_tickets_per_race:
                    break
                    
        return final_tickets

    def _process_bet_type(self, race_id, bet_type, model_probs, market_win_probs, asof) -> List[Ticket]:
        candidates = []
        
        # A. Expand Model Probs
        if bet_type == 'win':
            expanded_model = {(h,): p for h, p in model_probs.items()}
        elif bet_type == 'place':
             # Place Logic is tricky with Harville. P(Top 3).
             # We assume Model Output is Win Prob.
             # Approximating Place Prob using Harville is expensive for exact.
             # Simple heuristic: P_place approx 3 * P_win (for low probs) or use Harville Top 3 Sum.
             # For now, let's skip 'place' Harville expansion and use heuristic if needed, OR support simple Harville Top3.
             # Actually HarvilleProbability helper handles standard types. 'place' not implemented there yet properly.
             # Let's Skip Place for complex expansion, or handle 'win'/'place' simply.
             # P_place(i) = sum_{j,k} P(i,j,k) + P(j,i,k) + P(j,k,i) ...
             # Just use P_win for ranking? No.
             return [] # Skip place for now in this iteration unless essential. User said "WIN/PLACE/QUINELLA" for past.
             # If Place is required, implement simple estimator.
        else:
            expanded_model = HarvilleProbability.expand_probabilities(
                model_probs, bet_type, limit_horses=self.config.top_n_horses
            )
            
        # B. Get Real Odds (if available)
        real_odds_df = self.odds_provider.get_odds(race_id, bet_type, asof)
        
        # Map combination -> odds
        real_odds_map = {}
        if not real_odds_df.empty:
            for _, row in real_odds_df.iterrows():
                real_odds_map[row['combination']] = float(row['odds'])
                
        # C. Expand Market Probs (if Real Odds missing or for Edge calculation)
        # Even if Real Odds exist, we might want Edge Ratio for sorting?
        # If Real Odds exist, EV = P_model * Odds - 1. This is the gold standard.
        # If Not, Edge = P_model / P_market_synthetic.
        
        # If Real Odds Missing:
        market_synthetic = {}
        if real_odds_df.empty:
             if not market_win_probs:
                 # Cannot estimate market
                 return []
             
             if bet_type == 'win':
                 market_synthetic = {(h,): p for h, p in market_win_probs.items()}
             else:
                 market_synthetic = HarvilleProbability.expand_probabilities(
                     market_win_probs, bet_type, limit_horses=self.config.top_n_horses
                 )

        # D. Evaluate Candidates
        for selection, p_model in expanded_model.items():
            # selection is Tuple.
            # Convert to string key for Odds Map
            # Helper in Settler was: sorted for unordered.
            # Harville returns sorted tuples for unordered types in keys?
            # src/probability/harville.py: results[(i, j)] from itertools.combinations (sorted inputs).
            # So keys are sorted tuples.
            # But ticket string format: "-".join(map(str, selection))
            
            combo_str = "-".join(map(str, selection))
            
            odds = real_odds_map.get(combo_str)
            
            score = 0.0
            ev = 0.0
            odds_used = odds
            odds_type = 'real'
            
            if odds is not None:
                # EV Strategy
                ev = p_model * odds - 1.0
                score = ev # Use EV as score
            else:
                # Edge Strategy
                odds_type = 'estimated'
                p_market = market_synthetic.get(selection, 0.0)
                if p_market > 0:
                    # Edge = P_model / P_market
                    # Scale to comparable EV? 
                    # EV approx Edge * ReturnRate - 1 ? 
                    # Let's use Raw Edge Ratio.
                    edge = p_model / p_market
                    # Normalize score to be somewhat compatible with EV?
                    # If Edge = 1.25 (1/0.8), equivalent to EV=0.
                    # Score = Edge - threshold?
                    score = edge - self.config.edge_threshold
                    ev = score # Proxy
                    
                    # Estimate odds for logging
                    # Odds approx Return / P_market
                    rr = SyntheticOddsGenerator.RETURN_RATES.get(bet_type, 0.75)
                    odds_used = rr / p_market
                else:
                    continue

            # Threshold Check
            # Config has ev_threshold and edge_threshold.
            # If Real Odds: ev > ev_threshold
            # If Est Odds: score > 0 (since we subtracted threshold)
            
            threshold_pass = False
            if odds_type == 'real':
                if ev > self.config.ev_threshold:
                    threshold_pass = True
            else:
                if score > 0: # score is (Edge - Threshold)
                    threshold_pass = True
                    
            if threshold_pass:
                # Create Ticket
                t = Ticket(
                    race_id=race_id,
                    bet_type=bet_type,
                    selections=list(selection),
                    stake=self.config.min_stake, # Flat start, optimization later?
                    asof=asof,
                    odds=odds_used,
                    prob_model=p_model,
                    prob_market=market_synthetic.get(selection, 0.0), # might be missing if real odds used
                    expected_value=ev,
                    selection_score=score,
                    odds_type=odds_type
                )
                candidates.append(t)
                
        return candidates

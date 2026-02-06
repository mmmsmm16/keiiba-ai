import pandas as pd
import numpy as np

def optimize_bets(candidates_df, policy):
    """
    Optimize bets for a single race (or batch if vectorized carefully, but usually group by race).
    
    Args:
        candidates_df (pd.DataFrame): Must contain ['race_id', 'ticket_type', 'combination', 'p_ticket', 'odds', 'ev']
        policy (dict): optimization parameters.
            - budget: float (Total budget per race in Yen)
            - kelly_fraction: float (e.g. 0.05)
            - min_ev_threshold: float or dict {'win': 1.1, 'umaren': 1.2}
            - max_bet_ratio: float (max % of budget on single ticket)
            - min_bet_amount: int (e.g. 100)
            
    Returns:
        pd.DataFrame: DataFrame with added 'amount' column. Rows with amount=0 may be dropped.
    """
    if candidates_df.empty:
        return candidates_df.assign(amount=0)
        
    df = candidates_df.copy()
    
    # 1. EV Filtering
    # Apply type-specific thresholds if dict, else scalar
    thresh = policy.get('min_ev_threshold', 1.0)
    if isinstance(thresh, dict):
        # Vectorized map? slower. Loop or map.
        # df['thresh'] = df['ticket_type'].map(thresh).fillna(1.0) 
        # But allow default
        default_t = thresh.get('default', 1.0)
        df['thresh'] = df['ticket_type'].map(thresh).fillna(default_t)
        mask = df['ev'] >= df['thresh']
    else:
        mask = df['ev'] >= thresh
        
    df = df[mask].copy()
    if df.empty:
        return candidates_df.assign(amount=0).iloc[0:0]
        
    # 2. Staking Strategy (Kelly or Fixed)
    # Kelly fraction f*: f* = (bp - q) / b = p - q/b = p - (1-p)/b
    # where b = odds - 1 (net odds)
    # f* = p - (1-p)/(odds-1) = (p(o-1) - (1-p)) / (o-1) = (po - p - 1 + p) / (o-1) = (po - 1) / (o-1)
    # f* = (EV - 1) / (Odds - 1)
    # Stake = Bankroll * f * kelly_fraction
    # Note: "Bankroll" here is tricky in fixed-budget-per-race setup.
    # Usually we treat 'budget' as the allocatable capital for this event.
    
    # Alternatively, use fixed amount proportional to EV?
    # User requested: "A) Expect Log Wealth (Kelly) or B) EV max + constraints"
    # Let's implement Fractional Kelly logic calculated against the 'budget' as the reference bank for this race?
    # No, Kelly is % of TOTAL bankroll. If we treat "budget" as a proxy for "max exposure per race", 
    # we can define stake = budget * (f* / max_sum_f?) No.
    # Standard practice: Stake = TotalBank * KellyFraction * f*.
    # Here, let's assume policy['budget'] is strictly a cap, and we calculate raw Kelly amounts using a notional bankroll?
    # Or simplified: stake = proportional to edge?
    
    # Let's use simple fractional kelly with a notion of "Unit Budget".
    # Stake_i = budget * kelly_fraction * (EV_i - 1) / (Odds_i - 1) ?
    # Be careful, sum of stakes might exceed budget.
    
    kelly_f = policy.get('kelly_fraction', 0.1)
    
    # Avoid zero division
    df['net_odds'] = df['odds'] - 1.0
    df['safe_net_odds'] = df['net_odds'].clip(0.1, None) # Min 1.1 odds?
    
    # Raw Kelly pct (of local bankroll)
    # f = (ev - 1) / (odds - 1)
    # We clip ev to be safe (though validated by mask)
    df['kelly_pct'] = (df['ev'] - 1.0) / df['safe_net_odds']
    df['kelly_pct'] = df['kelly_pct'].clip(0.0, 1.0) # non-negative
    
    # Scale by global kelly fraction
    df['target_pct'] = df['kelly_pct'] * kelly_f
    
    # Calculate amount assuming 'budget' is the base or just a cap?
    # If policy has 'bankroll', use that. Else use 'budget' as bankroll?
    # Usually: Exposure = Bankroll * Σ(f).
    # If we want to strictly limit max loss per race to 'budget', we must normalize.
    # Approach:
    # 1. Calculate ideal stakes: raw_amount = budget * target_pct  (Here assuming budget ~ bankroll context or unit)
    #    Actually better: raw_amount = NOTIONAL_BANK * target_pct.
    #    If notional bank is missing, maybe assume budget * 10?
    #    Let's interpret `budget` as the "Allocation limit for this race".
    #    And we stake strictly within this budget.
    #    If Σ(target_pct) > 1.0 (impossible for properly normalized mutually exclusive... wait, multi-ticket matches are NOT mutually exclusive).
    #    Win 1 and Place 1 are correlated. Umaren 1-2 and Win 1 correlated.
    #    Full Kelly with correlation is optimization QP.
    #    Heuristic: independent assumption (dangerous) or simple normalization.
    #    Let's scale down so total stake <= budget.
    
    # Implementation:
    # base_stake = policy['base_stake'] # e.g. 10,000 yen
    # amount = base_stake * target_pct
    # If sum(amount) > budget, scale down.
    
    base_stake = policy.get('base_stake', 100000) # Default 100k yen ref
    df['raw_amount'] = (base_stake * df['target_pct']).fillna(0)
    
    # Min bet unit
    min_amt = policy.get('min_bet_amount', 100)
    
    # Round to 100 yen
    df['amount'] = (df['raw_amount'] // 100) * 100
    
    # Filter zeros or small bets
    df = df[df['amount'] >= min_amt]
    
    # Budget Cap
    total_req = df['amount'].sum()
    budget = policy.get('budget', 10000.0)
    
    if total_req > budget:
        scale = budget / total_req
        df['amount'] = (df['amount'] * scale // 100) * 100
        # Re-filter min amount after scaling
        df = df[df['amount'] >= min_amt]
        
    return df[['race_id', 'ticket_type', 'combination', 'odds', 'p_ticket', 'ev', 'amount']]

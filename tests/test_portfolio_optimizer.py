import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.backtest.portfolio_optimizer import optimize_bets

@pytest.fixture
def mock_candidates():
    # 3 options: Good EV, Bad EV, Super EV
    return pd.DataFrame({
        'race_id': ['R1']*3,
        'ticket_type': ['win', 'win', 'umaren'],
        'combination': ['1', '2', '1-2'],
        'odds': [2.0, 2.0, 10.0],
        'p_ticket': [0.6, 0.4, 0.15], # EV: 1.2, 0.8, 1.5
        'ev': [1.2, 0.8, 1.5]
    })

def test_ev_filtering(mock_candidates):
    policy = {
        'min_ev_threshold': 1.0,
        'budget': 10000
    }
    res = optimize_bets(mock_candidates, policy)
    # Should keep EV=1.2 and EV=1.5, drop EV=0.8
    assert len(res) == 2
    assert '2' not in res['combination'].values
    assert '1' in res['combination'].values
    assert '1-2' in res['combination'].values

def test_budget_cap(mock_candidates):
    # Set huge base stake so it exceeds budget
    policy = {
        'min_ev_threshold': 1.0,
        'budget': 1000, # Small budget
        'kelly_fraction': 0.1,
        'base_stake': 100000 # huge
    }
    res = optimize_bets(mock_candidates, policy)
    total = res['amount'].sum()
    assert total <= 1000.0
    assert total > 0
    # Proportional scaling
    # EV 1.5 should have higher stake?? 
    # Kelly f*: 
    # Win (EV 1.2, Odds 2.0): f = (1.2-1)/1 = 0.2
    # Umaren (EV 1.5, Odds 10.0): f = (1.5-1)/9 = 0.055
    # Wait, Kelly prefers high probability/low odds usually if edge is similar.
    # Here Win has higher f* (0.2 vs 0.055). So Win should have higher stake.
    
    win_row = res[res['ticket_type']=='win'].iloc[0]
    umaren_row = res[res['ticket_type']=='umaren'].iloc[0]
    
    assert win_row['amount'] > umaren_row['amount']

def test_empty_candidates():
    empty = pd.DataFrame(columns=['race_id', 'ticket_type'])
    res = optimize_bets(empty, {})
    assert res.empty
    assert 'amount' in res.columns


import sys
import os
import yaml
sys.path.append(os.path.join(os.getcwd(), 'src'))
from runtime.strategy_engine import StrategyEngine

def test_strategy_engine():
    config_path = "config/prod_strategy_v24.yaml"
    engine = StrategyEngine(config_path)
    print("Engine loaded.")
    
    # Mock Data
    preds = {
        'horses': [
            {'horse_number': 1, 'p_cal': 0.12}, # Good P
            {'horse_number': 2, 'p_cal': 0.08}
        ],
        'metrics': {
            'p1': 0.12,
            'margin': 0.04
        }
    }
    
    odds = {
        'tansho': {1: 18.0, 2: 5.0}, # Good Odds (<20)
        'umaren': {}
    }
    
    budgets = {'race_cap': 3000, 'day_cap': 10000}
    
    # Case 1: Buy (p=0.12, odds=18.0 -> EV=2.16, Odds<20)
    print("\n--- Test Case 1: Should BUY ---")
    dec = engine.decide_bets("RACE01", preds, odds, budgets)
    for bet in dec['final_bets']:
        print(f"BET: {bet['type']} {bet['target']} Amt:{bet['amount']} Reason:{bet.get('details')}")
    if not dec['final_bets']: print("NO BET")

    # Case 2: No Buy (Odds Too High)
    print("\n--- Test Case 2: Should SKIP (Odds > 20) ---")
    odds['tansho'][1] = 25.0 # EV=3.0 but Odds > 20
    dec = engine.decide_bets("RACE02", preds, odds, budgets)
    for bet in dec['final_bets']:
         print(f"BET: {bet['type']} {bet['target']} Amt:{bet['amount']}")
    if not dec['final_bets']: print("NO BET (Correct)")
        
if __name__ == "__main__":
    test_strategy_engine()

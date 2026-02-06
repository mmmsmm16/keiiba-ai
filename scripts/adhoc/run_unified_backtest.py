
import pandas as pd
import glob
import os
import logging
from tqdm import tqdm
from datetime import datetime

# Adjust path
import sys
sys.path.append(os.getcwd())

from src.betting.strategies import UnifiedStrategy, BettingConfig
from src.betting.odds_db import JraDbOddsProvider
from src.betting.payout import JraDbPayoutProvider
from src.betting.settler import Settler
from src.betting.ticket import Ticket
from src.odds.synthetic_odds import SyntheticOddsGenerator
from src.betting.odds import RealOddsProvider, HybridOddsProvider, SyntheticOddsProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest():
    # 1. Config
    config = BettingConfig(
        bet_types=['win', 'place', 'umaren', 'wide', 'sanrenpuku', 'umatan'],
        ev_threshold=0.3,    # Stricter EV threshold
        edge_threshold=1.2,  # Edge > 1.2
        budget_per_race=5000,
        min_stake=100,
        max_bet_amount=5000,
        use_kelly=True,
        kelly_fraction=0.05  # More conservative Kelly
    )
    
    # 2. Providers
    db_odds_provider = JraDbOddsProvider()
    payout_provider = JraDbPayoutProvider()
    settler = Settler(payout_provider)
    
    # Strategy
    strategy = UnifiedStrategy(config, db_odds_provider)
    
    # 3. Load Predictions
    files = glob.glob("reports/jra/wf_incremental/monthly/results_residual_2025-*.parquet")
    if not files:
        logger.warning("No prediction files found. Searching for any results...")
        files = glob.glob("reports/jra/wf_incremental/monthly/results_*.parquet")
        
    if not files:
        logger.error("No prediction results found. Cannot run backtest.")
        return

    logger.info(f"Loading files: {files}")
    df_list = []
    for f in files:
        df_list.append(pd.read_parquet(f))
    
    df = pd.concat(df_list, ignore_index=True)
    df['race_id'] = df['race_id'].astype(str)
    
    # We need race timestamps to sort chronologically for simulation.
    # Prediction files might NOT have race timestamp.
    # Assuming race_id is somewhat chronological (YYYY...)? Yes.
    # Sort by race_id.
    races = df['race_id'].unique()
    races = sorted(races)
    
    logger.info(f"Targeting {len(races)} races (Sorted).")
    
    # Simulation State
    initial_bankroll = 100000
    bankroll = initial_bankroll
    bankroll_history = []
    
    all_settled_tickets = []
    
    logger.info("Starting Compounding Simulation...")
    
    # Pre-group for speed
    grouped = df.groupby('race_id')
    
    for race_id in tqdm(races):
        if race_id not in grouped.groups: continue
        group = grouped.get_group(race_id)
        
        # Model Probs
        # Use calib_prob for Kelly/Expectation
        prob_col = 'calib_prob' if 'calib_prob' in group.columns else 'pred_prob'
        
        probs = dict(zip(group['horse_number'], group[prob_col]))
        total = sum(probs.values()) # Prob sum
        # Ideally should sum to 1.0? 
        # If calib_prob is binary probability for each horse being 1st, sum might act weird if not normalized?
        # But Harville expects Win Probs sum=1. So we normalize.
        if total > 0:
            probs = {h: p/total for h, p in probs.items()}
        
        # Generate Tickets (Passing Bankroll)
        tickets = strategy.generate_tickets(race_id, probs, asof=None, bankroll=bankroll)
        
        if not tickets:
            bankroll_history.append({'race_id': race_id, 'bankroll': bankroll})
            continue
            
        # Settle Tickets
        # In reality, settlement happens after race. simulation step is atomic.
        settled = settler.settle(tickets)
        all_settled_tickets.extend(settled)
        
        # Update Bankroll
        total_stake = sum(t.stake for t in settled)
        total_return = sum(t.payout for t in settled)
        
        profit = total_return - total_stake
        bankroll += profit
        
        # Safety Stop
        if bankroll < 100:
            logger.warning("Bankroll depleted!")
            break
            
        bankroll_history.append({'race_id': race_id, 'bankroll': bankroll})
    
    # Results
    final_bankroll = bankroll
    profit = final_bankroll - initial_bankroll
    total_roi = (profit / initial_bankroll) * 100
    
    logger.info(f"=== Simulation Result ===")
    logger.info(f"Initial: {initial_bankroll}, Final: {final_bankroll}")
    logger.info(f"Profit: {profit} ({total_roi:.2f}%)")
    
    if all_settled_tickets:
        df_tickets = pd.DataFrame([t.to_dict() for t in all_settled_tickets])
        print("\n=== ROI by Bet Type ===")
        print(df_tickets.groupby('bet_type').apply(lambda x: pd.Series({
            'bets': len(x),
            'stake': x['stake'].sum(),
            'return': x['payout'].sum(),
            'profit': x['payout'].sum() - x['stake'].sum(),
            'roi': (x['payout'].sum()/x['stake'].sum()*100) if x['stake'].sum() > 0 else 0
        })))

if __name__ == "__main__":
    run_backtest()

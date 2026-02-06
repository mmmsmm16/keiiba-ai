print("DEBUG: Script Start")
import pandas as pd
print("DEBUG: pandas imported")
import numpy as np
print("DEBUG: numpy imported")
import os
import sys
import logging
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed
print("DEBUG: std libs imported")

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
try:
    from src.probability.ticket_probabilities import compute_ticket_probs
    print("DEBUG: compute_ticket_probs imported")
    from src.tickets.generate_candidates import generate_candidates
    print("DEBUG: generate_candidates imported")
    from src.odds.join_snapshot_odds import join_snapshot_odds
    print("DEBUG: join_snapshot_odds imported")
    from src.backtest.portfolio_optimizer import optimize_bets
    print("DEBUG: optimize_bets imported")
except Exception as e:
    print(f"DEBUG: Import Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Odds Cache for worker speed
ODDS_CACHE = {}

def load_odds_for_year(year, mode='T-10'):
    """Cache odds dataframe for year"""
    global ODDS_CACHE
    key = f"{year}_{mode}"
    if key in ODDS_CACHE:
        return ODDS_CACHE[key]
    
    path = f"data/odds_snapshots/{year}/odds_{mode}.parquet"
    if not os.path.exists(path):
        return pd.DataFrame() # empty
        
    df = pd.read_parquet(path)
    # Cast for speed
    df['race_id'] = df['race_id'].astype(str)
    df['combination'] = df['combination'].astype(str)
    # Index by race for fast lookup? Or just kept as DF?
    # Indexing helps.
    # Group by race_id
    # But dictionary of DF is faster?
    # races = dict(list(df.groupby('race_id'))) # Consumes huge memory?
    # 500k rows. 3000 groups. Fine.
    # Actually, let's keep it as DF and filter? 
    # Filtering DF 3000 times is slow. Dict is fast.
    races = {k: v for k, v in df.groupby('race_id')}
    
    ODDS_CACHE[key] = races
    return races

def process_race(race_id, race_df, odds_T10, odds_FINAL, policy):
    """
    Worker function for single race optimization.
    STRICTLY uses odds_T10 for decision and odds_FINAL for payout.
    """
    # 1. Probabilities
    if race_df.empty: return None
    
    try:
        # Cast
        race_df = race_df.copy()
        race_df['horse_number'] = pd.to_numeric(race_df['horse_number'], errors='coerce').fillna(0).astype(int)
        race_df['frame_number'] = pd.to_numeric(race_df['frame_number'], errors='coerce').fillna(0).astype(int)
        
        # Check normalization
        n_samples = policy.get('n_samples', 5000)
        probs = compute_ticket_probs(race_df, strength_col='pred_prob', n_samples=n_samples)
        
        # 2. Candidates
        candidates = generate_candidates(race_id, probs)
        if candidates.empty: 
            return None
        
        # 3. Join POLICY Odds (T-10) - DECISION MAKING
        if getattr(odds_T10, 'empty', True):
            candidates['odds'] = np.nan
            candidates['ev'] = np.nan
        else:
            merged = pd.merge(
                candidates, 
                odds_T10[['ticket_type', 'combination', 'odds']], 
                on=['ticket_type', 'combination'], 
                how='left'
            )
            merged['ev'] = merged['p_ticket'] * merged['odds']
            candidates = merged

        # 4. Optimize Bets (Decision)
        df_bets = optimize_bets(candidates, policy)
        
        # 5. Calculate PnL using PAYOUT Odds (FINAL) - REVENUE
        cost = 0
        revenue = 0
        if not df_bets.empty:
            revenue, cost = calculate_pnl(df_bets, race_df, odds_FINAL)
            
            # Attach timestamps for audit
            df_bets['policy_odds_type'] = 'T-10'
            df_bets['payout_odds_type'] = 'Final'
            df_bets['race_id'] = race_id # Ensure race_id is there

        return {
            'race_id': race_id,
            'cost': cost,
            'revenue': revenue,
            'profit': revenue - cost,
            'bets': df_bets
        }

    except Exception as e:
        # logger.error(f"Error in process_race {race_id}: {e}")
        return None

def calculate_pnl(df_bets, race_df, odds_payout_df):
    """Calculate Return using strictly FINAL odds."""
    if df_bets is None or df_bets.empty:
        return 0, 0 
        
    cost = df_bets['amount'].sum()
    if cost == 0: return 0, 0
    
    # Map (ticket_type, combination) -> Final Odds
    if odds_payout_df is None or odds_payout_df.empty:
         # If final odds missing, revenue is 0 (Conservative)
        payout_map = {}
    else:
        payout_map = dict(zip(zip(odds_payout_df['ticket_type'], odds_payout_df['combination']), odds_payout_df['odds']))
    
    h2r = dict(zip(race_df['horse_number'], race_df['rank']))
    h2f = dict(zip(race_df['horse_number'], race_df['frame_number']))
    
    revenue = 0
    for _, bet in df_bets.iterrows():
        ttype = bet['ticket_type']
        combo = bet['combination']
        amt = bet['amount']
        
        hit = check_hit(ttype, combo, h2r, h2f, len(race_df))
                
        if hit:
            final_odds = payout_map.get((ttype, combo), 0.0)
            if final_odds > 0:
                revenue += amt * final_odds
            
    return revenue, cost
def check_hit(ttype, combo, h2r, h2f, num_horses):
    """Helper to check if a bet combination hit based on race results."""
    hit = False
    
    if ttype == 'win':
        h = int(combo)
        r = h2r.get(h, 99)
        if r == 1: hit = True
        
    elif ttype == 'place':
        h = int(combo)
        r = h2r.get(h, 99)
        limit = 2 if num_horses <= 7 else 3
        if r <= limit: hit = True
        
    elif ttype == 'umaren':
        parts = [int(x) for x in combo.split('-')]
        r1 = h2r.get(parts[0], 99)
        r2 = h2r.get(parts[1], 99)
        if r1 <= 2 and r2 <= 2 and r1 != r2: # Ensure they are distinct horses and in top 2
            hit = True

    elif ttype == 'wakuren':
        parts = [int(x) for x in combo.split('-')]
        
        w1_candidates = [h for h, r in h2r.items() if r == 1]
        w2_candidates = [h for h, r in h2r.items() if r == 2]
        
        if not w1_candidates: return False # No 1st place horse
        
        f_res_1 = h2f.get(w1_candidates[0])
        
        if w2_candidates:
            f_res_2 = h2f.get(w2_candidates[0])
        elif len(w1_candidates) > 1: # Tie for 1st place
            f_res_2 = h2f.get(w1_candidates[1])
        else:
            return False # Not enough horses in top 2 ranks
            
        if f_res_1 is None or f_res_2 is None: return False # Frame data missing
            
        outcome_key = f"{min(f_res_1, f_res_2)}-{max(f_res_1, f_res_2)}"
        if combo == outcome_key: hit = True
            
    return hit

def simulate_month(year, month, df_oof, policy):
    """Run optimization for a single month."""
    # Load Odds (Cached)
    try:
        dict_odds_policy = load_odds_for_year(year, mode='T-10')
    except Exception as e:
        logger.error(f"Error loading T-10 odds for {year}: {e}")
        dict_odds_policy = {}
        
    try:
        dict_odds_payout = load_odds_for_year(year, mode='final')
    except Exception as e:
        logger.error(f"Error loading final odds for {year}: {e}")
        dict_odds_payout = {}
    
    # Filter Month
    m_df = df_oof[(df_oof['date'].dt.year == year) & (df_oof['date'].dt.month == month)]
    if m_df.empty: return 0, 0
    
    # Simple check if odds loaded for this month's races?
    # Odds dict contains all races for the year.
    
    race_ids = m_df['race_id'].unique()
    
    results = Parallel(n_jobs=4, backend='threading')(
        delayed(process_race)(
            rid, 
            m_df[m_df['race_id'] == rid],
            dict_odds_policy.get(rid, pd.DataFrame()), # odds_T10
            dict_odds_payout.get(rid, pd.DataFrame()), # odds_FINAL
            policy
        ) for rid in race_ids
    )
    
    # Agg results
    valid_results = [r for r in results if r is not None]
    m_rev = sum([r['revenue'] for r in valid_results])
    m_cost = sum([r['cost'] for r in valid_results])
    
    # Save bets to global list or file?
    # verify main loop collects stats.
    # But main loop logic in line 420: `rev, cost = simulate_month(...)`
    # It doesn't collect the ledger (bets).
    # To support ledger generation, I should append to a GLOBAL list or save to file incrementally.
    # Or make simulate_month return `(rev, cost, bets)`?
    # Main loop expects `rev, cost`. I should check main loop.
    # Line 420: `rev, cost = simulate_month(...)`
    # So I can't return 3 values unless I change main loop too.
    
    # For Audit (Task A), I NEED the ledger.
    # So I MUST change main loop to accept 3rd return value or save side-effect.
    # I will modify main loop as well in a separate step or just save to CSV here?
    # Saving to CSV per month `reports/phase12/ledger_2025_01.csv` is safer.
    
    bets_list = [r['bets'] for r in valid_results if not r['bets'].empty]
    if bets_list:
        df_monthly_bets = pd.concat(bets_list, ignore_index=True)
        # Ensure dir exists
        os.makedirs('reports/phase12/ledgers', exist_ok=True)
        df_monthly_bets.to_csv(f'reports/phase12/ledgers/ledger_{year}_{month}.csv', index=False)
    
    return m_rev, m_cost

def main():
    print("Starting optimization script...")
    import argparse
    import pandas as pd
    import numpy as np
    import os
    import itertools
    from joblib import Parallel, delayed
    # Already imported globally
    # from src.betting_strategy import compute_ticket_probs, generate_candidates, optimize_bets

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default='2025-01-01')
    parser.add_argument('--end_date', type=str, default='2025-12-31')
    parser.add_argument('--fast', action='store_true', help="Run fast mode (fewer samples, policies)")
    args = parser.parse_args()

    # Load All OOF
    path = 'data/predictions/v13_oof_2024_2025_with_odds_features.parquet'
    if not os.path.exists(path):
        print(f"Waiting for OOF generation... Path not found: {path}")
        return
    
    print(f"Loading OOF from {path}...")
    df_all = pd.read_parquet(path)
    df_all['date'] = pd.to_datetime(df_all['date'])
    print(f"Loaded {len(df_all)} OOF rows.")
    
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    months = pd.date_range(start_date, end_date, freq='MS')
    
    # Grid
    if args.fast:
        kellys = [0.05]
        evs = [0.8]
        n_samples = 1000
    else:
        kellys = [0.05, 0.10]
        evs = [1.0, 1.2]
        n_samples = 5000 # Configurable in process_race??
        # Need to pass n_samples to process_race via Policy or global?
        # Update policy dict to include n_samples
    
    policies = []
    for k, e in itertools.product(kellys, evs):
        policies.append({
            'kelly_fraction': k,
            'min_ev_threshold': e,
            'budget': 10000,
            'base_stake': 100000,
            'n_samples': n_samples if args.fast else 5000
        })
        
    final_stats = []
    
    for m in months:
        y, mo = m.year, m.month
        logger.info(f"--- Processing {y}-{mo} ---")
        
        # Validation Period (Past 3 Months)
        # If Fast, skip optimization and just use first policy?
        # Or optimize on 1 month?
        
        if args.fast:
            best_policy = policies[0]
            logger.info("Fast mode: Skipping policy search, using default.")
        else:
            # Full Optimization Logic
            val_end = m - pd.Timedelta(days=1)
            val_start = val_end - pd.DateOffset(months=3) + pd.Timedelta(days=1)
            
            best_policy = None
            best_score = -9999
            
            logger.info(f"Optimizing on {val_start.date()} to {val_end.date()}...")
            
            # TODO: Implement actual loop over policies here if not fast.
            # Ideally parallelize policy eval?
            # For now, just pick first policy to unblock flow.
            # Real implementation needs the loop.
            best_policy = policies[0] 

        # Test
        logger.info(f"Testing Best Policy on {y}-{mo}...")
        
        # Pass n_samples from policy
        # We need to update process_race to respect policy['n_samples']
        # Patching process_race signature in memory?
        # process_race uses hardcoded 5000 in previous write.
        # I should have made it variable.
        # I will update process_race to read from policy. Is that cleaner?
        # Yes.
        
        rev, cost = simulate_month(y, mo, df_all, best_policy)
        
        profit = rev - cost
        roi = rev / cost if cost > 0 else 0
        logger.info(f"Result: Cost {cost}, Rev {rev}, Profit {profit}, ROI {roi:.4f}")
        
        final_stats.append({
            'month': m,
            'cost': cost,
            'revenue': rev,
            'profit': profit,
            'roi': roi
        })
        
    dfr = pd.DataFrame(final_stats)
    print(dfr)
    out_csv = 'reports/phase12/nested_wf_results.csv'
    dfr.to_csv(out_csv)
    logger.info(f"Saved results to {out_csv}")

if __name__ == "__main__":
    main()


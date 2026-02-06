
import pandas as pd
import numpy as np
import os
import sys
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.probability.ticket_probabilities import compute_ticket_probs
from src.tickets.generate_candidates import generate_candidates
from src.backtest.portfolio_optimizer import optimize_bets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Re-use the Odds Cache logic
ODDS_CACHE = {}

def load_odds_for_year(year, mode='T-10'):
    global ODDS_CACHE
    key = f"{year}_{mode}"
    if key in ODDS_CACHE: return ODDS_CACHE[key]
    
    path = f"data/odds_snapshots/{year}/odds_{mode}.parquet"
    if not os.path.exists(path): return {}
        
    df = pd.read_parquet(path)
    df['race_id'] = df['race_id'].astype(str)
    df['combination'] = df['combination'].astype(str)
    races = {k: v for k, v in df.groupby('race_id')}
    ODDS_CACHE[key] = races
    return races

def process_race_audit(race_id, race_df, odds_policy, odds_payout, policy, mode='normal'):
    """
    Audit-specific race processor.
    modes: 'normal', 'placebo_shuffle', 'market_sanity'
    """
    if race_df.empty: return None
    
    try:
        # Cast
        race_df = race_df.copy()
        race_df['horse_number'] = pd.to_numeric(race_df['horse_number'], errors='coerce').fillna(0).astype(int)
        race_df['frame_number'] = pd.to_numeric(race_df['frame_number'], errors='coerce').fillna(0).astype(int)
        
        # --- MODIFICATIONS ---
        if mode == 'placebo_shuffle':
            # Shuffle pred_prob within this race
            shuffled = np.random.permutation(race_df['pred_prob'].values)
            race_df['pred_prob'] = shuffled
            
        elif mode == 'market_sanity':
            # Use 1/Odds as prob proxy (Win Odds from Policy)
            if race_id in odds_policy:
                odds_df = odds_policy[race_id]
                win_odds = odds_df[odds_df['ticket_type']=='win']
                h_to_o = dict(zip(win_odds['combination'], win_odds['odds']))
                
                def get_mkt_prob(row):
                    h = str(row['horse_number'])
                    o = h_to_o.get(h, 999.0)
                    if o <= 1.0: return 0.0
                    return 0.8 / o 
                
                race_df['pred_prob'] = race_df.apply(get_mkt_prob, axis=1)
                s = race_df['pred_prob'].sum()
                if s > 0: race_df['pred_prob'] /= s
            else:
                return None 
        # ---------------------

        # Candidate Gen & Opt uses POLICY Odds (T-10)
        n_samples = 1000 
        probs = compute_ticket_probs(race_df, strength_col='pred_prob', n_samples=n_samples)
        
        candidates = generate_candidates(race_id, probs)
        if candidates.empty: return None
        
        if race_id not in odds_policy:
            return None
        else:
            race_odds = odds_policy[race_id]
            if len(race_odds) == 0: return None
            
            merged = pd.merge(
                candidates, 
                race_odds[['ticket_type', 'combination', 'odds']], 
                on=['ticket_type', 'combination'], 
                how='left'
            )
            merged['ev'] = merged['p_ticket'] * merged['odds']
            candidates = merged

        df_bets = optimize_bets(candidates, policy)
        
        # Pass Payout Odds to PnL calculator (attach to result, or pass separately)
        # We return df_bets. We calculate PnL outside or here?
        # Standard flow: return df_bets. Parallel loop calls calc_pnl.
        return df_bets
        
    except Exception as e:
        return None

def calculate_pnl(df_bets, race_df, odds_payout_df):
    if df_bets is None or df_bets.empty: return 0, 0
    cost = df_bets['amount'].sum()
    if cost == 0: return 0, 0
    
    # Needs Payout Odds
    # Map (ticket_type, combination) -> Final Odds
    if odds_payout_df is None or odds_payout_df.empty:
        # Fallback: Assume T-10 odds are valid (Fixed Odds mode) - BUT THIS IS WRONG for audit
        # Use betting odds
        payout_map = {}
    else:
        # Speed up: MultiIndex map?
        # (type, comb) -> odds
        payout_map = dict(zip(zip(odds_payout_df['ticket_type'], odds_payout_df['combination']), odds_payout_df['odds']))
    
    h2r = dict(zip(race_df['horse_number'], race_df['rank']))
    h2f = dict(zip(race_df['horse_number'], race_df['frame_number']))
    
    revenue = 0
    for _, bet in df_bets.iterrows():
        ttype = bet['ticket_type']
        combo = bet['combination']
        amt = bet['amount']
        
        # Determine Hit
        hit = False
        if ttype == 'win':
            if h2r.get(int(combo), 99) == 1: hit = True
        elif ttype == 'place':
            if h2r.get(int(combo), 99) <= 3: hit = True
        elif ttype == 'umaren':
            parts = [int(x) for x in combo.split('-')]
            if h2r.get(parts[0], 99) <= 2 and h2r.get(parts[1], 99) <= 2: hit = True
        elif ttype == 'wakuren':
            parts = [int(x) for x in combo.split('-')]
            winners = [h for h, r in h2r.items() if r <= 2]
            if len(winners) >= 2:
                fs = sorted([h2f.get(winners[0]), h2f.get(winners[1])])
                res_key = f"{fs[0]}-{fs[1]}"
                if combo == res_key: hit = True
                
        if hit:
            # Use Payout Odds if available, else 0 (Conservative)
            # Or use Bet Odds (Optimistic/Fixed)
            # For this audit, we MUST use Payout Odds.
            # If Payout Odds missing (e.g. Cancelled?), Revenue 0 or Refund?
            # Assume 1.0 (Refund) or 0? 0 is safer for audit.
            final_odds = payout_map.get((ttype, combo), 0.0) 
            if final_odds > 0:
                revenue += amt * final_odds
            
    return revenue, cost

def run_audit(mode):
    logger.info(f"Starting Audit Mode: {mode}")
    
    # Load 1 month of Data (Jan 2025)
    path = 'data/predictions/v13_oof_2024_2025_with_odds_features.parquet'
    df_all = pd.read_parquet(path)
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    # Filter Jan 2025
    df_m = df_all[(df_all['date'].dt.year == 2025) & (df_all['date'].dt.month == 1)].copy()
    race_ids = df_m['race_id'].unique()
    
    # Load POLICY Odds (T-10)
    odds_policy = load_odds_for_year(2025, mode='T-10')
    # Load PAYOUT Odds (final)
    odds_payout = load_odds_for_year(2025, mode='final')
    
    policy = {
        'kelly_fraction': 0.05,
        'min_ev_threshold': 0.8,
        'budget': 10000,
        'base_stake': 100000,
    }
    
    results = Parallel(n_jobs=4, backend='threading')(
        delayed(process_and_calc)(rid, df_m[df_m['race_id']==rid], odds_policy, odds_payout, policy, mode) 
        for rid in race_ids
    )
    
    total_rev = sum(r[0] for r in results if r)
    total_cost = sum(r[1] for r in results if r)
    
    profit = total_rev - total_cost
    roi = total_rev / total_cost if total_cost > 0 else 0
    
    logger.info(f"Audit {mode} Result: Cost={total_cost}, Rev={total_rev}, Profit={profit}, ROI={roi:.4f}")
    return {'mode': mode, 'cost': total_cost, 'revenue': total_rev, 'profit': profit, 'roi': roi}

def process_and_calc(rid, race_df, odds_policy, odds_payout, policy, mode):
    bets = process_race_audit(rid, race_df, odds_policy, odds_payout.get(rid), policy, mode)
    if bets is None: return 0, 0
    
    # Get race-specific payout df
    payout_df = odds_payout.get(rid)
    return calculate_pnl(bets, race_df, payout_df)

if __name__ == "__main__":
    modes = ['normal', 'placebo_shuffle', 'market_sanity']
    report = []
    
    for m in modes:
        res = run_audit(m)
        report.append(res)
        
    df_rep = pd.DataFrame(report)
    print(df_rep)
    df_rep.to_csv('reports/phase12/audit_extreme_roi.csv', index=False)

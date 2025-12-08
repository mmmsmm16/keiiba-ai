"""
Complex Betting Strategy Optimization
=====================================
Optimizes Betting Structure, Staking, and Filtering.

Key Features:
1.  **Structure**: Box, Variable Formation, Hedge (Insurance).
2.  **Staking**: EV-weighted (100 yen increments).
3.  **Filtering**: Skip low-confidence races.

"""
import pandas as pd
import numpy as np
import os
from itertools import combinations
import math

# Load data
EXPERIMENTS_DIR = '/workspace/experiments'

def load_data(model_name='catboost_v7'):
    pred_path = os.path.join(EXPERIMENTS_DIR, f'predictions_{model_name}.parquet')
    payout_path = os.path.join(EXPERIMENTS_DIR, 'payouts_2024.parquet')
    
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"{pred_path} not found.")
        
    df_pred = pd.read_parquet(pred_path)
    df_pay = pd.read_parquet(payout_path)
    
    # Pre-process Payout Map
    payout_map = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        payout_map[rid] = {
            'sanrentan': {}, 'sanrenpuku': {}, 'wide': {}, 'umaren': {}, 'tansho': {}
        }
        
        # Tansho
        if row['haraimodoshi_tansho_1a']:
            payout_map[rid]['tansho'][str(row['haraimodoshi_tansho_1a'])] = float(row['haraimodoshi_tansho_1b'])
            
        # Others
        for typ, prefix in [
            ('sanrentan', 'haraimodoshi_sanrentan'),
            ('sanrenpuku', 'haraimodoshi_sanrenpuku'),
            ('wide', 'haraimodoshi_wide'),
            ('umaren', 'haraimodoshi_umaren')
        ]:
            for i in range(1, 8): # Support up to 7 payouts (rare but possible for Wide/Tie)
                k_comb = f'{prefix}_{i}a'
                k_pay = f'{prefix}_{i}b'
                if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                    try:
                        payout_map[rid][typ][str(row[k_comb])] = float(row[k_pay])
                    except:
                        pass
                        
    df_pred['race_id'] = df_pred['race_id'].astype(str)
    return df_pred, payout_map

# ==========================================
# Strategy Logic
# ==========================================

def get_actual_rank(df_race, rank):
    r = df_race[df_race['rank'] == rank]
    if r.empty: return -1
    return int(r.iloc[0]['horse_number'])

def round_stake(amount):
    """ Round to nearest 100 yen, minimum 0 (if extremely low) or 100?
    User Constraint: 100 yen units.
    Strategy: Round to nearest 100. If < 50, becomes 0 (no bet).
    """
    return int(round(amount / 100) * 100)

class StrategySimulator:
    def __init__(self, df, pm):
        self.df = df
        self.pm = pm
        self.df_sorted = df.sort_values(['race_id', 'score'], ascending=[True, False])
        
        # Pre-calc race info
        self.race_groups = {rid: grp for rid, grp in self.df_sorted.groupby('race_id')}
        self.actuals = {}
        for rid, grp in self.df[self.df['rank'].isin([1,2,3])].groupby('race_id'):
            self.actuals[rid] = {
                1: int(grp[grp['rank']==1]['horse_number'].iloc[0]) if not grp[grp['rank']==1].empty else -1,
                2: int(grp[grp['rank']==2]['horse_number'].iloc[0]) if not grp[grp['rank']==2].empty else -1,
                3: int(grp[grp['rank']==3]['horse_number'].iloc[0]) if not grp[grp['rank']==3].empty else -1
            }

    def simulate(self, strategy_func, staking_func, filter_func=None, label="Strategy"):
        """
        strategy_func(race_id, df_race) -> list of {'type': 'sanrentan', 'combo': '010203'}
        staking_func(bet_item, race_meta) -> amount (int)
        filter_func(race_meta) -> bool (True to bet, False to skip)
        """
        total_cost = 0
        total_return = 0
        hits = 0
        races_count = 0
        bets_count = 0
        
        # Iterate unique races
        unique_races = self.df_sorted['race_id'].unique()
        
        for rid in unique_races:
            grp = self.race_groups[rid]
            top_pick = grp.iloc[0]
            
            # 1. Filter
            if filter_func and not filter_func(top_pick, grp):
                continue
            
            # 2. Get Bets
            bets = strategy_func(rid, grp)
            if not bets: continue
            
            race_cost = 0
            race_return = 0
            hit_flag = False
            
            # 3. Process Bets
            for bet in bets:
                amount = staking_func(bet, top_pick)
                if amount <= 0: continue
                
                type_ = bet['type']
                combo = bet['combo'] # e.g. "010203"
                
                race_cost += amount
                bets_count += 1
                
                # Check Hit
                # Actual Payout
                pay = self.pm.get(rid, {}).get(type_, {}).get(combo, 0)
                if pay > 0:
                    # Payout is per 100 yen
                    ret = (pay * (amount / 100))
                    race_return += ret
                    hit_flag = True
            
            total_cost += race_cost
            total_return += race_return
            races_count += 1
            if hit_flag: hits += 1
            
        roi = (total_return / total_cost * 100) if total_cost > 0 else 0
        return {
            'label': label,
            'roi': roi,
            'profit': total_return - total_cost,
            'bets_avg': total_cost/100/races_count if races_count else 0,
            'hit_rate': hits/races_count*100 if races_count else 0,
            'races': races_count
        }

# ==========================================
# Concrete Strategies
# ==========================================

# --- 1. Betting Logic Generators ---

def strat_formation_adaptive(rid, grp):
    """
    Adaptive Formation:
    - Odds < 3.0: Formation 1->(2-5)->(2-8) (24 pts)
    - Odds >= 10.0: Nagashi 1->(2-7) (30 pts)
    - Else: Nagashi 1->(2-6) (20 pts)
    """
    top = grp.iloc[0]
    odds = top['odds']
    if pd.isna(odds): return []
    
    h1 = int(top['horse_number'])
    
    # Determine Opponents
    # Rank 2-X. Note: Rank in grp is implicit by order (grp is sorted by score)
    
    opps = grp.iloc[1:10] # grab top 10 opponents to be safe
    # Map rank to horse number
    rank_map = {i+2: int(row['horse_number']) for i, row in enumerate(opps.to_dict('records'))}
    # i=0 -> rank 2.
    
    bets = []
    
    if odds < 3.0:
        # Formation 1 -> 2-5 -> 2-8
        ranks2 = [2,3,4,5]
        ranks3 = [2,3,4,5,6,7,8]
        for r2 in ranks2:
            if r2 not in rank_map: continue
            h2 = rank_map[r2]
            for r3 in ranks3:
                if r2 == r3: continue
                if r3 not in rank_map: continue
                h3 = rank_map[r3]
                bets.append({'type': 'sanrentan', 'combo': f"{h1:02}{h2:02}{h3:02}"})
                
    elif odds >= 10.0:
        # Nagashi 1 -> 2-7
        opp_ranks = [2,3,4,5,6,7]
        for r2 in opp_ranks:
            if r2 not in rank_map: continue
            h2 = rank_map[r2]
            for r3 in opp_ranks:
                if r2 == r3: continue
                if r3 not in rank_map: continue
                h3 = rank_map[r3]
                bets.append({'type': 'sanrentan', 'combo': f"{h1:02}{h2:02}{h3:02}"})
    else:
        # Middle: Narrower Nagashi 1 -> 2-6 (20 pts)
        opp_ranks = [2,3,4,5,6]
        for r2 in opp_ranks:
            if r2 not in rank_map: continue
            h2 = rank_map[r2]
            for r3 in opp_ranks:
                if r2 == r3: continue
                if r3 not in rank_map: continue
                h3 = rank_map[r3]
                bets.append({'type': 'sanrentan', 'combo': f"{h1:02}{h2:02}{h3:02}"})
                
    return bets

def strat_box_sanrenpuku(rid, grp):
    """
    Box Strategy for Chaos (Low Confidence)
    Top 5 Box (10 pts)
    """
    # Pick Top 5
    top5 = grp.iloc[:5]['horse_number'].astype(int).tolist()
    if len(top5) < 3: return []
    
    bets = []
    for comb in combinations(top5, 3):
        # Sort for key
        s = sorted(comb)
        bets.append({'type': 'sanrenpuku', 'combo': f"{s[0]:02}{s[1]:02}{s[2]:02}"})
    return bets

def strat_perfect_portfolio(rid, grp):
    """
    The 'Perfect Portfolio' Strategy:
    1. Solid Match (Odds < 3.0, EV >= 1.0) -> Formation (2-5 / 2-8)
    2. Longshot Match (Odds >= 10.0, EV >= 1.3) -> Nagashi (2-7)
    3. Others -> Skip
    """
    top = grp.iloc[0]
    odds = top['odds']
    ev = top['expected_value']
    
    if pd.isna(odds) or pd.isna(ev): return []
    
    h1 = int(top['horse_number'])
    opps = grp.iloc[1:10] # Top 9 opponents
    rank_map = {i+2: int(row['horse_number']) for i, row in enumerate(opps.to_dict('records'))}
    
    bets = []
    
    # 1. Solid Match
    if odds < 3.0 and ev >= 1.0:
        # Formation: 1 -> 2-5 -> 2-8
        ranks2 = [2,3,4,5]
        ranks3 = [2,3,4,5,6,7,8]
        for r2 in ranks2:
            if r2 not in rank_map: continue
            h2 = rank_map[r2]
            for r3 in ranks3:
                if r2 == r3: continue
                if r3 not in rank_map: continue
                h3 = rank_map[r3]
                bets.append({'type': 'sanrentan', 'combo': f"{h1:02}{h2:02}{h3:02}"})
                
    # 2. Longshot Match
    elif odds >= 10.0 and ev >= 1.3:
        # Nagashi: 1 -> 2-7
        opp_ranks = [2,3,4,5,6,7]
        for r2 in opp_ranks:
            if r2 not in rank_map: continue
            h2 = rank_map[r2]
            for r3 in opp_ranks:
                if r2 == r3: continue
                if r3 not in rank_map: continue
                h3 = rank_map[r3]
                bets.append({'type': 'sanrentan', 'combo': f"{h1:02}{h2:02}{h3:02}"})
                
    return bets


# --- 2. Staking Functions ---

def stake_flat(bet, meta):
    return 100

def stake_ev_weighted(bet, meta):
    """
    EV-Weighted Staking (100 yen units)
    Amount = round(100 * EV / 100) * 100
    Minimum 100 yen.
    Example: EV 1.5 -> 150 -> 200 yen. EV 1.2 -> 120 -> 100 yen.
    """
    ev = meta['expected_value']
    # Aggressive scaling: Base 100 * EV
    raw_amount = 100 * ev
    # Round to nearest 100
    rounded = int(round(raw_amount / 100) * 100)
    return max(100, rounded)


# --- 3. Filters ---

def filter_all(top, grp):
    return True
    
# ... (existing filters) ...


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    import sys
    print("Loading Data...")
    try:
        df, pm = load_data('catboost_v7') # Default v7
        sim = StrategySimulator(df, pm)
        
        print(f"Loaded {len(df)} predictions.")
        
        # Define Strategies to Test
        strategies = [
            {
                'label': 'Perfect Portfolio (Flat Stakes)',
                'strat': strat_perfect_portfolio,
                'stake': stake_flat,
                'filter': filter_all
            },
            {
                'label': 'Perfect Portfolio (EV-Weighted)',
                'strat': strat_perfect_portfolio,
                'stake': stake_ev_weighted,
                'filter': filter_all
            }
        ]
        
        results = []
        for s in strategies:
            print(f"Running {s['label']}...")
            res = sim.simulate(s['strat'], s['stake'], s['filter'], label=s['label'])
            results.append(res)
            
        # Display
        results.sort(key=lambda x: x['roi'], reverse=True)
        print("\n" + "="*80)
        print(f"{'Strategy Name':<45} | {'ROI':<6} | {'Profit':<10} | {'Hit%':<5} | {'Races':<5}")
        print("-" * 80)
        for r in results:
            profit_str = f"{int(r['profit']):,}"
            print(f"{r['label']:<45} | {r['roi']:.1f}% | {profit_str:>10} | {r['hit_rate']:.1f}% | {r['races']}")
        print("="*80)
        
    except Exception as e:
        import traceback
        traceback.print_exc()

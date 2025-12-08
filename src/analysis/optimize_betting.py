
import pandas as pd
import numpy as np
import os
import itertools
from tabulate import tabulate
import argparse

def main():
    parser = argparse.ArgumentParser(description='Betting Strategy Optimizer (Formation)')
    parser.add_argument('--model', type=str, default='lgbm', help='Model name')
    parser.add_argument('--version', type=str, default='v4_1', help='Model version')
    args = parser.parse_args()

    # Paths
    exp_dir = os.path.join(os.path.dirname(__file__), '../../experiments')
    pred_path = os.path.join(exp_dir, f'predictions_{args.model}_{args.version}.parquet')
    payout_path = os.path.join(exp_dir, 'payouts_2024.parquet')

    if not os.path.exists(pred_path) or not os.path.exists(payout_path):
        print("Data files not found. Run evaluate.py first.")
        return

    # Load data
    df = pd.read_parquet(pred_path)
    payout_df = pd.read_parquet(payout_path)
    print(f"Loaded Predictions: {len(df)}, Payouts: {len(payout_df)}")

    # Pre-process Payouts into a dictionary for fast lookup
    # payout_map[race_id]['sanrentan'] = {'1-2-3': 1000, ...}
    payout_map = {}
    print("Building payout map...")
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'sanrentan': {}, 'sanrenpuku': {}, 'umaren': {}, 'wide': {}}
        
        # Sanrentan (Train)
        for i in range(1, 4):
            k_comb = f'haraimodoshi_sanrentan_{i}a'
            k_pay = f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = float(row[k_pay])
                    payout_map[rid]['sanrentan'][str(row[k_comb])] = pay
                except: pass

        # Sanrenpuku (Trio)
        for i in range(1, 4):
            k_comb = f'haraimodoshi_sanrenpuku_{i}a'
            k_pay = f'haraimodoshi_sanrenpuku_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = float(row[k_pay])
                    payout_map[rid]['sanrenpuku'][str(row[k_comb])] = pay
                except: pass

        # Umaren (Exacta)
        for i in range(1, 4):
            k_comb = f'haraimodoshi_umaren_{i}a'
            k_pay = f'haraimodoshi_umaren_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = float(row[k_pay])
                    payout_map[rid]['umaren'][str(row[k_comb])] = pay
                except: pass

        # Wide
        for i in range(1, 6): # Wide can have up to 7? usually 3-5 lines
            k_comb = f'haraimodoshi_wide_{i}a'
            k_pay = f'haraimodoshi_wide_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = float(row[k_pay])
                    payout_map[rid]['wide'][str(row[k_comb])] = pay
                except: pass

    df['expected_value'] = df['prob'] * df['odds'].fillna(0)
    
    # Grid Search Parameters
    min_probs = [0.1, 0.2]
    min_evs = [0.8, 1.0, 1.2]
    min_odds_list = [2.0, 3.0, 5.0]
    opponent_counts = [4, 5, 6] # Number of opponents
    
    bet_types = ['sanrentan', 'sanrenpuku', 'umaren', 'wide']

    # Group predictions
    df_grouped = df.sort_values(['race_id', 'score'], ascending=[True, False]).groupby('race_id')
    race_data = []
    for race_id, group in df_grouped:
        if race_id not in payout_map: continue
        race_data.append((race_id, group.to_dict('records')))

    print(f"Loaded {len(race_data)} races for simulation.")
    
import pandas as pd
import numpy as np
import os
import itertools
from tabulate import tabulate
import argparse

def main():
    parser = argparse.ArgumentParser(description='Betting Strategy Optimizer (Contextual)')
    parser.add_argument('--model', type=str, default='lgbm', help='Model name')
    parser.add_argument('--version', type=str, default='v4_1', help='Model version')
    args = parser.parse_args()

    # Paths
    exp_dir = os.path.join(os.path.dirname(__file__), '../../experiments')
    pred_path = os.path.join(exp_dir, f'predictions_{args.model}_{args.version}.parquet')
    payout_path = os.path.join(exp_dir, 'payouts_2024.parquet')

    if not os.path.exists(pred_path) or not os.path.exists(payout_path):
        print("Data files not found. Run evaluate.py first.")
        return

    # Load data
    df = pd.read_parquet(pred_path)
    payout_df = pd.read_parquet(payout_path)
    print(f"Loaded Predictions: {len(df)}, Payouts: {len(payout_df)}")

    # Pre-process Payouts
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'sanrentan': {}, 'sanrenpuku': {}, 'umaren': {}, 'wide': {}}
        
        # Helper to load columns
        def load_pay(type_key, col_prefix):
            for i in range(1, 8): # Support up to 7 for Wide
                k_comb = f'{col_prefix}_{i}a'
                k_pay = f'{col_prefix}_{i}b'
                if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                    try:
                        pay = float(row[k_pay])
                        payout_map[rid][type_key][str(row[k_comb])] = pay
                    except: pass

        load_pay('sanrentan', 'haraimodoshi_sanrentan')
        load_pay('sanrenpuku', 'haraimodoshi_sanrenpuku')
        load_pay('umaren', 'haraimodoshi_umaren')
        load_pay('wide', 'haraimodoshi_wide')

    df['expected_value'] = df['prob'] * df['odds'].fillna(0)

    # Helper for Payout Retrieval
    def get_payout(race_id, bet_type, h_nums):
        if race_id not in payout_map: return 0
        pm = payout_map[race_id][bet_type]
        h_nums_sorted = sorted(h_nums)
        k = ""
        if bet_type == 'sanrentan':
             k = f"{h_nums[0]:02}{h_nums[1]:02}{h_nums[2]:02}"
        elif bet_type == 'sanrenpuku':
             k = f"{h_nums_sorted[0]:02}{h_nums_sorted[1]:02}{h_nums_sorted[2]:02}"
        elif bet_type in ['umaren', 'wide']:
             k = f"{h_nums_sorted[0]:02}{h_nums_sorted[1]:02}"
        return pm.get(k, 0)

    # Group predictions
    df_grouped = df.sort_values(['race_id', 'score'], ascending=[True, False]).groupby('race_id')
    race_data = []
    for race_id, group in df_grouped:
        if race_id not in payout_map: continue
        race_data.append((race_id, group.to_dict('records')))

    print(f"Loaded {len(race_data)} races for simulation.")

    # --- Scenario Definitions ---
    # Based on Top Scorer's Odds
    scenarios = [
        {'name': 'Solid_Favorite',  'condition': lambda h: h['odds'] < 3.0,          'desc': 'Top Pick Odds < 3.0'},
        {'name': 'Middle_Hole',     'condition': lambda h: 3.0 <= h['odds'] < 10.0, 'desc': '3.0 <= Top Pick Odds < 10.0'},
        {'name': 'Longshot',        'condition': lambda h: h['odds'] >= 10.0,       'desc': 'Top Pick Odds >= 10.0'}
    ]

    bet_types = ['sanrentan', 'umaren', 'wide'] # Focus on these
    opponent_counts = [4, 5, 6, 7]
    min_evs = [0.8, 1.0, 1.2]
    
    # We want to find best strategy for each scenario
    
    for sc in scenarios:
        print(f"\n\n=== Scenario: {sc['name']} ({sc['desc']}) ===")
        
        best_rows = []
        
        for bet_type in bet_types:
            results = []
            params = list(itertools.product(min_evs, opponent_counts))
            
            for min_ev, opp_count in params:
                total_bet = 0
                total_return = 0
                hit_count = 0
                race_count = 0
                
                for race_id, horses in race_data:
                    top_horse = horses[0]
                    
                    # 1. Filter by Scenario (Does this race fit the context?)
                    if not sc['condition'](top_horse): continue
                    
                    # 2. Filter by Strategy (EV check)
                    # Note: We trust Top Horse as Axis, but maybe enforce EV
                    if top_horse['expected_value'] < min_ev: continue
                    
                    # 3. Opponents
                    if len(horses) < opp_count + 1: continue
                    opponents = horses[1 : opp_count + 1]
                    
                    axis_num = int(top_horse['horse_number'])
                    opp_nums = [int(h['horse_number']) for h in opponents]
                    
                    # Bet Calculation
                    bet_amount = 0
                    if bet_type == 'sanrentan':
                        # Axis 1 head -> Opps Nagashi (opp_count * (opp_count-1))
                        bet_amount = opp_count * (opp_count - 1) * 100
                    elif bet_type == 'umaren' or bet_type == 'wide':
                        # Axis -> Opps Nagashi (opp_count)
                        bet_amount = opp_count * 100
                    
                    # Check Result
                    race_return = 0
                    is_hit = 0
                    
                    actual_rank1 = next((h for h in horses if h['rank'] == 1), None)
                    actual_rank2 = next((h for h in horses if h['rank'] == 2), None)
                    actual_rank3 = next((h for h in horses if h['rank'] == 3), None) # Need for wide/3ren
                    
                    if not (actual_rank1 and actual_rank2):
                         # Data missing? Skip
                         continue

                    # Hit Check Logic
                    win_amt = 0
                    
                    if bet_type == 'sanrentan' and actual_rank3:
                        h1, h2, h3 = int(actual_rank1['horse_number']), int(actual_rank2['horse_number']), int(actual_rank3['horse_number'])
                        if h1 == axis_num and h2 in opp_nums and h3 in opp_nums:
                            win_amt += get_payout(race_id, 'sanrentan', [h1, h2, h3])
                            
                    elif bet_type == 'umaren':
                        h1, h2 = int(actual_rank1['horse_number']), int(actual_rank2['horse_number'])
                        winners = {h1, h2}
                        if axis_num in winners:
                            other = (winners - {axis_num}).pop()
                            if other in opp_nums:
                                win_amt += get_payout(race_id, 'umaren', [h1, h2])

                    elif bet_type == 'wide' and actual_rank3:
                        h1, h2, h3 = int(actual_rank1['horse_number']), int(actual_rank2['horse_number']), int(actual_rank3['horse_number'])
                        winners = [h1, h2, h3]
                        for w in winners:
                            if w == axis_num: continue
                            if w in opp_nums:
                                win_amt += get_payout(race_id, 'wide', [axis_num, w])

                    if win_amt > 0:
                        is_hit = 1
                        race_return += win_amt
                        
                    total_bet += bet_amount
                    total_return += race_return
                    hit_count += is_hit
                    race_count += 1
                
                if race_count < 20: continue # Skip low sample size
                
                roi = (total_return / total_bet) * 100 if total_bet > 0 else 0
                acc = hit_count / race_count * 100
                
                results.append({
                    'bet_type': bet_type,
                    'min_ev': min_ev,
                    'opp_count': opp_count,
                    'roi': roi,
                    'hit': acc,
                    'races': race_count,
                    'profit': total_return - total_bet
                })
            
            # Find best for this bet type in this scenario
            if results:
                df_res = pd.DataFrame(results)
                best = df_res.sort_values('roi', ascending=False).iloc[0]
                best_rows.append(best)
        
        if best_rows:
            print(tabulate(pd.DataFrame(best_rows), headers='keys', tablefmt='psql', floatfmt=".2f"))
        else:
            print("No valid strategies found (low sample size?)")

if __name__ == "__main__":
    main()

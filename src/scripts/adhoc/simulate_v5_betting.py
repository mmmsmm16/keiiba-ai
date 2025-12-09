
import pandas as pd
import numpy as np
import logging
import os
import sys
import itertools
from sqlalchemy import text

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from src.inference.loader import InferenceDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_predictions(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Ensure race_id is string
    df['race_id'] = df['race_id'].astype(str)
    return df

def fetch_payouts(race_ids):
    logger.info("Fetching Payout Data from jvd_hr...")
    loader = InferenceDataLoader()
    
    chunk_size = 1000
    payout_list = []
    
    unique_ids = list(set(race_ids))
    
    for i in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[i:i+chunk_size]
        ids_str = ",".join([f"'{rid}'" for rid in chunk])
        
        query = text(f"""
        SELECT
            CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
            haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
            haraimodoshi_umaren_2a, haraimodoshi_umaren_2b,
            haraimodoshi_wide_1a, haraimodoshi_wide_1b,
            haraimodoshi_wide_2a, haraimodoshi_wide_2b,
            haraimodoshi_wide_3a, haraimodoshi_wide_3b,
            haraimodoshi_wide_4a, haraimodoshi_wide_4b,
            haraimodoshi_wide_5a, haraimodoshi_wide_5b,
            haraimodoshi_wide_6a, haraimodoshi_wide_6b,
            haraimodoshi_wide_7a, haraimodoshi_wide_7b,
            haraimodoshi_sanrenpuku_1a, haraimodoshi_sanrenpuku_1b,
            haraimodoshi_sanrenpuku_2a, haraimodoshi_sanrenpuku_2b,
            haraimodoshi_sanrenpuku_3a, haraimodoshi_sanrenpuku_3b,
            haraimodoshi_sanrentan_1a, haraimodoshi_sanrentan_1b,
            haraimodoshi_sanrentan_2a, haraimodoshi_sanrentan_2b,
            haraimodoshi_sanrentan_3a, haraimodoshi_sanrentan_3b,
            haraimodoshi_sanrentan_4a, haraimodoshi_sanrentan_4b,
            haraimodoshi_sanrentan_5a, haraimodoshi_sanrentan_5b,
            haraimodoshi_sanrentan_6a, haraimodoshi_sanrentan_6b
        FROM jvd_hr
        WHERE CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) IN ({ids_str})
        """)
        
        try:
            with loader.engine.connect() as conn:
                tmp = pd.read_sql(query, conn)
                payout_list.append(tmp)
        except Exception as e:
            logger.error(f"Error fetching payouts for chunk {i}: {e}")

    payout_map = {}
    if payout_list:
        payout_df = pd.concat(payout_list)
        for _, row in payout_df.iterrows():
            rid = row['race_id']
            if rid not in payout_map: 
                payout_map[rid] = {'umaren': {}, 'wide': {}, 'sanrenpuku': {}, 'sanrentan': {}}
            
            # Helper to parse fields
            def parse_pay(prefix, count):
                d = {}
                for k in range(1, count+1):
                    comb = row.get(f'{prefix}_{k}a')
                    pay = row.get(f'{prefix}_{k}b')
                    if comb and pay and str(pay).strip():
                        try:
                            d[str(comb).strip()] = int(float(str(pay).strip()))
                        except: pass
                return d

            payout_map[rid]['umaren'] = parse_pay('haraimodoshi_umaren', 3)
            payout_map[rid]['wide'] = parse_pay('haraimodoshi_wide', 7)
            payout_map[rid]['sanrenpuku'] = parse_pay('haraimodoshi_sanrenpuku', 3)
            payout_map[rid]['sanrentan'] = parse_pay('haraimodoshi_sanrentan', 6)
            
    return payout_map

def simulate_strategies(df):
    logger.info("Starting Betting Simulation on v5 Predictions...")
    
    # Cleaning
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    # Calculate Pred Rank
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    # Fetch Payouts
    payout_map = fetch_payouts(df['race_id'].unique().tolist())
    logger.info(f"Loaded payouts for {len(payout_map)} races.")

    # Group by race
    races = df.groupby('race_id')
    
    # Strategies Definitions
    strategies = [
        # Win
        ('Win', 'Rank 1 Single', 1, lambda r1, r2, top: [[r1]]),
        ('Win', 'Adaptive (Diff<0.15 -> Rank 2)', 1, lambda r1, r2, top, diff=0: [[r2]] if diff < 0.15 else [[r1]]),
        
        # Umaren
        ('Umaren', 'Rank 1-2 (One Point)', 1, lambda r1, r2, top: [sorted([r1, r2])]),
        ('Umaren', 'Adaptive (Diff<0.03 -> 1-2 Only)', 1, lambda r1, r2, top, diff=0: [sorted([r1, r2])] if diff < 0.03 else []),
        
        # SanRenPuku
        ('SanRenPuku', 'Box 5 (Top 5)', 10, lambda r1, r2, top: list(itertools.combinations(top[:5], 3))),
        
        # SanRenTan (Advanced)
        ('SanRenTan', 'Adaptive Fold (Diff < 0.15)', 0, None), # Special logic
        ('SanRenTan', 'Gap: 1 > 3-6 > 2-6', 16, lambda r1, r2, top: [(r1, x, y) for x in top[2:6] for y in top[1:6] if y != x]),
        ('SanRenTan', 'Double Sandwich (1,2 > 3-6 > 1,2)', 8, lambda r1, r2, top: [(r1, x, r2) for x in top[2:6]] + [(r2, x, r1) for x in top[2:6]]),
        ('SanRenTan', '1st Fixed > 2nd/3rd (Top 6)', 20, lambda r1, r2, top: [(r1, x, y) for x, y in itertools.permutations(top[1:6], 2)]),
    ]
    
    print("\n=== 📊 Simulation Results (v5) ===")
    print(f"{'Type':<12} | {'Strategy':<35} | {'Cost':<6} | {'Hit Rate':<8} | {'Return':<8} | {'ROI':<8}")
    print("-" * 100)

    for st_type, st_name, st_avg_cost, st_func in strategies:
        total_bet = 0
        total_ret = 0
        total_hit = 0
        race_count = 0
        
        for rid, group in races:
            if rid not in payout_map: continue
            
            sorted_g = group.sort_values('pred_rank')
            if len(sorted_g) < 7: continue
            
            top_nums = sorted_g['horse_number'].astype(int).tolist()
            r1 = top_nums[0]; r2 = top_nums[1]
            
            r1_score = sorted_g.iloc[0]['score']
            r2_score = sorted_g.iloc[1]['score']
            diff = r1_score - r2_score
            
            combos = []
            
            # Special Logic for Adaptive Fold
            if st_name == 'Adaptive Fold (Diff < 0.15)':
                if diff < 0.15:
                    # Multi 1,2 -> 3rd (Rank 3-6) (8 pts)
                    opps = top_nums[2:6]
                    combos = [(x, y, z) for x, y in itertools.permutations([r1, r2], 2) for z in opps]
                else:
                    # 1st Fixed -> 2nd/3rd (Rank 2-6) (20 pts)
                    opps = top_nums[1:6]
                    combos = [(r1, x, y) for x, y in itertools.permutations(opps, 2)]
            elif st_name.startswith('Adaptive') and 'Diff' in st_name: # Simple adaptive wrapper
                 # Pass diff
                 combos = st_func(r1, r2, top_nums, diff=diff)
            else:
                 combos = st_func(r1, r2, top_nums)
                 
            if not combos: continue
            
            race_bet = len(combos)
            race_ret = 0
            hit_flag = 0
            
            for c in combos:
                # Format key
                if st_type == 'Win':
                    # Check rank directly from group (faster than checking payout?)
                    # But payout has actual return.
                    # Win Payout Key? JVD HR doesn't have Tansho in this query (Wait, I fetched Umaren/Wide/3Ren)
                    # I missed Tansho in existing robust script?
                    # The robust script fetched Umaren... but printed Win ROI using 'odds' column.
                    # OK, for WIN we use 'odds' column from input DF.
                    target = c[0]
                    winner = group[group['rank'] == 1]
                    if not winner.empty and int(winner['horse_number'].iloc[0]) == target:
                        race_ret += winner['odds'].iloc[0] * 100
                        hit_flag = 1
                
                elif st_type == 'Umaren':
                     c_s = sorted(c)
                     key = f"{c_s[0]:02}{c_s[1]:02}"
                     if key in payout_map[rid]['umaren']:
                         race_ret += payout_map[rid]['umaren'][key] / 100
                         hit_flag = 1
                         
                elif st_type == 'SanRenPuku':
                     c_s = sorted(c)
                     key = f"{c_s[0]:02}{c_s[1]:02}{c_s[2]:02}"
                     if key in payout_map[rid]['sanrenpuku']:
                         race_ret += payout_map[rid]['sanrenpuku'][key] / 100
                         hit_flag = 1
                         
                elif st_type == 'SanRenTan':
                     key = f"{c[0]:02}{c[1]:02}{c[2]:02}"
                     if key in payout_map[rid]['sanrentan']:
                         race_ret += payout_map[rid]['sanrentan'][key] / 100
                         hit_flag = 1
            
            total_bet += race_bet
            total_ret += race_ret
            if hit_flag: total_hit += 1
            race_count += 1
            
        roi = total_ret / total_bet * 100 if total_bet > 0 else 0
        hit_rate = total_hit / race_count if race_count > 0 else 0
        avg_c = total_bet / race_count if race_count > 0 else 0
        
        print(f"{st_type:<12} | {st_name:<35} | {avg_c:<6.1f} | {hit_rate:8.1%} | {total_ret:<8.1f} | {roi:8.1f}%")

if __name__ == "__main__":
    path = 'reports/predictions_v5_2025.csv'
    if os.path.exists(path):
        df = load_predictions(path)
        simulate_strategies(df)
    else:
        print(f"File not found: {path}")

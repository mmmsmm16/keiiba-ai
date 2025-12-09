import pandas as pd
import numpy as np
import logging
import itertools
from sqlalchemy import text
from typing import List, Dict, Any, Optional

from src.inference.loader import InferenceDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BettingSimulator:
    def __init__(self):
        self.loader = InferenceDataLoader()
        self.payout_map = {}

    def fetch_payouts(self, race_ids: List[str]):
        """Fetch payout data for listed race_ids efficiently."""
        logger.info("Fetching Payout Data...")
        chunk_size = 1000
        payout_list = []
        unique_ids = list(set(race_ids))
        
        for i in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[i:i+chunk_size]
            ids_str = ",".join([f"'{rid}'" for rid in chunk])
            
            query = text(f"""
            SELECT
                CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
                haraimodoshi_tansho_1a, haraimodoshi_tansho_1b,
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
                with self.loader.engine.connect() as conn:
                    tmp = pd.read_sql(query, conn)
                    payout_list.append(tmp)
            except Exception as e:
                logger.error(f"Error fetching payouts for chunk {i}: {e}")

        if payout_list:
            payout_df = pd.concat(payout_list)
            for _, row in payout_df.iterrows():
                rid = row['race_id']
                if rid not in self.payout_map: 
                    self.payout_map[rid] = {
                        'tansho': {}, 'umaren': {}, 'wide': {}, 'sanrenpuku': {}, 'sanrentan': {}
                    }
                
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

                self.payout_map[rid]['tansho'] = parse_pay('haraimodoshi_tansho', 1) # Tansho usually 3 slots but using 1 for simplicity main winner
                self.payout_map[rid]['umaren'] = parse_pay('haraimodoshi_umaren', 3)
                self.payout_map[rid]['wide'] = parse_pay('haraimodoshi_wide', 7)
                self.payout_map[rid]['sanrenpuku'] = parse_pay('haraimodoshi_sanrenpuku', 3)
                self.payout_map[rid]['sanrentan'] = parse_pay('haraimodoshi_sanrentan', 6)

    def run(self, predictions_df: pd.DataFrame, strategies: List[Dict[str, Any]]):
        """
        Run simulation for given strategies.
        strategies definition:
        [
            {
               "type": "SanRenTan",
               "name": "My Strategy",
               "formation": [[1], [2,3], [2,3,4,5]]  # List of Rank Lists.
               # Index 0 = 1st place Candidates (Ranks), Index 1 = 2nd place...
            } 
        ]
        """
        # Preprocessing
        df = predictions_df.copy()
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        # Calculate Rank within Race based on Score
        df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
        
        # Load Payouts
        race_ids = df['race_id'].unique().tolist()
        self.fetch_payouts(race_ids)
        
        results = {}
        
        grouped = df.groupby('race_id')
        
        for st in strategies:
            st_type = st.get('type') # SanRenTan, SanRenPuku, Umaren, Wide, Tansho
            st_name = st.get('name', 'Strategy')
            formation = st.get('formation', []) # [[1], [2,3]] means 1st: Rank1, 2nd: Rank2,3
            
            total_bet = 0
            total_ret = 0
            total_hit = 0
            race_count = 0
            
            st_key = f"{st_type}_{st_name}"
            daily_stats = {}
            
            for rid, group in grouped:
                if rid not in self.payout_map: continue
                
                # Sort by prediction rank
                sorted_g = group.sort_values('pred_rank')
                if len(sorted_g) < 6: continue # Skip small races?
                
                # Convert Rank Formation to Horse Numbers
                # formation is like [[1], [2,3], [1,2,3,4,5]]
                
                # Check max rank required
                max_rank_needed = 0
                for ranks in formation:
                    if ranks:
                        max_rank_needed = max(max_rank_needed, max(ranks))
                
                if len(sorted_g) < max_rank_needed: continue
                
                # Map Rank -> Horse Number
                # sorted_g is 0-indexed, so Rank R is at index R-1
                # But we need to handle if formation has large rank
                
                # Helper to get horse numbers for a list of ranks
                def get_horses(rank_list):
                    horses = []
                    for r in rank_list:
                        if r <= len(sorted_g):
                            # Rank 1 is index 0
                            horses.append(int(sorted_g.iloc[r-1]['horse_number']))
                    return horses

                tickets = []
                
                if st_type == 'SanRenTan':
                    # Expecting formation length 3: [1st_ranks, 2nd_ranks, 3rd_ranks]
                    if len(formation) < 3: continue
                    h1 = get_horses(formation[0])
                    h2 = get_horses(formation[1])
                    h3 = get_horses(formation[2])
                    
                    for f in h1:
                        for s in h2:
                            if f == s: continue
                            for t in h3:
                                if t == f or t == s: continue
                                tickets.append((f, s, t))
                                
                elif st_type == 'SanRenPuku':
                     # Typically [1st, 2nd, 3rd] but order doesn't matter for box/formation
                     # Usually for formation: 1st row is axis?
                     # Let's assume standard formation logic:
                     # 1. Expand all combinations
                     # 2. Filter valid (unique)
                     # 3. Sort and dedup
                     if len(formation) < 1: continue
                     
                     # Flatten formation or handle axis?
                     # Design decision: Frontend should send "Box" as one list [[1,2,3,4,5]]?
                     # Or "Axis" as [[1], [2,3,4,5,6]]?
                     # Let's support general product.
                     
                     potential_tickets = []
                     if len(formation) == 1: # Box
                         h_all = get_horses(formation[0])
                         for c in itertools.combinations(h_all, 3):
                             potential_tickets.append(tuple(sorted(c)))
                     elif len(formation) >= 2: # Axis
                         # e.g. [[1], [2,3,4,5]] -> Rank 1 AND one of Rank 2-5... wait Sanrenpuku usually needs 3 horses.
                         # Common formats:
                         # - Box: 1 list
                         # - 1-Axis: [Axis1], [Others] -> Pick 1 from Axis, 2 from Others? No.
                         # - Formation: Standard JRA style.
                         #   Form 1: A - B - C (Pick 1 from A, 1 from B, 1 from C)
                         #   Logic: Itertools product unique sorted.
                         lists = [get_horses(f) for f in formation]
                         if len(lists) < 3: 
                             # Maybe 1-2 axis?
                             # Let's simplify: User must provide 3 lists for product logic, even if same.
                             # e.g. Box 1-5 -> [[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]]
                             pass
                         
                         if len(lists) >= 3:
                             for c in itertools.product(lists[0], lists[1], lists[2]):
                                 if len(set(c)) == 3:
                                     potential_tickets.append(tuple(sorted(c)))
                                     
                     # Dedup
                     tickets = list(set(potential_tickets))
                     
                elif st_type == 'Umaren':
                     if len(formation) == 1: # Box
                         h_all = get_horses(formation[0])
                         potential_tickets = [tuple(sorted(c)) for c in itertools.combinations(h_all, 2)]
                     elif len(formation) >= 2: # 1-2
                         lists = [get_horses(f) for f in formation]
                         potential_tickets = []
                         for c in itertools.product(lists[0], lists[1]):
                             if c[0] != c[1]:
                                 potential_tickets.append(tuple(sorted(c)))
                     tickets = list(set(potential_tickets))
                     
                elif st_type == 'Wide':
                     # Same as Umaren
                     if len(formation) == 1: # Box
                         h_all = get_horses(formation[0])
                         potential_tickets = [tuple(sorted(c)) for c in itertools.combinations(h_all, 2)]
                     elif len(formation) >= 2: # 1-2
                         lists = [get_horses(f) for f in formation]
                         potential_tickets = []
                         for c in itertools.product(lists[0], lists[1]):
                             if c[0] != c[1]:
                                 potential_tickets.append(tuple(sorted(c)))
                     tickets = list(set(potential_tickets))
                     
                elif st_type == 'Tansho':
                     h_all = get_horses(formation[0])
                     tickets = [(h,) for h in h_all]

                if not tickets: continue
                
                # Evaluate
                race_bet = len(tickets)
                race_ret = 0
                hit_flag = 0
                
                pay_data = self.payout_map[rid]
                
                for t in tickets:
                    # Construct key
                    key = ""
                    val = 0
                    
                    if st_type == 'SanRenTan':
                        key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                        if key in pay_data['sanrentan']:
                            val = pay_data['sanrentan'][key]
                    
                    elif st_type == 'SanRenPuku':
                        key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                        if key in pay_data['sanrenpuku']:
                            val = pay_data['sanrenpuku'][key]
                            
                    elif st_type == 'Umaren':
                        key = f"{t[0]:02}{t[1]:02}"
                        if key in pay_data['umaren']:
                            val = pay_data['umaren'][key]
                            
                    elif st_type == 'Wide':
                        key = f"{t[0]:02}{t[1]:02}"
                        if key in pay_data['wide']:
                            val = pay_data['wide'][key]
                            
                    elif st_type == 'Tansho':
                        # Tansho key is tricky if multiple winners (dead heat)
                        # Helper stores as horse_num string -> payout?
                        # My fetch_payouts parser logic:
                        # self.payout_map[rid]['tansho'] = { '1': 320, '5': 200 } etc (horse number string)
                        k = str(t[0])
                        if k in pay_data['tansho']:
                            val = pay_data['tansho'][k]
                            
                    if val > 0:
                        race_ret += val / 100 # Unit bet 100 yen
                        hit_flag = 1
                
                total_bet += race_bet
                total_ret += race_ret
                if hit_flag: total_hit += 1
                race_count += 1

                # Daily Aggregation
                # Assuming df has 'date' column. If not, we might need to join it or extraction.
                # Since we grouped by race_id, we can pick any row's date.
                r_date = group['date'].iloc[0] if 'date' in group.columns else 'Unknown'
                
                if r_date not in daily_stats:
                    daily_stats[r_date] = {
                        'date': r_date,
                        'bet': 0, 
                        'return': 0, 
                        'hit': 0, 
                        'race_count': 0,
                        'races': [] # List of race results
                    }
                
                daily_stats[r_date]['bet'] += race_bet
                daily_stats[r_date]['return'] += race_ret
                daily_stats[r_date]['hit'] += hit_flag
                daily_stats[r_date]['race_count'] += 1
                
                # Race Detail (Minimal)
                daily_stats[r_date]['races'].append({
                    'race_id': rid,
                    'title': group.iloc[0]['start_time'] if 'start_time' in group.columns else rid, # Fallback title
                    'bet': race_bet,
                    'return': race_ret,
                    'hit': hit_flag > 0
                })
            
            roi = (total_ret / total_bet * 100) if total_bet > 0 else 0
            
            # Sort daily stats by date
            daily_list = sorted(list(daily_stats.values()), key=lambda x: x['date'])
            
            results[st_name] = {
                'summary': {
                    'total_bet': total_bet,
                    'total_return': total_ret,
                    'total_profit': total_ret - total_bet,
                    'hit_rate': (total_hit / race_count * 100) if race_count > 0 else 0,
                    'roi': roi,
                    'race_count': race_count
                },
                'daily': daily_list
            }
            
        return results

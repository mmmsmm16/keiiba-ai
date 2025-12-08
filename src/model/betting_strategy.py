
import pandas as pd
import numpy as np
import logging
from itertools import combinations, permutations
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BettingSimulator:
    """
    複合馬券（馬連、3連複など）のシミュレーションと最適化を行うクラス。
    """
    def __init__(self, df: pd.DataFrame, payout_df: pd.DataFrame = None):
        """
        Args:
            df (pd.DataFrame): 予測結果 (score, prob, etc.)
            payout_df (pd.DataFrame): 実際の払戻データ (jvd_hr)
        """
        self.df = df
        self.payout_df = payout_df
        self.payout_map = self._build_payout_map(payout_df) if payout_df is not None else {}

    def _build_payout_map(self, payout_df):
        logger.info("Building payout map...")
        pmap = {}
        for _, row in payout_df.iterrows():
            # race_id construction needs to match loader's race_id
            # Assuming payout_df has standard columns or pre-processed 'race_id'
            if 'race_id' in row:
                rid = row['race_id']
            else:
                # Fallback construction (jvd_hr standard)
                rid = str(row['kaisai_nen']) + str(row['keibajo_code']) + str(row['kaisai_kai']) + str(row['kaisai_nichime']) + str(row['race_bango'])
            
            pmap[rid] = {'umaren': {}, 'wide': {}, 'umatan': {}, 'sanrenpuku': {}, 'sanrentan': {}}
            
            # Map columns
            # Umaren
            for i in range(1, 4):
                k_comb = f'haraimodoshi_umaren_{i}a'
                k_pay = f'haraimodoshi_umaren_{i}b'
                if k_comb in row and row[k_comb]:
                    try:
                        pmap[rid]['umaren'][str(row[k_comb])] = int(row[k_pay])
                    except: pass
            
            # Wide
            for i in range(1, 8): # Wide usually up to 7 returns? 3 normally.
                k_comb = f'haraimodoshi_wide_{i}a'
                k_pay = f'haraimodoshi_wide_{i}b'
                if k_comb in row and row[k_comb]:
                    try:
                        pmap[rid]['wide'][str(row[k_comb])] = int(row[k_pay])
                    except: pass

            # Sanrenpuku
            for i in range(1, 4):
                k_comb = f'haraimodoshi_sanrenpuku_{i}a'
                k_pay = f'haraimodoshi_sanrenpuku_{i}b'
                if k_comb in row and row[k_comb]:
                    try:
                        pmap[rid]['sanrenpuku'][str(row[k_comb])] = int(row[k_pay])
                    except: pass

            # Sanrentan
            for i in range(1, 7):
                k_comb = f'haraimodoshi_sanrentan_{i}a'
                k_pay = f'haraimodoshi_sanrentan_{i}b'
                if k_comb in row and row[k_comb]:
                    try:
                        pmap[rid]['sanrentan'][str(row[k_comb])] = int(row[k_pay])
                    except: pass

        return pmap

    def simulate_formation(self, axis_threshold_score=None, axis_ev_threshold=None, axis_rank=1, opponent_ranks=[2,3,4,5,6], ticket_type='sanrenpuku'):
        """
        フォーメーション馬券のシミュレーション
        軸1頭 - 相手N頭 の流し
        
        Args:
            axis_threshold_score: 軸馬の最低スコア
            axis_ev_threshold: 軸馬の最低期待値 (expected_value)
            axis_rank: 順位
            opponent_ranks: 相手順位
            ticket_type: 券種
        """
        stats = {'bet': 0, 'return': 0, 'hit': 0, 'races': 0}
        
        for race_id, group in self.df.groupby('race_id'):
            if race_id not in self.payout_map:
                continue
                
            sorted_horses = group.sort_values('score', ascending=False)
            if len(sorted_horses) < max(opponent_ranks):
                continue
            
            # Select Axis
            axis_horse = sorted_horses.iloc[axis_rank-1]
            
            # Filters
            if axis_threshold_score and axis_horse['score'] < axis_threshold_score:
                continue
            if axis_ev_threshold and 'expected_value' in axis_horse and axis_horse['expected_value'] < axis_ev_threshold:
                continue
                
            axis_num = int(axis_horse['horse_number'])
            
            # Select Opponents
            opp_nums = []
            for r in opponent_ranks:
                if r <= len(sorted_horses):
                    opp_conf = sorted_horses.iloc[r-1]
                    opp_nums.append(int(opp_conf['horse_number']))
            
            if not opp_nums:
                continue
                
            # Generate Combinations
            bet_count = 0
            return_amount = 0
            hit_flag = 0
            
            combos = []
            
            if ticket_type == 'umaren':
                # Axis - Opp (N combos)
                combos = [(axis_num, o) for o in opp_nums]
                bet_count = len(combos)
                for c in combos:
                    c_str = f"{min(c):02}{max(c):02}"
                    if c_str in self.payout_map[race_id]['umaren']:
                        return_amount += self.payout_map[race_id]['umaren'][c_str]
                        hit_flag = 1
                        
            elif ticket_type == 'wide':
                 combos = [(axis_num, o) for o in opp_nums]
                 bet_count = len(combos)
                 for c in combos:
                    c_str = f"{min(c):02}{max(c):02}"
                    if c_str in self.payout_map[race_id]['wide']:
                        return_amount += self.payout_map[race_id]['wide'][c_str]
                        hit_flag = 1 # Wide multiple hits possible? Yes. usually sum returns.

            elif ticket_type == 'sanrenpuku':
                # Axis - Opp - Opp (NC2 combos)
                if len(opp_nums) >= 2:
                    opp_combos = list(combinations(opp_nums, 2))
                    bet_count = len(opp_combos)
                    for oc in opp_combos:
                        c = sorted([axis_num, oc[0], oc[1]])
                        c_str = f"{c[0]:02}{c[1]:02}{c[2]:02}"
                        if c_str in self.payout_map[race_id]['sanrenpuku']:
                            return_amount += self.payout_map[race_id]['sanrenpuku'][c_str]
                            hit_flag = 1

            elif ticket_type == 'sanrentan':
                # Axis(1st) - Opp(2nd) - Opp(3rd) + Multi?
                # Usually Formation means "Axis Fixed 1st" -> Opps 2nd/3rd
                # Or "Axis Multi" (Axis any place)
                # Here implement: Axis Fixed 1st (Nagashi)
                if len(opp_nums) >= 2:
                    opp_perms = list(permutations(opp_nums, 2))
                    bet_count = len(opp_perms)
                    for op in opp_perms:
                        # 1st: Axis, 2nd: op[0], 3rd: op[1]
                        c_str = f"{axis_num:02}{op[0]:02}{op[1]:02}"
                        if c_str in self.payout_map[race_id]['sanrentan']:
                            return_amount += self.payout_map[race_id]['sanrentan'][c_str]
                            hit_flag = 1

            if bet_count > 0:
                stats['bet'] += bet_count * 100
                stats['return'] += return_amount
                stats['hit'] += hit_flag
                stats['races'] += 1
                
        roi = stats['return'] / stats['bet'] * 100 if stats['bet'] > 0 else 0
        accuracy = stats['hit'] / stats['races'] if stats['races'] > 0 else 0
        
        return {
            'roi': roi, 
            'accuracy': accuracy, 
            'bet': stats['bet'], 
            'return': stats['return'], 
            'races': stats['races'],
            'params': {
                'axis_rank': axis_rank,
                'opp_ranks': opponent_ranks,
                'type': ticket_type
            }
        }

if __name__ == "__main__":
    # Test stub
    pass

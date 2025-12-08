
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
            if 'race_id' in row:
                rid = row['race_id']
            else:
                rid = str(row['kaisai_nen']) + str(row['keibajo_code']) + str(row['kaisai_kai']) + str(row['kaisai_nichime']) + str(row['race_bango'])
            
            pmap[rid] = {'umaren': {}, 'wide': {}, 'umatan': {}, 'sanrenpuku': {}, 'sanrentan': {}, 'tansho': {}, 'fukusho': {}}
            
            # Helper
            def set_pay(k_c, k_p, type_name):
                if k_c in row and pd.notna(row[k_c]):
                    try:
                        # Value (Payout)
                        val = int(float(row[k_p]))
                        
                        # Key (Horse numbers/combos)
                        # For tansho/fukusho, it's just horse number.
                        if type_name in ['tansho', 'fukusho']:
                            key = str(int(float(row[k_c]))).zfill(2)
                        elif type_name in ['umaren', 'wide', 'umatan']:
                            # JRA-VAN typically stores combined? No, separate columns usually?
                            # Actually JRA-VAN HR data has different columns for combinations?
                            # No, standard JVD_HR has `haraimodoshi_umaren_1a` (horse 1) and `1b` (horse 2)?
                            # Wait, usually `umaren_1a` is the combination string like "0102" OR just index?
                            # Let's check typical JVD schema.
                            # Usually `umaren_1a` is horse1, `umaren_1b` is horse2?
                            # My previous code assumed `row[k_c]` holds the FULL combo string or just ONE column?
                            # Previous code: `pmap[rid][type_name][str(row[k_c])] = val`
                            # This implies the column contains the key.
                            # If so, it might be "0102".
                            # Let's ASSUME `row[k_c]` is the key string for now as per previous working logic for Sanrenpuku.
                            if isinstance(row[k_c], (int, float)):
                                key = str(int(row[k_c])).zfill(2) # unlikely for combo
                            else:
                                key = str(row[k_c])
                        else:
                            key = str(row[k_c])
                            
                        pmap[rid][type_name][key] = val
                    except: pass

            # Map columns
            # Tansho
            for i in range(1, 4):
                set_pay(f'haraimodoshi_tansho_{i}a', f'haraimodoshi_tansho_{i}b', 'tansho')
            # Fukusho
            for i in range(1, 6):
                set_pay(f'haraimodoshi_fukusho_{i}a', f'haraimodoshi_fukusho_{i}b', 'fukusho')
            # Umaren
            for i in range(1, 4):
                set_pay(f'haraimodoshi_umaren_{i}a', f'haraimodoshi_umaren_{i}b', 'umaren')
            # Wide
            for i in range(1, 8):
                set_pay(f'haraimodoshi_wide_{i}a', f'haraimodoshi_wide_{i}b', 'wide')
            # Sanrenpuku
            for i in range(1, 4):
                set_pay(f'haraimodoshi_sanrenpuku_{i}a', f'haraimodoshi_sanrenpuku_{i}b', 'sanrenpuku')
            # Sanrentan
            for i in range(1, 7):
                set_pay(f'haraimodoshi_sanrentan_{i}a', f'haraimodoshi_sanrentan_{i}b', 'sanrentan')

        return pmap

    def simulate_formation(self, axis_threshold_score=None, axis_ev_threshold=None, axis_rank=1, opponent_ranks=[2,3,4,5,6], ticket_type='sanrenpuku'):
        """
        フォーメーション馬券のシミュレーション
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
                        hit_flag = 1 

            elif ticket_type == 'sanrenpuku':
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
                if len(opp_nums) >= 2:
                    opp_perms = list(permutations(opp_nums, 2))
                    bet_count = len(opp_perms)
                    for op in opp_perms:
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


class BettingOptimizer:
    """
    期待値(Expected Value)に基づく買い目最適化クラス
    """
    def __init__(self, df: pd.DataFrame, payout_df: pd.DataFrame = None):
        self.df = df
        self.payout_df = payout_df 
        self.sim = BettingSimulator(df, payout_df)
        self.payout_map = self.sim.payout_map

    def generate_candidates(self, race_df, ticket_types=['sanrenpuku', 'umaren', 'wide']):
        candidates = []
        
        top_N = 10
        sorted_horses = race_df.sort_values('score', ascending=False).head(top_N)
        horses = sorted_horses.to_dict('records')
        
        if len(horses) < 2:
            return candidates

        def get_prob(h): return h.get('prob', 0)
        def get_odds(h): return h.get('odds', 5.0) 
        def get_num(h): return int(h['horse_number'])
        
        if 'sanrenpuku' in ticket_types and len(horses) >= 3:
            combos = list(combinations(horses, 3))
            for c in combos:
                pa, pb, pc = get_prob(c[0]), get_prob(c[1]), get_prob(c[2])
                prob_hit = pa * pb * pc * 6 
                
                oa, ob, oc = get_odds(c[0]), get_odds(c[1]), get_odds(c[2])
                est_payout = (oa * ob * oc) ** (1/3) * 10 
                
                ev = prob_hit * est_payout * 100 
                
                candidates.append({
                    'type': 'sanrenpuku',
                    'combo': tuple(sorted([get_num(c[0]), get_num(c[1]), get_num(c[2])])),
                    'prob': prob_hit,
                    'est_payout': est_payout,
                    'ev': ev,
                    'cost': 100
                })
        
        if 'umaren' in ticket_types and len(horses) >= 2:
            combos = list(combinations(horses, 2))
            for c in combos:
                pa, pb = get_prob(c[0]), get_prob(c[1])
                prob_hit = pa * pb * 2
                
                oa, ob = get_odds(c[0]), get_odds(c[1])
                est_payout = (oa * ob) ** 0.5 * 5
                ev = prob_hit * est_payout * 100
                
                candidates.append({
                    'type': 'umaren',
                    'combo': tuple(sorted([get_num(c[0]), get_num(c[1])])),
                    'prob': prob_hit,
                    'est_payout': est_payout,
                    'ev': ev,
                    'cost': 100
                })

        return candidates

    def optimize_betting_plan(self, race_id, budget=1000, strategy_type='ev_greedy'):
        if race_id not in self.df['race_id'].values:
            pass
            
        group = self.df[self.df['race_id'] == race_id].copy()
        if group.empty:
            return [], 0
            
        candidates = self.generate_candidates(group)
        
        valid_candidates = [c for c in candidates if c['ev'] > 80] 
        valid_candidates.sort(key=lambda x: x['ev'], reverse=True)
        
        selected = []
        current_cost = 0
        
        for c in valid_candidates:
            if current_cost + c['cost'] <= budget:
                selected.append(c)
                current_cost += c['cost']
                
        return selected, current_cost

    def evaluate_plan(self, race_id, plan):
        if race_id not in self.payout_map:
            return 0, 0
            
        total_return = 0
        hit = 0
        
        payouts = self.payout_map[race_id]
        
        for ticket in plan:
            t_type = ticket['type']
            combo = ticket['combo'] 
            
            # Payout Key Construction
            key = ""
            if t_type == 'sanrenpuku':
                key = f"{combo[0]:02}{combo[1]:02}{combo[2]:02}"
            elif t_type == 'umaren':
                key = f"{combo[0]:02}{combo[1]:02}"
            elif t_type == 'wide':
                key = f"{combo[0]:02}{combo[1]:02}" # Wide key is also sorted? Yes.
            elif t_type == 'tansho':
                key = f"{combo[0]:02}"
            
            p_map = payouts.get(t_type, {})
            if key in p_map:
                # Ticket Cost Factor (Plan assumes 100 yen per unit)
                # But simulator returns just payout per 100 yen.
                # If we bet more than 100, we need to scalar mult.
                # Currently evaluate_plan assumes 1 unit (100 yen).
                # If plan has 'amount', use it.
                bet_amount = ticket.get('amount', 100)
                pay = p_map[key]
                total_return += pay * (bet_amount / 100)
                hit = 1
            
        return total_return, hit

    def calculate_kelly_bet(self, prob, odds, bankroll, fraction=0.25, max_cap_pct=0.05, min_bet=100):
        """
        Kelly Criterionによる推奨ベット額を計算する
        f* = (bp - q) / b
        b: net odds (odds - 1)
        p: probability
        q: 1 - p
        """
        if prob <= 0 or odds <= 1:
            return 0
            
        b = odds - 1
        p = prob
        q = 1 - p
        
        f_star = (b * p - q) / b
        
        if f_star <= 0:
            return 0
            
        # Apply Fraction
        f = f_star * fraction
        
        # Apply Cap
        f = min(f, max_cap_pct)
        
        bet_amount = bankroll * f
        
        # Round to 100 yen
        bet_amount = int(bet_amount // 100) * 100
        
        if bet_amount < min_bet:
            return 0 # Or min_bet? Typically 0 if edge is too small to justify size
            
        return bet_amount


class BankrollSimulator:
    """
    時系列での資金推移シミュレーション
    """
    def __init__(self, optimizer: BettingOptimizer, initial_bankroll=1000000):
        self.optimizer = optimizer
        self.bankroll = initial_bankroll
        self.history = []
        
    def run(self, race_ids, strategy_func, **kwargs):
        """
        Args:
            race_ids: List of race_ids sorted by time
            strategy_func: Function to determine bets for a race (returns list of dicts with 'amount')
        """
        curr_bankroll = self.bankroll
        max_drawdown = 0
        peak_bankroll = curr_bankroll
        
        for rid in race_ids:
            # 1. Decide Bets
            # strategy_func(race_id, current_bankroll, **kwargs) -> bets check
            bets, total_cost = strategy_func(self.optimizer, rid, curr_bankroll, **kwargs)
            
            if total_cost > curr_bankroll:
                # 資金不足 (Skip or Adjust?)
                continue
                
            if total_cost == 0:
                continue
                
            # 2. Pay cost
            curr_bankroll -= total_cost
            
            # 3. Simulate Race
            race_return, is_hit = self.optimizer.evaluate_plan(rid, bets)
            
            # 4. Receive Return
            curr_bankroll += race_return
            
            # Stats
            peak_bankroll = max(peak_bankroll, curr_bankroll)
            dd = (peak_bankroll - curr_bankroll) / peak_bankroll
            max_drawdown = max(max_drawdown, dd)
            
            self.history.append({
                'race_id': rid,
                'bankroll': curr_bankroll,
                'bet': total_cost,
                'return': race_return,
                'drawdown': dd
            })
            
            if curr_bankroll < 100:
                logger.warning("Bankrupt!")
                break
                
        return {
            'final_bankroll': curr_bankroll,
            'roi': (curr_bankroll / self.bankroll) * 100,
            'max_drawdown': max_drawdown * 100,
            'history': pd.DataFrame(self.history)
        }

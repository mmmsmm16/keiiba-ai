import numpy as np
import logging
from typing import Dict, List, Optional
import yaml
import itertools

logger = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def decide_bets(self, 
                    race_id: str, 
                    preds: Dict,  # {h_id: {rank, p_cal, ...}, metrics: {p1, margin...}}
                    odds: Dict,   # {tansho: {h:o}, umaren: {(h1,h2):o}}
                    budgets: Dict # {race_cap_remaining: ...}
                   ) -> Dict:
        """
        Decide bets for a single race.
        Returns:
            {
                'actions': [
                    {'type': 'win', 'buy': [h], 'amount': 100, 'reason': '...'},
                    ...
                ],
                'logs': ...
            }
        """
        
        c_strats = self.config['strategies']
        metrics = preds['metrics']
        
        # Sort horses by p_cal
        # preds['horses'] need to be list?
        # Let's assume input is list of dicts
        horses = sorted(preds['horses'], key=lambda x: x['p_cal'], reverse=True)
        top_h = horses[0]
        
        decisions = []
        
        # --- 1. Evaluate Win Core ---
        win_cfg = c_strats['win_core']
        win_decision = None
        if win_cfg['enabled']:
            # Odds
            h1 = top_h['horse_number']
            o1 = odds['tansho'].get(h1, 0.0)
            
            p1 = metrics['p1']
            margin = metrics['margin']
            ev = p1 * o1
            
            th = win_cfg['thresholds']
            max_odds = win_cfg.get('max_odds', 999.9) # Guardrail
            
            if p1 >= th['p1'] and margin >= th['margin'] and ev >= th['ev'] and o1 < max_odds:
                win_decision = {
                    'strategy': 'win_core',
                    'type': win_cfg['bet_type'],
                    'target': [h1],
                    'amount': win_cfg['amount_fixed'],
                    'priority': win_cfg['priority'],
                    'details': f"p1={p1:.2f}, m={margin:.2f}, ev={ev:.1f}",
                    'buy': True
                }
            else:
                 win_decision = {
                    'strategy': 'win_core',
                    'buy': False,
                    'reason': f"条件未達 (p1={p1:.2f}, m={margin:.2f}, ev={ev:.1f})"
                }
        
        # --- 2. Evaluate Umaren (High & Balanced) ---
        # Generate candidates (Top K Box)
        # Calculate Harville Probs
        # Calculate EV
        
        umaren_cands = []
        k = self.config['prediction']['umaren_k']
        top_k_horses = horses[:k]
        
        # Normalize p_cal for Harville within Top K? 
        # Or use global p_cal. Harville uses relative.
        # Let's use raw p_cal as-is for approximation (or re-normalize).
        # optimize_multiticket.py logic:
        # p_pair = pi * pj * ...
        
        for i in range(len(top_k_horses)):
            for j in range(i+1, len(top_k_horses)):
                 hi = top_k_horses[i]
                 hj = top_k_horses[j]
                 
                 pi = hi['p_cal']
                 pj = hj['p_cal']
                 
                 # Harville approx
                 den_i = 1.0 - pi
                 den_j = 1.0 - pj
                 if den_i <= 0: den_i = 1e-9
                 if den_j <= 0: den_j = 1e-9
                 
                 p_pair = pi * pj * (1.0/den_i + 1.0/den_j)
                 
                 pair_key = tuple(sorted((hi['horse_number'], hj['horse_number'])))
                 o_pair = odds['umaren'].get(pair_key, 0.0)
                 ev_pair = p_pair * o_pair
                 
                 umaren_cands.append({
                     'pair': pair_key,
                     'p_pair': p_pair,
                     'odds': o_pair,
                     'ev': ev_pair
                 })
        
        # Sort candidates by EV desc
        umaren_cands.sort(key=lambda x: x['ev'], reverse=True)
        
        # 2a. Umaren High
        high_cfg = c_strats['umaren_high']
        high_decision = None
        if high_cfg['enabled']:
            # Filter
            valid_high = [c for c in umaren_cands if c['ev'] >= high_cfg['thresholds']['ev']]
            if valid_high:
                # Take top 1
                picks = valid_high[:high_cfg['max_pairs']]
                amt = high_cfg['amount_fixed'] * len(picks)
                high_decision = {
                    'strategy': 'umaren_high',
                    'type': 'umaren',
                    'target': [p['pair'] for p in picks],
                    'amount': amt,
                    'priority': high_cfg['priority'],
                    'details': f"TopEV={picks[0]['ev']:.1f}",
                    'buy': True
                }
            else:
                high_decision = {'strategy': 'umaren_high', 'buy': False, 'reason': '条件未達 (EV<10)'}

        # 2b. Umaren Balanced
        bal_cfg = c_strats['umaren_balanced']
        bal_decision = None
        if bal_cfg['enabled']:
            # Rule: High Priority fires? Skip Balanced?
            # User spec: "HighReturnが発火したレースは、Balancedは買わない"
            if high_decision and high_decision['buy']:
                bal_decision = {'strategy': 'umaren_balanced', 'buy': False, 'reason': '上位戦略(High)発動中'}
            else:
                th = bal_cfg['thresholds']
                valid_bal = [c for c in umaren_cands if c['ev'] >= th['ev'] and c['p_pair'] >= th['p_pair']]
                if valid_bal:
                    picks = valid_bal[:bal_cfg['max_pairs']]
                    amt = bal_cfg['amount_fixed'] * len(picks)
                    # Check race cap specific to Balanced? (Optionally)
                    if amt > bal_cfg.get('budget_cap_race', 9999):
                        # Trim? Simplified: just take whatever fits or cap amount
                        amt = bal_cfg.get('budget_cap_race', 9999)
                        
                    bal_decision = {
                        'strategy': 'umaren_balanced',
                        'type': 'umaren',
                        'target': [p['pair'] for p in picks],
                        'amount': amt,
                        'priority': bal_cfg['priority'],
                        'details': f"Count={len(picks)}, TopEV={picks[0]['ev']:.1f}",
                        'buy': True
                    }
                else:
                    bal_decision = {'strategy': 'umaren_balanced', 'buy': False, 'reason': '条件未達 (EV<4.0 or P<1.5%)'}

        # --- 3. Wide Frequency ---
        wide_cfg = c_strats['wide_freq']
        wide_decision = None
        if wide_cfg['enabled']:
            # Rule: Skip if others are BUY
            any_other_buy = (win_decision and win_decision['buy']) or \
                            (high_decision and high_decision['buy']) or \
                            (bal_decision and bal_decision['buy'])
            
            if any_other_buy:
                 wide_decision = {'strategy': 'wide_freq', 'buy': False, 'reason': '他戦略BUYのため見送り'}
            else:
                th = wide_cfg['thresholds']
                if metrics['p1'] >= th['p1'] and metrics['margin'] >= th['margin']:
                     # W1: Top1-Top2
                     h1 = horses[0]['horse_number']
                     h2 = horses[1]['horse_number']
                     pair = tuple(sorted((h1, h2)))
                     
                     wide_decision = {
                        'strategy': 'wide_freq',
                        'type': 'wide',
                        'target': [pair],
                        'amount': wide_cfg['amount_fixed'],
                        'priority': wide_cfg['priority'],
                        'details': f"p1={metrics['p1']:.2f} (Freq)",
                        'buy': True
                     }
                else:
                    wide_decision = {'strategy': 'wide_freq', 'buy': False, 'reason': '条件未達 (p1<0.4 or m<0.1)'}

        # --- 4. Budget Resolution ---
        all_proposals = [d for d in [high_decision, bal_decision, win_decision, wide_decision] if d and d['buy']]
        all_proposals.sort(key=lambda x: x['priority']) # 1 is Highest
        
        final_bets = []
        spent_race = 0
        race_cap = budgets.get('race_cap', 9999)
        day_cap = budgets.get('day_cap', 99999)
        current_day_spent = budgets.get('current_day_spent', 0)
        
        for prop in all_proposals:
            cost = prop['amount']
            
            # Check Race Cap
            if spent_race + cost > race_cap:
                prop['buy'] = False
                prop['reason'] = 'レース予算上限超過'
                continue
                
            # Check Day Cap
            if current_day_spent + spent_race + cost > day_cap:
                 prop['buy'] = False
                 prop['reason'] = '1日予算上限超過'
                 continue
                 
            final_bets.append(prop)
            spent_race += cost
            
            # --- Skip Logic (Unreachable as intended if we append) ---
            # prop['buy'] = False
            # prop['reason'] = 'レース予算上限超過'
            # Keep tracked as skipped
        
        return {
            'decisions': [d for d in [high_decision, bal_decision, win_decision, wide_decision] if d],
            'final_bets': final_bets,
            'total_cost': spent_race
        }

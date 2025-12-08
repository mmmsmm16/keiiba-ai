"""
オッズ帯別戦略シミュレーション (Odds-based Strategy Optimization)

AIスコア1位のオッズに応じて買い方を変える戦略を検証する。
- 本命サイド (オッズ < 3.0): 3連単7頭流し or ワイド
- 中穴サイド (3.0 <= オッズ < 10.0): 3連単6頭流し
- 大穴サイド (オッズ >= 10.0): ワイド or 単勝

各オッズ帯で最適な券種・点数を探索する。
"""
import pandas as pd
import numpy as np
import os
from itertools import product

# Load data
EXPERIMENTS_DIR = '/workspace/experiments'

def load_data(model_name='lgbm_v4_1'):
    pred_path = os.path.join(EXPERIMENTS_DIR, f'predictions_{model_name}.parquet')
    payout_path = os.path.join(EXPERIMENTS_DIR, 'payouts_2024.parquet')
    
    df_pred = pd.read_parquet(pred_path)
    df_pay = pd.read_parquet(payout_path)
    
    # Build payout map
    payout_map = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'sanrentan': {}, 'wide': {}, 'umaren': {}, 'sanrenpuku': {}}
        
        for i in range(1, 8):
            for typ, prefix in [
                ('sanrentan', 'haraimodoshi_sanrentan'),
                ('wide', 'haraimodoshi_wide'),
                ('umaren', 'haraimodoshi_umaren'),
                ('sanrenpuku', 'haraimodoshi_sanrenpuku')
            ]:
                k_comb = f'{prefix}_{i}a'
                k_pay = f'{prefix}_{i}b'
                if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                    try:
                        payout_map[rid][typ][str(row[k_comb])] = float(row[k_pay])
                    except:
                        pass
    
    df_pred['race_id'] = df_pred['race_id'].astype(str)
    return df_pred, payout_map


def simulate_odds_based_strategy(df, pm, strategy_config):
    """
    既存のオッズ帯別戦略シミュレーション
    """
    total_cost = 0; total_return = 0; details = {'low': {'cost':0,'ret':0,'races':0,'hits':0}, 'mid': {'cost':0,'ret':0,'races':0,'hits':0}, 'high': {'cost':0,'ret':0,'races':0,'hits':0}}
    for rid, grp in df.groupby('race_id'):
        sub = grp.sort_values('score', ascending=False)
        if sub.empty: continue
        top = sub.iloc[0]
        odds = top['odds']
        if pd.isna(odds) or odds <= 0: continue

        if odds < strategy_config['low']['threshold']: zone = 'low'; cfg = strategy_config['low']
        elif odds < strategy_config['mid']['threshold']: zone = 'mid'; cfg = strategy_config['mid']
        else: zone = 'high'; cfg = strategy_config['high']
        
        bet_type = cfg['bet_type']
        opp_count = cfg['opp_count']
        if bet_type=='sanrentan': pts=opp_count*(opp_count-1)
        elif bet_type=='wide': pts=opp_count
        elif bet_type=='tansho': pts=1
        elif bet_type=='umaren': pts=opp_count
        else: pts=opp_count
        
        cost = pts * 100
        
        # Payout Logic
        actual_1 = sub[sub['rank'] == 1]; h1 = int(actual_1['horse_number'].iloc[0]) if not actual_1.empty else -1
        actual_2 = sub[sub['rank'] == 2]; h2 = int(actual_2['horse_number'].iloc[0]) if not actual_2.empty else -1
        actual_3 = sub[sub['rank'] == 3]; h3 = int(actual_3['horse_number'].iloc[0]) if not actual_3.empty else -1
        axis = int(top['horse_number'])
        opps = sub.iloc[1:opp_count+1]['horse_number'].tolist()
        opp_nums = [int(x) for x in opps if not pd.isna(x)]
        payout = 0; hit = False
        
        if bet_type == 'sanrentan':
            if h1 == axis and h2 in opp_nums and h3 in opp_nums:
                k = f"{h1:02}{h2:02}{h3:02}"
                payout = pm.get(rid, {}).get('sanrentan', {}).get(k, 0)
        elif bet_type == 'wide':
            for w in [h1, h2, h3]:
                if w != axis and w in opp_nums:
                    k = f"{min(axis,w):02}{max(axis,w):02}"
                    payout += pm.get(rid, {}).get('wide', {}).get(k, 0)
        elif bet_type == 'tansho':
            if h1 == axis: payout = odds * 100
            
        if payout > 0: hit = True
        total_cost += cost; total_return += payout
        details[zone]['cost'] += cost; details[zone]['ret'] += payout
        details[zone]['races'] += 1
        if hit: details[zone]['hits'] += 1
            
    return {'total_cost': total_cost, 'total_return': total_return, 'total_roi': (total_return/total_cost*100) if total_cost>0 else 0, 'details': details}


def find_golden_rules(df, pm):
    """ ROI 100%超えを目指す厳選条件探索 """
    print("Preparing data for Golden Rule Search...")
    df_sorted = df.sort_values(['race_id', 'score'], ascending=[True, False])
    top_picks = df_sorted.drop_duplicates(subset=['race_id'], keep='first').copy()
    
    # 実際の結果マップ
    actual_map = {}
    for rid, grp in df[df['rank'].isin([1, 2, 3])].groupby('race_id'):
        actual_map[rid] = {
            1: grp[grp['rank']==1]['horse_number'].iloc[0] if not grp[grp['rank']==1].empty else -1,
            2: grp[grp['rank']==2]['horse_number'].iloc[0] if not grp[grp['rank']==2].empty else -1,
            3: grp[grp['rank']==3]['horse_number'].iloc[0] if not grp[grp['rank']==3].empty else -1
        }
        
    # Opponents Map
    opp_df = df_sorted.groupby('race_id').head(8) # ample
    opp_map = opp_df.groupby('race_id')['horse_number'].apply(list).to_dict()

    # Pre-calc columns
    t_pays = []; s_pays = []
    
    for _, row in top_picks.iterrows():
        rid = row['race_id']
        axis = int(row['horse_number'])
        acts = actual_map.get(rid, {1:-1, 2:-1, 3:-1})
        h1, h2, h3 = acts[1], acts[2], acts[3]
        
        # Tansho
        tp = row['odds'] * 100 if h1 == axis else 0
        t_pays.append(tp)
        
        # Sanrentan (1->6opps)
        sp = 0
        opps = opp_map.get(rid, [])
        opps6 = [int(x) for x in opps if int(x) != axis][:6]
        if h1 == axis and h2 in opps6 and h3 in opps6:
            k = f"{h1:02}{h2:02}{h3:02}"
            sp = pm.get(rid, {}).get('sanrentan', {}).get(k, 0)
        s_pays.append(sp)
        
    top_picks['payout_tansho'] = t_pays
    top_picks['payout_sanrentan'] = s_pays
    
    # Grid Search
    bet_types = ['tansho', 'sanrentan']
    min_probs = [0.0, 0.20, 0.25, 0.30]
    min_evs = [0.0, 1.0, 1.1, 1.2, 1.3]
    odds_filters = [
        {'label': 'All', 'min': 0, 'max': 9999},
        {'label': 'Solid(<3.0)', 'min': 0, 'max': 3.0},
        {'label': 'Middle(3.0-10.0)', 'min': 3.0, 'max': 10.0},
        {'label': 'High(10.0+)', 'min': 10.0, 'max': 9999},
        {'label': 'Over5(5.0+)', 'min': 5.0, 'max': 9999},
    ]
    
    results = []
    sanrentan_cost = 3000; tansho_cost = 100
    
    print("Running Grid Search (Instant)...")
    for bt in bet_types:
        base_cost = tansho_cost if bt == 'tansho' else sanrentan_cost
        col = 'payout_tansho' if bt == 'tansho' else 'payout_sanrentan'
        
        for mp in min_probs:
            for me in min_evs:
                for of in odds_filters:
                    mask = (top_picks['prob']>=mp) & (top_picks['expected_value']>=me) & (top_picks['odds']>=of['min']) & (top_picks['odds']<of['max'])
                    sub = top_picks[mask]
                    count = len(sub)
                    if count < 20: continue
                    ret = sub[col].sum()
                    cost = count * base_cost
                    roi = (ret/cost*100) if cost>0 else 0
                    if roi > 100:
                        results.append({'bet_type': bt, 'prob': mp, 'ev': me, 'odds_label': of['label'], 'roi': roi, 'bets': count, 'hit_rate': (sub[col]>0).mean()*100, 'profit': ret-cost})

    results.sort(key=lambda x: x['roi'], reverse=True)
    print("\n【Golden Rules Top 15 (ROI > 100%)】")
    for r in results[:15]:
        print(f"{r['bet_type']} | Prob>={r['prob']:.2f}, EV>={r['ev']:.1f}, Odds={r['odds_label']} | ROI: {r['roi']:.1f}% (Bets: {r['bets']}, Hit: {r['hit_rate']:.1f}%, Profit: {r['profit']:,})")


def find_ev_based_axis_strategy(df, pm):
    """ Test Top 3 EV Axis """
    print("Preparing data for EV-based Axis Search (Top 3)...")
    df['rank_in_race'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    top_n_df = df[df['rank_in_race'] <= 3].copy()
    
    actual_map = {}
    for rid, grp in df[df['rank'].isin([1, 2, 3])].groupby('race_id'):
        actual_map[rid] = {1: grp[grp['rank']==1]['horse_number'].iloc[0] if not grp[grp['rank']==1].empty else -1, 2: grp[grp['rank']==2]['horse_number'].iloc[0] if not grp[grp['rank']==2].empty else -1, 3: grp[grp['rank']==3]['horse_number'].iloc[0] if not grp[grp['rank']==3].empty else -1}
        
    opp_df = df[df['rank_in_race'] <= 8]
    opp_map = opp_df.groupby('race_id')['horse_number'].apply(list).to_dict()
    
    strategies = [{'label': 'MaxEV_Axis (Multi)', 'is_multi': True}, {'label': 'MaxEV_Axis (Nagashi)', 'is_multi': False}]
    results = []
    
    for strat in strategies:
        total_cost = 0; total_ret = 0; hits = 0; races = 0
        opp_count = 6; points = opp_count * (opp_count-1) * (3 if strat['is_multi'] else 1)
        cost_per_race = points * 100
        
        for rid, grp in top_n_df.groupby('race_id'):
            cands = grp[grp['expected_value'] >= 1.0]
            if cands.empty: continue
            axis = int(cands.loc[cands['expected_value'].idxmax()]['horse_number'])
            
            all_opps = opp_map.get(rid, [])
            clean_opps = [int(x) for x in all_opps if int(x) != axis]
            sel_opps = set(clean_opps[:opp_count])
            
            acts = actual_map.get(rid, {1:-1, 2:-1, 3:-1})
            h1,h2,h3 = int(acts[1]), int(acts[2]), int(acts[3])
            
            payout = 0
            if strat['is_multi']:
                winners = {h1, h2, h3}
                if axis in winners:
                    rem = list(winners - {axis})
                    if len(rem) == 2 and rem[0] in sel_opps and rem[1] in sel_opps:
                         k = f"{h1:02}{h2:02}{h3:02}"
                         payout = pm.get(rid, {}).get('sanrentan', {}).get(k, 0)
            else:
                if h1 == axis and h2 in sel_opps and h3 in sel_opps:
                    k = f"{h1:02}{h2:02}{h3:02}"
                    payout = pm.get(rid, {}).get('sanrentan', {}).get(k, 0)
                    
            if payout > 0: hits += 1
            total_ret += payout; total_cost += cost_per_race; races += 1
            
        roi = (total_ret/total_cost*100) if total_cost>0 else 0
        results.append({'label': strat['label'], 'roi': roi, 'bets': races, 'hit': hits/races*100})
        
    print("\n【EV-based Axis Strategy Results】")
    for r in results: print(f"{r['label']} | ROI: {r['roi']:.1f}% (Bets: {r['bets']}, Hit: {r['hit']:.1f}%)")


def find_nitou_jiku_strategy(df, pm):
    """ 2-Horse Axis """
    print("Preparing data for 2-Horse Axis Search...")
    df['rank_in_race'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    actual_map = {}
    for rid, grp in df[df['rank'].isin([1, 2, 3])].groupby('race_id'):
        actual_map[rid] = {1: grp[grp['rank']==1]['horse_number'].iloc[0] if not grp[grp['rank']==1].empty else -1, 2: grp[grp['rank']==2]['horse_number'].iloc[0] if not grp[grp['rank']==2].empty else -1, 3: grp[grp['rank']==3]['horse_number'].iloc[0] if not grp[grp['rank']==3].empty else -1}
        
    strategies = [{'label': '2Axis(Score1+2) Multi', 'mode': '12'}, {'label': '2Axis(Score1+EV) Multi', 'mode': '1e'}]
    results = []
    
    opp_count = 5; points = opp_count * 6; cost_per_race = points * 100
    
    for strat in strategies:
        total_cost = 0; total_ret = 0; hits = 0; count = 0
        for rid, grp in df.groupby('race_id'):
            sgrp = grp.sort_values('score', ascending=False)
            if len(sgrp) < 2: continue
            
            a1 = int(sgrp.iloc[0]['horse_number'])
            a2 = -1
            if strat['mode'] == '12':
                a2 = int(sgrp.iloc[1]['horse_number'])
            else:
                cands = sgrp.iloc[1:6]
                if cands.empty: continue
                good_cands = cands[cands['expected_value'] >= 1.0]
                if good_cands.empty: continue
                a2 = int(good_cands.loc[good_cands['expected_value'].idxmax()]['horse_number'])
                
            if a1 == a2: continue
            
            cands_opp = sgrp[~sgrp['horse_number'].isin([a1, a2])]
            sel_opps = set(cands_opp.iloc[:opp_count]['horse_number'].astype(int).tolist())
            
            acts = actual_map.get(rid, {1:-1, 2:-1, 3:-1})
            h1,h2,h3 = int(acts[1]), int(acts[2]), int(acts[3])
            
            payout = 0
            wins = {h1, h2, h3}
            if a1 in wins and a2 in wins:
                rem = list(wins - {a1, a2})[0]
                if rem in sel_opps:
                    k = f"{h1:02}{h2:02}{h3:02}"
                    payout = pm.get(rid, {}).get('sanrentan', {}).get(k, 0)
            
            if payout > 0: hits += 1
            total_ret += payout; total_cost += cost_per_race; count += 1
            
        roi = (total_ret/total_cost*100) if total_cost>0 else 0
        results.append({'label': strat['label'], 'roi': roi, 'bets': count, 'hit': hits/count*100})
        
    print("\n【2-Horse Axis Strategy Results】")
    for r in results: print(f"{r['label']} | ROI: {r['roi']:.1f}% (Bets: {r['bets']}, Hit: {r['hit']:.1f}%)")


def find_formation_strategy(df, pm):
    """
    3連単フォーメーション戦略 (Score #1 Axis) + Golden Rule Grid Search
    """
    print("Preparing data for Formation Search (Score #1 Axis) with Filters...")
    
    df_sorted = df.sort_values(['race_id', 'score'], ascending=[True, False])
    top_picks = df_sorted.drop_duplicates(subset=['race_id'], keep='first').copy()
    
    # Pre-calc rank
    df_sorted['ai_rank'] = df_sorted.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    # Candidate Map
    cand_map = {}
    for rid, grp in df_sorted[df_sorted['ai_rank'] <= 12].groupby('race_id'):
        cand_map[rid] = {int(r['ai_rank']): int(r['horse_number']) for _, r in grp.iterrows()}
        
    actual_map = {}
    for rid, grp in df[df['rank'].isin([1, 2, 3])].groupby('race_id'):
        actual_map[rid] = {
            1: grp[grp['rank']==1]['horse_number'].iloc[0] if not grp[grp['rank']==1].empty else -1,
            2: grp[grp['rank']==2]['horse_number'].iloc[0] if not grp[grp['rank']==2].empty else -1,
            3: grp[grp['rank']==3]['horse_number'].iloc[0] if not grp[grp['rank']==3].empty else -1
        }
        
    patterns = [
        ([2,3,4], [2,3,4,5,6,7], "Narrow 2nd(2-4) / Wide 3rd(2-7)"),
        ([2,3], [2,3,4,5,6],     "Super Narrow 2nd(2-3) / Mid 3rd(2-6)"),
        ([2,3,4,5], [2,3,4,5,6,7,8], "Mid 2nd(2-5) / Wide 3rd(2-8)"),
        # ([2,3,4], [2,3,4,5],     "Narrow 2nd(2-4) / Narrow 3rd(2-5)"),
        # ([2,3,4,5,6], [2,3,4,5,6,7], "Wide 2nd(2-6) / Wide 3rd(2-7)"),
    ]
    
    # Pre-calculate payout for each pattern for each top pick row
    # To be fast, add columns to top_picks for each pattern payout
    print("Pre-calculating pattern payouts...")
    for i, pat in enumerate(patterns):
        pname = f"pat_{i}"
        ranks2 = pat[0]; ranks3 = pat[1]
        
        pays = []
        for _, row in top_picks.iterrows():
            rid = row['race_id']
            cands = cand_map.get(rid, {})
            h1 = cands.get(1) # Axis (Score #1)
            
            # If h1 is invalid (shouldn't happen for #1), skip
            if h1 is None: 
                pays.append(0)
                continue
                
            acts = actual_map.get(rid, {1:-1, 2:-1, 3:-1})
            ah1, ah2, ah3 = int(acts[1]), int(acts[2]), int(acts[3])
            
            # Hit check logic
            # We only hit if ah1 == h1 AND ah2 in ranks2 AND ah3 in ranks3
            # But wait, ah2 must be the horse corresponding to one of ranks2.
            # We have cand_map. check if ah2 is in [cand_map[r] for r in ranks2]
            
            # Inverse lookup or just check generated combinations?
            # Generating combination string is safer.
            
            # Optimization: Only generate if ah1 == h1
            if ah1 != h1:
                pays.append(0)
                continue
            
            hit_combo = f"{ah1:02}{ah2:02}{ah3:02}"
            val = 0
            
            # Check if hit_combo is in pattern
            # ah2 must be mapped from some rank in ranks2
            # ah3 must be mapped from some rank in ranks3
            
            # Find rank of ah2
            # Slow to reverse map. 
            # Better: Generate valid combinations and check set membership.
            
            combos = set()
            for r2 in ranks2:
                h2 = cands.get(r2)
                if h2 is None: continue
                for r3 in ranks3:
                    if r2 == r3: continue
                    h3 = cands.get(r3)
                    if h3 is None: continue
                    combos.add(f"{h1:02}{h2:02}{h3:02}")
            
            if hit_combo in combos:
                val = pm.get(rid, {}).get('sanrentan', {}).get(hit_combo, 0)
            
            pays.append(val)
        
        top_picks[pname] = pays
        patterns[i] = patterns[i] + (pname,) # Add col name to tuple


    # Grid Search
    min_probs = [0.0, 0.20]
    min_evs = [0.0, 1.0, 1.2, 1.3]
    odds_filters = [
        {'label': 'All', 'min': 0, 'max': 9999},
        {'label': 'Solid(<3.0)', 'min': 0, 'max': 3.0},
        {'label': 'High(10.0+)', 'min': 10.0, 'max': 9999},
    ]

    results = []
    print("Running Grid Search on Formations...")
    
    for pat in patterns:
        ranks2 = pat[0]; ranks3 = pat[1]; label = pat[2]; col = pat[3]
        
        # Num points calculation (constant per pattern approx, but depends on available horses, assumes all rank horses exist)
        # Exact points: sum of valid combinations (h2!=h3 etc).
        # Helper to calc theoretical max points
        cnt = 0
        for r2 in ranks2:
            for r3 in ranks3:
                if r2 != r3: cnt += 1
        bets_per_race = cnt
        cost_unit = bets_per_race * 100
        
        for mp in min_probs:
            for me in min_evs:
                for of in odds_filters:
                    mask = (top_picks['prob']>=mp) & (top_picks['expected_value']>=me) & (top_picks['odds']>=of['min']) & (top_picks['odds']<of['max'])
                    sub = top_picks[mask]
                    count = len(sub)
                    if count < 20: continue
                    
                    ret = sub[col].sum()
                    cost = count * cost_unit
                    
                    roi = (ret/cost*100) if cost>0 else 0
                    
                    if roi > 100:
                         results.append({
                             'label': label,
                             'cond': f"P>={mp}, E>={me}, O={of['label']}",
                             'roi': roi, 'bets': count, 'hit_rate': (sub[col]>0).mean()*100, 
                             'profit': ret-cost
                         })

    results.sort(key=lambda x: x['roi'], reverse=True)
    print("\n【Formation Strategy + Golden Rule (ROI > 100%)】")
    for r in results[:10]:
        print(f"{r['label']} | {r['cond']} | ROI: {r['roi']:.1f}% (Bets: {r['bets']}, Hit: {r['hit_rate']:.1f}%, Profit: {r['profit']:,})")



def grid_search_strategies(df, pm):
    bet_types = ['sanrentan', 'wide', 'umaren', 'tansho']
    configs_to_test = [
        {'low': {'bet_type': 'sanrentan', 'opp_count': 7, 'threshold': 3.0}, 'mid': {'bet_type': 'sanrentan', 'opp_count': 6, 'threshold': 10.0}, 'high': {'bet_type': 'wide', 'opp_count': 7}},
        {'low': {'bet_type': 'tansho', 'opp_count': 1, 'threshold': 3.0}, 'mid': {'bet_type': 'tansho', 'opp_count': 1, 'threshold': 10.0}, 'high': {'bet_type': 'tansho', 'opp_count': 1}},
    ]
    print("="*70); print("オッズ帯別戦略 グリッドサーチ (簡易版)"); print("="*70)
    for i, cfg in enumerate(configs_to_test, 1):
        result = simulate_odds_based_strategy(df, pm, cfg)
        print(f"Config {i} TOTAL ROI: {result['total_roi']:.2f}%")


if __name__ == "__main__":
    MODELS = ['catboost_v7']
    for model_name in MODELS:
        print("\n\n" + "#"*70)
        print(f"[START] Processing: {model_name}")
        print("#"*70)
        
        try:
            df, pm = load_data(model_name)
            df = df.sort_values(['race_id', 'score'], ascending=[True, False])
            print(f"Loaded: {len(df)} rows, {df['race_id'].nunique()} races")
            
            # Run all
            find_golden_rules(df, pm)
            find_formation_strategy(df, pm)
            find_ev_based_axis_strategy(df, pm)
            find_nitou_jiku_strategy(df, pm)
            
        except FileNotFoundError:
            print(f"Predictions for {model_name} not found. Skipping.")
        except Exception as e:
            import traceback
            print(f"Error processing {model_name}: {e}")
            traceback.print_exc()
        
        print(f"[END] Processing: {model_name}")

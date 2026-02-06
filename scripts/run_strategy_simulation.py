"""
Zero-Based Betting Strategy Simulation
- Single bets (Win/Place): 1 race = 1 bet (Top1 only)
- Combination bets: Multi-point allowed (BOX, etc.)

Usage: python scripts/run_strategy_simulation.py
"""
import os
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_db_engine():
    user = os.getenv("POSTGRES_USER", "postgres")
    pw = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "host.docker.internal")
    port = os.getenv("POSTGRES_PORT", "5433")
    db = os.getenv("POSTGRES_DB", "pckeiba")
    return create_engine(f'postgresql://{user}:{pw}@{host}:{port}/{db}')

def load_payout_data(engine, years):
    year_list = ','.join([f"'{y}'" for y in years])
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_tansho_1a as win_1_horse, haraimodoshi_tansho_1b as win_1_pay,
        haraimodoshi_fukusho_1a as place_1_horse, haraimodoshi_fukusho_1b as place_1_pay,
        haraimodoshi_fukusho_2a as place_2_horse, haraimodoshi_fukusho_2b as place_2_pay,
        haraimodoshi_fukusho_3a as place_3_horse, haraimodoshi_fukusho_3b as place_3_pay,
        haraimodoshi_wide_1a as wide_1_horses, haraimodoshi_wide_1b as wide_1_pay,
        haraimodoshi_wide_2a as wide_2_horses, haraimodoshi_wide_2b as wide_2_pay,
        haraimodoshi_wide_3a as wide_3_horses, haraimodoshi_wide_3b as wide_3_pay,
        haraimodoshi_umaren_1a as quinella_horses, haraimodoshi_umaren_1b as quinella_pay,
        haraimodoshi_umatan_1a as exacta_1st, haraimodoshi_umatan_1b as exacta_2nd, haraimodoshi_umatan_1c as exacta_pay
    FROM jvd_hr 
    WHERE kaisai_nen IN ({year_list})
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df['race_id'].astype(str)
    return df

def parse_payouts(df_pay):
    payout_dict = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        pay = {'win': {}, 'place': {}, 'wide': {}, 'quinella': {}, 'exacta': {}}
        try:
            h = int(row['win_1_horse'])
            p = int(row['win_1_pay'])
            if p > 0: pay['win'][h] = p
        except: pass
        for i in [1, 2, 3]:
            try:
                h = int(row[f'place_{i}_horse'])
                p = int(row[f'place_{i}_pay'])
                if p > 0: pay['place'][h] = p
            except: pass
        for i in [1, 2, 3]:
            try:
                horses_str = str(row[f'wide_{i}_horses']).zfill(4)
                h1, h2 = int(horses_str[:2]), int(horses_str[2:])
                p = int(row[f'wide_{i}_pay'])
                if p > 0: pay['wide'][(min(h1,h2), max(h1,h2))] = p
            except: pass
        try:
            horses_str = str(row['quinella_horses']).zfill(4)
            h1, h2 = int(horses_str[:2]), int(horses_str[2:])
            p = int(row['quinella_pay'])
            if p > 0: pay['quinella'][(min(h1,h2), max(h1,h2))] = p
        except: pass
        try:
            # Exacta (馬単): haraimodoshi_umatan_1a = '1405' (14番→05番), 1b = payout
            horses_str = str(row['exacta_1st']).zfill(4)
            h1 = int(horses_str[:2])  # 1位
            h2 = int(horses_str[2:])  # 2位
            p = int(row['exacta_2nd'])  # これが実は払戻金
            if p > 0: pay['exacta'][(h1, h2)] = p
        except: pass
        payout_dict[rid] = pay
    return payout_dict

# ========== STRATEGY FUNCTIONS ==========

def strategy_win_confident(df, payout_dict, prob_th=0.25, gap_th=0.10, min_odds=2.0, max_odds=15, ev_th=1.5):
    """
    Strategy A: 高確信度単勝 (Win-Confident)
    - 1R1ベット: Top1予測馬のみ
    - 条件: 高確率 + 大きな分離度 + EV条件
    """
    cost, returns, hits, bets, races = 0, 0, 0, 0, 0
    
    for rid, group in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        group = group.sort_values('pred_prob', ascending=False)
        
        if len(group) < 2: continue
        
        top1 = group.iloc[0]
        top2 = group.iloc[1]
        
        prob1 = top1['pred_prob']
        prob2 = top2['pred_prob']
        gap = prob1 - prob2  # 分離度
        odds = top1['odds_final']
        ev = prob1 * odds
        horse = top1['horse_number']
        
        # Entry conditions
        if prob1 >= prob_th and gap >= gap_th and min_odds <= odds <= max_odds and ev >= ev_th:
            cost += 100
            bets += 1
            races += 1
            if horse in pay['win']:
                returns += pay['win'][horse]
                hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    hit_rate = (hits / bets * 100) if bets > 0 else 0
    return {'strategy': 'Win-Confident', 'bets': bets, 'races': races, 'hits': hits, 'hit_rate': hit_rate, 'roi': roi,
            'params': f"prob>={prob_th*100:.0f}%, gap>={gap_th*100:.0f}%, odds[{min_odds}-{max_odds}], EV>={ev_th}"}

def strategy_dynamic_ev_max(df, payout_dict, spread_th=0.10, min_prob=0.10, ev_th=1.5, min_odds=2.0, max_odds=30):
    """
    Strategy F: 動的Top-N + EV最大 (Dynamic-EV-Max)
    - 確率分布に基づいてNを動的に決定
    - Top N の中でEVが最大の馬にベット
    - spread_th: Top1-Top2の差がこれ以下なら広くNを取る
    """
    cost, returns, hits, bets, races = 0, 0, 0, 0, 0
    
    for rid, group in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        group = group.sort_values('pred_prob', ascending=False).reset_index(drop=True)
        
        if len(group) < 2: continue
        
        # 確率分布の分析
        top1_prob = group.iloc[0]['pred_prob']
        top2_prob = group.iloc[1]['pred_prob']
        spread = top1_prob - top2_prob  # 分離度
        
        # 動的N決定: 分離度が小さい（混戦）ほどNを広く
        if spread <= 0.03:      # 非常に混戦
            N = 5
        elif spread <= spread_th:  # 混戦
            N = 4
        elif spread <= 0.15:    # やや本命
            N = 3
        else:                   # 本命明確
            N = 2
        
        # Top N の候補馬
        candidates = group.head(N)
        candidates = candidates[candidates['pred_prob'] >= min_prob]
        
        if len(candidates) == 0: continue
        
        # EVを計算して最大を選択
        candidates = candidates.copy()
        candidates['ev'] = candidates['pred_prob'] * candidates['odds_final']
        
        # EV条件とオッズ条件でフィルタ
        valid = candidates[(candidates['ev'] >= ev_th) & 
                          (candidates['odds_final'] >= min_odds) & 
                          (candidates['odds_final'] <= max_odds)]
        
        if len(valid) == 0: continue
        
        # EV最大の馬を選択
        best = valid.loc[valid['ev'].idxmax()]
        
        cost += 100
        bets += 1
        races += 1
        
        horse = int(best['horse_number'])
        if horse in pay['win']:
            returns += pay['win'][horse]
            hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    hit_rate = (hits / bets * 100) if bets > 0 else 0
    return {'strategy': 'Dynamic-EV-Max', 'bets': bets, 'races': races, 'hits': hits, 'hit_rate': hit_rate, 'roi': roi,
            'params': f"spread<={spread_th*100:.0f}%, minProb>={min_prob*100:.0f}%, EV>={ev_th}, odds[{min_odds}-{max_odds}]"}

def strategy_place_stable(df, payout_dict, prob_th=0.30, max_field=16, min_odds=1.3, max_odds=3.0):
    """
    Strategy B: 安定複勝 (Place-Stable)
    - 1R1ベット: Top1予測馬のみ
    - 条件: 高確率 + 安定レース（少頭数）
    """
    cost, returns, hits, bets, races = 0, 0, 0, 0, 0
    
    for rid, group in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        group = group.sort_values('pred_prob', ascending=False)
        
        field_size = len(group)
        if field_size > max_field: continue
        
        top1 = group.iloc[0]
        prob1 = top1['pred_prob']
        odds = top1['odds_final']
        horse = top1['horse_number']
        
        # Estimate place odds (approx 1/3 of win odds)
        place_odds_est = max(1.1, odds / 3)
        
        # Entry conditions
        if prob1 >= prob_th and min_odds <= place_odds_est <= max_odds:
            cost += 100
            bets += 1
            races += 1
            if horse in pay['place']:
                returns += pay['place'][horse]
                hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    hit_rate = (hits / bets * 100) if bets > 0 else 0
    return {'strategy': 'Place-Stable', 'bets': bets, 'races': races, 'hits': hits, 'hit_rate': hit_rate, 'roi': roi,
            'params': f"prob>={prob_th*100:.0f}%, field<={max_field}, odds[{min_odds}-{max_odds}]"}

def strategy_quinella_focus(df, payout_dict, prob1_th=0.20, prob2_th=0.15, sum_th=0.40):
    """
    Strategy C: 本命馬連 (Quinella-Focus)
    - 1点買い: Top1-Top2の馬連
    - 条件: Top1 + Top2 確率合計が高い
    """
    cost, returns, hits, bets, races = 0, 0, 0, 0, 0
    
    for rid, group in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        group = group.sort_values('pred_prob', ascending=False)
        
        if len(group) < 2: continue
        
        top1 = group.iloc[0]
        top2 = group.iloc[1]
        
        prob1 = top1['pred_prob']
        prob2 = top2['pred_prob']
        prob_sum = prob1 + prob2
        
        h1 = top1['horse_number']
        h2 = top2['horse_number']
        
        # Entry conditions
        if prob1 >= prob1_th and prob2 >= prob2_th and prob_sum >= sum_th:
            cost += 100
            bets += 1
            races += 1
            key = (min(h1, h2), max(h1, h2))
            if key in pay['quinella']:
                returns += pay['quinella'][key]
                hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    hit_rate = (hits / bets * 100) if bets > 0 else 0
    return {'strategy': 'Quinella-Focus', 'bets': bets, 'races': races, 'hits': hits, 'hit_rate': hit_rate, 'roi': roi,
            'params': f"prob1>={prob1_th*100:.0f}%, prob2>={prob2_th*100:.0f}%, sum>={sum_th*100:.0f}%"}

def strategy_wide_box(df, payout_dict, gap_max=0.05, prob3_th=0.15, min_field=12):
    """
    Strategy D: 混戦ワイドBOX (Wide-Chaos)
    - 3点買い: Top1-Top2-Top3のワイドBOX
    - 条件: 混戦レース（分離度が小さい）+ 大頭数
    """
    cost, returns, hits, bets, races = 0, 0, 0, 0, 0
    
    for rid, group in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        group = group.sort_values('pred_prob', ascending=False)
        
        field_size = len(group)
        if field_size < min_field: continue
        if len(group) < 3: continue
        
        top1 = group.iloc[0]
        top2 = group.iloc[1]
        top3 = group.iloc[2]
        
        gap = top1['pred_prob'] - top2['pred_prob']
        prob3 = top3['pred_prob']
        
        h1 = top1['horse_number']
        h2 = top2['horse_number']
        h3 = top3['horse_number']
        
        # Entry conditions: 混戦 + Top3も有力
        if gap <= gap_max and prob3 >= prob3_th:
            races += 1
            # 3 wide bets (BOX)
            for pair in combinations([h1, h2, h3], 2):
                cost += 100
                bets += 1
                key = (min(pair), max(pair))
                if key in pay['wide']:
                    returns += pay['wide'][key]
                    hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    hit_rate = (hits / bets * 100) if bets > 0 else 0
    return {'strategy': 'Wide-BOX', 'bets': bets, 'races': races, 'hits': hits, 'hit_rate': hit_rate, 'roi': roi,
            'params': f"gap<={gap_max*100:.0f}%, prob3>={prob3_th*100:.0f}%, field>={min_field}"}

def strategy_exacta_flow(df, payout_dict, prob1_th=0.25, gap_th=0.08):
    """
    Strategy E: 馬単流し (Exacta-Flow)
    - 2点買い: Top1→(Top2,Top3)の馬単流し
    - 条件: Top1が確信度高い
    """
    cost, returns, hits, bets, races = 0, 0, 0, 0, 0
    
    for rid, group in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        group = group.sort_values('pred_prob', ascending=False)
        
        if len(group) < 3: continue
        
        top1 = group.iloc[0]
        top2 = group.iloc[1]
        top3 = group.iloc[2]
        
        prob1 = top1['pred_prob']
        gap = prob1 - top2['pred_prob']
        
        h1 = top1['horse_number']
        h2 = top2['horse_number']
        h3 = top3['horse_number']
        
        # Entry conditions
        if prob1 >= prob1_th and gap >= gap_th:
            races += 1
            # Top1 → Top2, Top1 → Top3
            for h2nd in [h2, h3]:
                cost += 100
                bets += 1
                key = (h1, h2nd)
                if key in pay['exacta']:
                    returns += pay['exacta'][key]
                    hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    hit_rate = (hits / bets * 100) if bets > 0 else 0
    return {'strategy': 'Exacta-Flow', 'bets': bets, 'races': races, 'hits': hits, 'hit_rate': hit_rate, 'roi': roi,
            'params': f"prob1>={prob1_th*100:.0f}%, gap>={gap_th*100:.0f}%"}

def main():
    logger.info("🎯 Zero-Based Betting Strategy Simulation (All Models)")
    
    pred_dir = "models/experiments/exp_t2_refined_v3/cv_results"
    
    # Load all model predictions
    df_binary = pd.read_parquet(f"{pred_dir}/preds_binary.parquet")
    df_top2 = pd.read_parquet(f"{pred_dir}/preds_top2.parquet")
    df_top3 = pd.read_parquet(f"{pred_dir}/preds_top3.parquet")
    
    for df in [df_binary, df_top2, df_top3]:
        df['race_id'] = df['race_id'].astype(str)
        df['date'] = pd.to_datetime(df['date'])
        df['horse_number'] = df['horse_number'].astype(int)
    
    # Create Blend: 0.4*Binary + 0.6*Top3
    df_blend = df_binary[['race_id', 'horse_number', 'date', 'rank', 'odds_final']].copy()
    df_blend['pred_prob'] = 0.4 * df_binary['pred_prob'].values + 0.6 * df_top3['pred_prob'].values
    
    models = {
        'Binary': df_binary[df_binary['date'].dt.year == 2023],
        'Top2': df_top2[df_top2['date'].dt.year == 2023],
        'Top3': df_top3[df_top3['date'].dt.year == 2023],
        'Blend': df_blend[df_blend['date'].dt.year == 2023]
    }
    
    logger.info(f"Loaded 4 models with predictions")
    
    # Load payouts
    engine = get_db_engine()
    df_pay = load_payout_data(engine, [2023])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Loaded {len(payout_dict)} payout records\n")
    
    all_results = []
    
    # Best parameter configs for each strategy
    configs = [
        # Win-Confident
        ('Win-Confident', strategy_win_confident, {'prob_th': 0.28, 'gap_th': 0.12, 'ev_th': 1.8}),
        ('Win-Confident', strategy_win_confident, {'prob_th': 0.30, 'gap_th': 0.12, 'ev_th': 1.5}),
        # Place-Stable
        ('Place-Stable', strategy_place_stable, {'prob_th': 0.30, 'max_field': 16}),
        ('Place-Stable', strategy_place_stable, {'prob_th': 0.32, 'max_field': 14}),
        # Dynamic-EV-Max (NEW)
        ('Dynamic-EV-Max', strategy_dynamic_ev_max, {'spread_th': 0.10, 'min_prob': 0.10, 'ev_th': 1.5}),
        ('Dynamic-EV-Max', strategy_dynamic_ev_max, {'spread_th': 0.10, 'min_prob': 0.12, 'ev_th': 1.8}),
        ('Dynamic-EV-Max', strategy_dynamic_ev_max, {'spread_th': 0.10, 'min_prob': 0.15, 'ev_th': 2.0}),
        ('Dynamic-EV-Max', strategy_dynamic_ev_max, {'spread_th': 0.08, 'min_prob': 0.10, 'ev_th': 1.5, 'max_odds': 20}),
        ('Dynamic-EV-Max', strategy_dynamic_ev_max, {'spread_th': 0.08, 'min_prob': 0.12, 'ev_th': 2.0, 'max_odds': 25}),
        # Quinella-Focus
        ('Quinella-Focus', strategy_quinella_focus, {'prob1_th': 0.25, 'prob2_th': 0.18, 'sum_th': 0.45}),
        # Exacta-Flow
        ('Exacta-Flow', strategy_exacta_flow, {'prob1_th': 0.28, 'gap_th': 0.10}),
        ('Exacta-Flow', strategy_exacta_flow, {'prob1_th': 0.30, 'gap_th': 0.12}),
    ]
    
    for model_name, df in models.items():
        for name, func, kwargs in configs:
            result = func(df, payout_dict, **kwargs)
            result['model'] = model_name
            all_results.append(result)
    
    # Print results sorted by ROI
    print("="*120)
    print(" Strategy Simulation Results - All Models (2023 Validation)")
    print("="*120)
    print(f"{'Model':<10} {'Strategy':<18} {'Params':<45} {'Races':>5} {'Bets':>5} {'Hits':>4} {'Hit%':>7} {'ROI':>8}")
    print("-"*120)
    
    for r in sorted(all_results, key=lambda x: -x['roi'])[:25]:  # Top 25
        print(f"{r['model']:<10} {r['strategy']:<18} {r['params']:<45} {r['races']:>5} {r['bets']:>5} {r['hits']:>4} {r['hit_rate']:>6.1f}% {r['roi']:>7.1f}%")
    
    print("="*120)
    
    # Highlight best overall
    best = max(all_results, key=lambda x: x['roi'])
    print(f"\n🏆 Best: {best['model']} + {best['strategy']} → ROI={best['roi']:.1f}%")
    print(f"   Params: {best['params']}")
    print(f"   Races: {best['races']}, Bets: {best['bets']}, Hits: {best['hits']}, Hit%: {best['hit_rate']:.1f}%")

if __name__ == "__main__":
    main()

"""
最適統合戦略シミュレーション
全グリッドサーチ結果を組み合わせた最適戦略の総合ROIを計算
"""
import os
import sys
import pandas as pd
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict
import logging
from scipy.special import softmax
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_data(years=[2024, 2025]):
    data_path = 'data/processed/preprocessed_data.parquet'
    df = pd.read_parquet(data_path)
    df = df[df['year'].isin(years)].copy()
    return df

def load_model_and_predict(df):
    sys.path.append('src')
    from model.ensemble import EnsembleModel
    
    model = EnsembleModel()
    model.load_model('models/ensemble_v4_2025.pkl')
    
    import pickle
    with open('data/processed/lgbm_datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    
    X = df[feature_cols]
    scores = model.predict(X)
    df['score'] = scores
    return df

def load_payouts(years=[2024, 2025]):
    engine = get_db_engine()
    years_str = ",".join([f"'{y}'" for y in years])
    query = text(f"SELECT * FROM jvd_hr WHERE kaisai_nen IN ({years_str})")
    df = pd.read_sql(query, engine)
    
    df['race_id'] = (
        df['kaisai_nen'].astype(str) +
        df['keibajo_code'].astype(str) +
        df['kaisai_kai'].astype(str) +
        df['kaisai_nichime'].astype(str) +
        df['race_bango'].astype(str)
    )
    return df

def build_payout_map(pay_df):
    payout_map = defaultdict(lambda: {'tansho': {}, 'umaren': {}, 'wide': {}, 'sanrenpuku': {}, 'sanrentan': {}})
    
    for _, row in pay_df.iterrows():
        rid = row['race_id']
        
        for prefix, max_count in [('haraimodoshi_tansho', 3), ('haraimodoshi_umaren', 3),
                                   ('haraimodoshi_wide', 7),
                                   ('haraimodoshi_sanrenpuku', 3), ('haraimodoshi_sanrentan', 6)]:
            bet_type = prefix.split('_')[1]
            for i in range(1, max_count + 1):
                col_a = f'{prefix}_{i}a'
                col_b = f'{prefix}_{i}b'
                if col_a in row and row[col_a] and str(row[col_a]).strip():
                    try:
                        key = str(row[col_a]).strip()
                        val = int(float(str(row[col_b]).strip()))
                        payout_map[rid][bet_type][key] = val
                    except:
                        pass
    
    return dict(payout_map)

def get_race_data(df):
    """レースごとのデータを取得"""
    race_data = {}
    
    for rid, grp in df.groupby('race_id'):
        sorted_g = grp.sort_values('score', ascending=False)
        if len(sorted_g) < 6:
            continue
        
        top1 = sorted_g.iloc[0]
        scores = sorted_g.head(6)['score'].values
        
        race_data[rid] = {
            'horses': sorted_g['horse_number'].astype(int).tolist(),
            'top1_popularity': int(top1['popularity']) if not pd.isna(top1['popularity']) else 99,
            'top1_odds': top1['odds'] if not pd.isna(top1['odds']) else 0,
            'top1_rank': top1['rank'],
            'score_range': scores[0] - scores[5],
            'top3_gap': scores[0] - scores[2],
            'bottom_gap': scores[2] - scores[5],
        }
    
    return race_data

def run_optimal_strategy(race_data, payout_map):
    """
    最適統合戦略
    
    優先順位:
    1. Top1が7番人気以上 → 3連単1頭軸6点 (ROI 232%)
    2. Top1が4-6番人気 → 3連単1頭軸20点 (ROI 124%)
    3. Top3優勢・下位団子 → 3連複軸ながし (ROI 119%)
    4. スコア均衡(gap<0.3) → 3連単2頭軸12点 (ROI 100%)
    5. その他 → 見送り
    """
    
    print("\n" + "="*80)
    print("📊 最適統合戦略シミュレーション")
    print("="*80)
    
    total_races = len(race_data)
    
    # 戦略別統計
    strategies = {
        'sanrentan_pop7+_6': {'name': '3連単1頭軸6点(7番人気以上)', 'races': 0, 'cost': 0, 'return': 0, 'hits': 0},
        'sanrentan_pop4-6_20': {'name': '3連単1頭軸20点(4-6番人気)', 'races': 0, 'cost': 0, 'return': 0, 'hits': 0},
        'sanrenpuku_top3dom': {'name': '3連複軸ながし(Top3優勢)', 'races': 0, 'cost': 0, 'return': 0, 'hits': 0},
        'sanrentan_gap_12': {'name': '3連単2頭軸12点(均衡)', 'races': 0, 'cost': 0, 'return': 0, 'hits': 0},
        'skip': {'name': '見送り', 'races': 0},
    }
    
    for rid, rd in race_data.items():
        if rid not in payout_map:
            strategies['skip']['races'] += 1
            continue
        
        h = rd['horses']
        pop = rd['top1_popularity']
        score_range = rd['score_range']
        top3_gap = rd['top3_gap']
        bottom_gap = rd['bottom_gap']
        
        strategy = None
        cost = 0
        ret = 0
        hit = 0
        
        # 条件1: Top1が7番人気以上 → 3連単1頭軸6点
        if pop >= 7:
            strategy = 'sanrentan_pop7+_6'
            if len(h) >= 4:
                axis = h[0]
                opps = h[1:4]
                tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
                cost = len(tickets) * 100
                for t in tickets:
                    key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                    if key in payout_map[rid]['sanrentan']:
                        ret += payout_map[rid]['sanrentan'][key]
                        hit = 1
        
        # 条件2: Top1が4-6番人気 → 3連単1頭軸20点
        elif 4 <= pop <= 6:
            strategy = 'sanrentan_pop4-6_20'
            if len(h) >= 6:
                axis = h[0]
                opps = h[1:6]
                tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
                cost = len(tickets) * 100
                for t in tickets:
                    key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                    if key in payout_map[rid]['sanrentan']:
                        ret += payout_map[rid]['sanrentan'][key]
                        hit = 1
        
        # 条件3: Top3優勢・下位団子 → 3連複軸ながし
        elif top3_gap >= 0.3 and bottom_gap < 0.1:
            strategy = 'sanrenpuku_top3dom'
            if len(h) >= 6:
                axis = h[0]
                opps = h[1:6]
                tickets = list(combinations([axis] + opps[:5], 3))
                tickets = [t for t in tickets if axis in t]
                cost = len(tickets) * 100
                for t in tickets:
                    c_sorted = sorted(t)
                    key = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
                    if key in payout_map[rid]['sanrenpuku']:
                        ret += payout_map[rid]['sanrenpuku'][key]
                        hit = 1
        
        # 条件4: スコア均衡(gap<0.3) → 見送り（ROI 96%でマイナスのため）
        elif score_range < 0.3:
            strategy = 'skip'  # 均衡レースは見送り
        
        # 見送り
        else:
            strategy = 'skip'
        
        if strategy and strategy != 'skip':
            strategies[strategy]['races'] += 1
            strategies[strategy]['cost'] += cost
            strategies[strategy]['return'] += ret
            strategies[strategy]['hits'] += hit
        else:
            strategies['skip']['races'] += 1
    
    # 結果表示
    print(f"\n【全 {total_races} レース中の内訳】\n")
    
    bet_races = 0
    total_cost = 0
    total_return = 0
    total_hits = 0
    
    for key, s in strategies.items():
        if key == 'skip':
            continue
        
        if s['races'] > 0:
            roi = s['return'] / s['cost'] * 100 if s['cost'] > 0 else 0
            hit_rate = s['hits'] / s['races'] * 100 if s['races'] > 0 else 0
            profit = s['return'] - s['cost']
            avg_cost = s['cost'] / s['races'] if s['races'] > 0 else 0
            
            print(f"📍 {s['name']}:")
            print(f"   対象レース: {s['races']} 件")
            print(f"   平均点数: {avg_cost/100:.0f}点")
            print(f"   投資額: ¥{s['cost']:,}")
            print(f"   払戻: ¥{s['return']:,.0f}")
            print(f"   利益: ¥{profit:+,.0f}")
            print(f"   ROI: {roi:.1f}%, 的中率: {hit_rate:.1f}%")
            print()
            
            bet_races += s['races']
            total_cost += s['cost']
            total_return += s['return']
            total_hits += s['hits']
    
    skip_races = strategies['skip']['races']
    bet_rate = bet_races / total_races * 100 if total_races > 0 else 0
    
    print("-" * 60)
    print(f"\n🎯 【総合結果】")
    print(f"   全レース数: {total_races}")
    print(f"   ベット対象: {bet_races} レース ({bet_rate:.1f}%)")
    print(f"   見送り: {skip_races} レース ({100-bet_rate:.1f}%)")
    print()
    print(f"   💰 総投資額: ¥{total_cost:,}")
    print(f"   💰 総払戻: ¥{total_return:,.0f}")
    print(f"   💰 総利益: ¥{total_return - total_cost:+,.0f}")
    print()
    
    if total_cost > 0:
        combined_roi = total_return / total_cost * 100
        combined_hit_rate = total_hits / bet_races * 100 if bet_races > 0 else 0
        print(f"   📈 総合ROI: {combined_roi:.1f}%")
        print(f"   📈 総合的中率: {combined_hit_rate:.1f}%")
        print(f"   📈 ベット率: {bet_rate:.1f}%")
        
        # 週あたりの目安
        weeks = 104  # 約2年分
        bets_per_week = bet_races / weeks
        profit_per_week = (total_return - total_cost) / weeks
        print(f"\n   📅 週あたりベット数: {bets_per_week:.1f} レース")
        print(f"   📅 週あたり利益: ¥{profit_per_week:+,.0f}")
        
        # ファイル保存
        with open('reports/optimal_strategy_result.txt', 'w', encoding='utf-8') as f:
            f.write("=== 最適統合戦略シミュレーション結果 (2024+2025年) ===\n\n")
            f.write(f"全レース数: {total_races}\n")
            f.write(f"ベット対象: {bet_races} レース ({bet_rate:.1f}%)\n")
            f.write(f"見送り: {skip_races} レース\n\n")
            f.write(f"総投資額: ¥{total_cost:,}\n")
            f.write(f"総払戻: ¥{total_return:,.0f}\n")
            f.write(f"総利益: ¥{total_return - total_cost:+,.0f}\n\n")
            f.write(f"総合ROI: {combined_roi:.1f}%\n")
            f.write(f"総合的中率: {combined_hit_rate:.1f}%\n")
            f.write(f"ベット率: {bet_rate:.1f}%\n\n")
            
            f.write("--- 戦略別内訳 ---\n")
            for key, s in strategies.items():
                if key != 'skip' and s['races'] > 0:
                    roi = s['return'] / s['cost'] * 100 if s['cost'] > 0 else 0
                    hit_rate = s['hits'] / s['races'] * 100
                    f.write(f"{s['name']}: {s['races']}レース, ROI {roi:.1f}%, Hit {hit_rate:.1f}%\n")
        
        print("\n結果を reports/optimal_strategy_result.txt に保存しました")
    
    print("\n" + "="*80)

def main():
    print("\n" + "#"*80)
    print("# 📊 最適統合戦略 シミュレーション (2024+2025年)")
    print("# 全グリッドサーチ結果を組み合わせた最適戦略")
    print("#"*80)
    
    years = [2024, 2025]
    
    logger.info(f"Loading data for years: {years}")
    df = load_data(years)
    df = load_model_and_predict(df)
    
    pay_df = load_payouts(years)
    payout_map = build_payout_map(pay_df)
    
    race_data = get_race_data(df)
    logger.info(f"Prepared data for {len(race_data)} races")
    
    run_optimal_strategy(race_data, payout_map)
    
    print("\n✅ シミュレーション完了!")

if __name__ == "__main__":
    main()

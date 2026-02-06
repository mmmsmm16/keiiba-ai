"""
馬券種類×条件別グリッドサーチ
- 単勝 / 馬連 / 3連複 / 3連単
- x 人気別 / スコア差別 / 競馬場別
"""
import os
import sys
import pandas as pd
import numpy as np
from itertools import combinations, permutations, product
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

def load_predictions_from_db(years=[2024, 2025]):
    data_path = 'data/processed/preprocessed_data.parquet'
    df = pd.read_parquet(data_path)
    df = df[df['year'].isin(years)].copy()
    logger.info(f"Loaded {len(df)} rows for years {years}")
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

def preprocess_data(df):
    df = df.copy()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    df['prob'] = df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    df['ev'] = df['prob'] * df['odds'].fillna(0)
    
    df['score_max'] = df.groupby('race_id')['score'].transform('max')
    df['score_second'] = df.groupby('race_id')['score'].transform(lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else x.max())
    df['score_gap'] = df['score_max'] - df['score_second']
    
    return df

def get_race_data(df):
    """レースごとのデータを取得"""
    race_data = {}
    
    for rid, grp in df.groupby('race_id'):
        sorted_g = grp.sort_values('score', ascending=False)
        if len(sorted_g) < 3:
            continue
        
        top1 = sorted_g.iloc[0]
        
        venue = str(rid)[4:6] if len(str(rid)) >= 6 else ''
        
        pop = top1.get('popularity', 99)
        if pd.isna(pop): pop = 99
        
        score_gap = top1.get('score_gap', 0)
        if pd.isna(score_gap): score_gap = 0
        
        race_data[rid] = {
            'venue': venue,
            'top1_popularity': int(pop),
            'score_gap': score_gap,
            'top1_odds': top1['odds'] if not pd.isna(top1['odds']) else 0,
            'top1_rank': top1['rank'],
            'horses': sorted_g['horse_number'].astype(int).tolist()
        }
    
    return race_data

def simulate_bet(race_data, payout_map, rid, bet_type, opp_count=5):
    """特定の馬券を賭けた場合のコストとリターンを計算"""
    rd = race_data[rid]
    h_nums = rd['horses']
    
    if bet_type == 'tansho':
        cost = 100
        axis = h_nums[0]
        key = f"{axis:02}"
        ret = payout_map[rid]['tansho'].get(key, 0)
        hit = 1 if ret > 0 else 0
        return cost, ret, hit
    
    elif bet_type == 'umaren':
        if len(h_nums) < opp_count + 1:
            return 0, 0, 0
        axis = h_nums[0]
        opps = h_nums[1:opp_count+1]
        cost = len(opps) * 100
        ret = 0
        hit = 0
        for opp in opps:
            c_sorted = sorted([axis, opp])
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in payout_map[rid]['umaren']:
                ret += payout_map[rid]['umaren'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'wide':
        if len(h_nums) < opp_count + 1:
            return 0, 0, 0
        axis = h_nums[0]
        opps = h_nums[1:opp_count+1]
        cost = len(opps) * 100
        ret = 0
        hit = 0
        for opp in opps:
            c_sorted = sorted([axis, opp])
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in payout_map[rid].get('wide', {}):
                ret += payout_map[rid]['wide'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'sanrenpuku':
        if len(h_nums) < 6:
            return 0, 0, 0
        axis = h_nums[0]
        opps = h_nums[1:6]
        tickets = list(combinations([axis] + opps[:5], 3))
        tickets = [t for t in tickets if axis in t]  # 軸固定
        cost = len(tickets) * 100
        ret = 0
        hit = 0
        for t in tickets:
            c_sorted = sorted(t)
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
            if key in payout_map[rid]['sanrenpuku']:
                ret += payout_map[rid]['sanrenpuku'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'sanrentan':
        if len(h_nums) < opp_count + 1:
            return 0, 0, 0
        axis = h_nums[0]
        opps = h_nums[1:opp_count+1]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        cost = len(tickets) * 100
        ret = 0
        hit = 0
        for t in tickets:
            key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
            if key in payout_map[rid]['sanrentan']:
                ret += payout_map[rid]['sanrentan'][key]
                hit = 1
        return cost, ret, hit
    
    return 0, 0, 0

def run_bet_type_grid_search(race_data, payout_map):
    """馬券種類×条件のグリッドサーチ"""
    
    bet_types = ['tansho', 'umaren', 'wide', 'sanrenpuku', 'sanrentan']
    
    # 人気カテゴリ
    def pop_cat(pop):
        if pop == 1: return '1番人気'
        elif pop <= 3: return '2-3番人気'
        elif pop <= 6: return '4-6番人気'
        elif pop <= 10: return '7-10番人気'
        else: return '11+番人気'
    
    # スコア差カテゴリ
    def gap_cat(gap):
        if gap >= 0.3: return 'Gap≥0.3'
        elif gap >= 0.1: return 'Gap 0.1-0.3'
        else: return 'Gap<0.1'
    
    results = []
    
    for bet_type in bet_types:
        # 全体
        stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
        for rid, rd in race_data.items():
            if rid not in payout_map:
                continue
            cost, ret, hit = simulate_bet(race_data, payout_map, rid, bet_type)
            if cost > 0:
                stats['races'] += 1
                stats['cost'] += cost
                stats['return'] += ret
                stats['hits'] += hit
        
        if stats['cost'] > 0:
            roi = stats['return'] / stats['cost'] * 100
            results.append({
                'bet_type': bet_type,
                'condition': '全体',
                'races': stats['races'],
                'roi': roi,
                'hit_rate': stats['hits'] / stats['races'] * 100 if stats['races'] > 0 else 0
            })
        
        # 人気別
        for pop_filter in ['1番人気', '2-3番人気', '4-6番人気', '7-10番人気']:
            stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
            for rid, rd in race_data.items():
                if rid not in payout_map:
                    continue
                if pop_cat(rd['top1_popularity']) != pop_filter:
                    continue
                cost, ret, hit = simulate_bet(race_data, payout_map, rid, bet_type)
                if cost > 0:
                    stats['races'] += 1
                    stats['cost'] += cost
                    stats['return'] += ret
                    stats['hits'] += hit
            
            if stats['races'] >= 30 and stats['cost'] > 0:
                roi = stats['return'] / stats['cost'] * 100
                results.append({
                    'bet_type': bet_type,
                    'condition': pop_filter,
                    'races': stats['races'],
                    'roi': roi,
                    'hit_rate': stats['hits'] / stats['races'] * 100
                })
        
        # スコア差別
        for gap_filter in ['Gap<0.1', 'Gap 0.1-0.3', 'Gap≥0.3']:
            stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
            for rid, rd in race_data.items():
                if rid not in payout_map:
                    continue
                if gap_cat(rd['score_gap']) != gap_filter:
                    continue
                cost, ret, hit = simulate_bet(race_data, payout_map, rid, bet_type)
                if cost > 0:
                    stats['races'] += 1
                    stats['cost'] += cost
                    stats['return'] += ret
                    stats['hits'] += hit
            
            if stats['races'] >= 30 and stats['cost'] > 0:
                roi = stats['return'] / stats['cost'] * 100
                results.append({
                    'bet_type': bet_type,
                    'condition': gap_filter,
                    'races': stats['races'],
                    'roi': roi,
                    'hit_rate': stats['hits'] / stats['races'] * 100
                })
    
    return results

def main():
    print("\n" + "#"*80)
    print("# 📊 馬券種類×条件 グリッドサーチ (2024+2025年)")
    print("#"*80)
    
    years = [2024, 2025]
    
    df = load_predictions_from_db(years)
    df = load_model_and_predict(df)
    df = preprocess_data(df)
    
    pay_df = load_payouts(years)
    payout_map = build_payout_map(pay_df)
    
    race_data = get_race_data(df)
    logger.info(f"Prepared data for {len(race_data)} races")
    
    results = run_bet_type_grid_search(race_data, payout_map)
    
    # 結果表示
    results = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'='*80}")
    print("📊 馬券種類×条件 グリッドサーチ結果 (ROI順)")
    print(f"{'='*80}")
    print(f"{'馬券種類':<12} | {'条件':<15} | {'Races':>6} | {'ROI':>8} | {'的中率':>7}")
    print("-" * 70)
    
    for r in results[:30]:
        print(f"{r['bet_type']:<12} | {r['condition']:<15} | {r['races']:>6} | {r['roi']:>7.1f}% | {r['hit_rate']:>6.1f}%")
    
    # ROI 100%以上のみ
    over_100 = [r for r in results if r['roi'] >= 100]
    
    print(f"\n{'='*80}")
    print(f"🏆 ROI 100%以上の戦略: {len(over_100)}件")
    print(f"{'='*80}")
    
    for r in over_100:
        print(f"  {r['bet_type']} x {r['condition']}: ROI {r['roi']:.1f}%, {r['races']}レース")
    
    # ファイル保存
    with open('reports/bet_type_grid_search.txt', 'w', encoding='utf-8') as f:
        f.write("=== 馬券種類×条件 グリッドサーチ (2024+2025年) ===\n\n")
        f.write("--- ROI上位30 ---\n")
        for r in results[:30]:
            f.write(f"{r['bet_type']} x {r['condition']}: ROI {r['roi']:.1f}%, {r['races']}レース, Hit {r['hit_rate']:.1f}%\n")
        
        f.write(f"\n--- ROI 100%以上: {len(over_100)}件 ---\n")
        for r in over_100:
            f.write(f"{r['bet_type']} x {r['condition']}: ROI {r['roi']:.1f}%, {r['races']}レース\n")
    
    print("\n結果を reports/bet_type_grid_search.txt に保存しました")
    print("\n✅ グリッドサーチ完了!")

if __name__ == "__main__":
    main()

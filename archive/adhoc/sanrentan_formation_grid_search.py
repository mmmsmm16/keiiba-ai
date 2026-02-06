"""
3連単フォーメーション網羅テスト
- Box買い (Top3/4/5/6)
- 1頭軸ながし (軸Top1, 相手Top2-N)
- 2頭軸ながし (軸Top1-2, 相手Top3-N)
- フォーメーション (1着/2着/3着候補を別々指定)
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
    payout_map = defaultdict(dict)
    
    for _, row in pay_df.iterrows():
        rid = row['race_id']
        
        for i in range(1, 7):
            col_a = f'haraimodoshi_sanrentan_{i}a'
            col_b = f'haraimodoshi_sanrentan_{i}b'
            if col_a in row and row[col_a] and str(row[col_a]).strip():
                try:
                    key = str(row[col_a]).strip()
                    val = int(float(str(row[col_b]).strip()))
                    payout_map[rid][key] = val
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
        
        # スコア分布
        scores = sorted_g.head(6)['score'].values
        score_range = scores[0] - scores[5]
        
        race_data[rid] = {
            'horses': sorted_g['horse_number'].astype(int).tolist(),
            'top1_popularity': int(top1['popularity']) if not pd.isna(top1['popularity']) else 99,
            'top1_odds': top1['odds'] if not pd.isna(top1['odds']) else 0,
            'score_range': score_range
        }
    
    return race_data

def simulate_sanrentan(rd, payout_map, rid, formation_type, params=None):
    """
    3連単シミュレーション
    
    formation_type:
    - 'box_N': Box買い (Top N馬の全順列)
    - 'nagashi_1_N': 1頭軸ながし (Top1軸, 相手Top2-N)
    - 'nagashi_2_N': 2頭軸ながし (Top1-2軸, 相手Top3-N)
    - 'formation_A_B_C': フォーメーション (1着Top1-A, 2着Top1-B, 3着Top1-C)
    """
    h = rd['horses']
    
    tickets = []
    
    if formation_type.startswith('box_'):
        # Box買い
        n = int(formation_type.split('_')[1])
        if len(h) < n:
            return 0, 0, 0
        top_n = h[:n]
        tickets = list(permutations(top_n, 3))
        
    elif formation_type.startswith('nagashi_1_'):
        # 1頭軸ながし
        n = int(formation_type.split('_')[2])
        if len(h) < n:
            return 0, 0, 0
        axis = h[0]
        opps = h[1:n]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        
    elif formation_type.startswith('nagashi_2_'):
        # 2頭軸ながし (Top1またはTop2が1着)
        n = int(formation_type.split('_')[2])
        if len(h) < n:
            return 0, 0, 0
        axis1, axis2 = h[0], h[1]
        opps = h[2:n]
        # Top1が1着
        tickets += [(axis1, axis2, o) for o in opps]
        tickets += [(axis1, o, axis2) for o in opps]
        for o1, o2 in permutations(opps, 2):
            tickets.append((axis1, o1, o2))
        # Top2が1着
        tickets += [(axis2, axis1, o) for o in opps]
        tickets += [(axis2, o, axis1) for o in opps]
        for o1, o2 in permutations(opps, 2):
            tickets.append((axis2, o1, o2))
        tickets = list(set(tickets))
        
    elif formation_type.startswith('formation_'):
        # フォーメーション
        parts = formation_type.split('_')
        a, b, c = int(parts[1]), int(parts[2]), int(parts[3])
        if len(h) < max(a, b, c):
            return 0, 0, 0
        first = h[:a]
        second = h[:b]
        third = h[:c]
        for f in first:
            for s in second:
                for t in third:
                    if f != s and s != t and f != t:
                        tickets.append((f, s, t))
        tickets = list(set(tickets))
        
    elif formation_type == 'top1_fixed_23':
        # Top1頭固定, 2着3着はTop2-3のみ
        if len(h) < 3:
            return 0, 0, 0
        axis = h[0]
        tickets = [(axis, h[1], h[2]), (axis, h[2], h[1])]
        
    elif formation_type == 'top1_fixed_234':
        # Top1頭固定, 2着3着はTop2-4
        if len(h) < 4:
            return 0, 0, 0
        axis = h[0]
        opps = h[1:4]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        
    elif formation_type == 'top12_1st_rest':
        # Top1かTop2が1着, 残りはTop3-6
        if len(h) < 6:
            return 0, 0, 0
        for first in h[:2]:
            for s, t in permutations(h[2:6], 2):
                tickets.append((first, s, t))
        tickets = list(set(tickets))
        
    elif formation_type == 'top1_2nd_fixed':
        # Top1が1着, Top2が2着固定, 3着はTop3-6
        if len(h) < 6:
            return 0, 0, 0
        tickets = [(h[0], h[1], h[i]) for i in range(2, 6)]
        
    elif formation_type == 'reverse_formation':
        # Top1-3のうちどれかが1着2着3着 (6点)
        if len(h) < 3:
            return 0, 0, 0
        tickets = list(permutations(h[:3], 3))
    
    if not tickets:
        return 0, 0, 0
    
    cost = len(tickets) * 100
    ret = 0
    hit = 0
    
    for t in tickets:
        key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
        if key in payout_map.get(rid, {}):
            ret += payout_map[rid][key]
            hit = 1
    
    return cost, ret, hit

def run_sanrentan_grid_search(race_data, payout_map):
    """3連単フォーメーション網羅テスト"""
    
    formations = [
        # Box
        ('box_3', 'Box Top3 (6点)'),
        ('box_4', 'Box Top4 (24点)'),
        ('box_5', 'Box Top5 (60点)'),
        ('box_6', 'Box Top6 (120点)'),
        
        # 1頭軸ながし
        ('nagashi_1_3', '1頭軸 Top1軸-相手Top2-3 (2点)'),
        ('nagashi_1_4', '1頭軸 Top1軸-相手Top2-4 (6点)'),
        ('nagashi_1_5', '1頭軸 Top1軸-相手Top2-5 (12点)'),
        ('nagashi_1_6', '1頭軸 Top1軸-相手Top2-6 (20点)'),
        ('nagashi_1_7', '1頭軸 Top1軸-相手Top2-7 (30点)'),
        
        # 2頭軸ながし
        ('nagashi_2_4', '2頭軸 Top1-2軸-相手Top3-4'),
        ('nagashi_2_5', '2頭軸 Top1-2軸-相手Top3-5'),
        ('nagashi_2_6', '2頭軸 Top1-2軸-相手Top3-6'),
        
        # フォーメーション
        ('formation_1_2_3', 'F 1着Top1/2着Top1-2/3着Top1-3'),
        ('formation_1_3_5', 'F 1着Top1/2着Top1-3/3着Top1-5'),
        ('formation_2_3_5', 'F 1着Top1-2/2着Top1-3/3着Top1-5'),
        ('formation_2_4_6', 'F 1着Top1-2/2着Top1-4/3着Top1-6'),
        ('formation_3_5_6', 'F 1着Top1-3/2着Top1-5/3着Top1-6'),
        
        # 特殊パターン
        ('top1_fixed_23', 'Top1固定-2,3着Top2-3のみ (2点)'),
        ('top1_fixed_234', 'Top1固定-2,3着Top2-4 (6点)'),
        ('top12_1st_rest', 'Top1-2が1着-相手Top3-6'),
        ('top1_2nd_fixed', 'Top1-1着/Top2-2着固定 (4点)'),
        ('reverse_formation', 'Top3内のみ全順列 (6点)'),
    ]
    
    results = []
    
    # 全体テスト
    print("\n--- 全レースでのテスト ---")
    for form_id, form_name in formations:
        stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
        
        for rid, rd in race_data.items():
            if rid not in payout_map:
                continue
            
            cost, ret, hit = simulate_sanrentan(rd, payout_map, rid, form_id)
            if cost > 0:
                stats['races'] += 1
                stats['cost'] += cost
                stats['return'] += ret
                stats['hits'] += hit
        
        if stats['cost'] > 0:
            roi = stats['return'] / stats['cost'] * 100
            hit_rate = stats['hits'] / stats['races'] * 100 if stats['races'] > 0 else 0
            results.append({
                'formation': form_id,
                'name': form_name,
                'condition': '全体',
                'races': stats['races'],
                'cost': stats['cost'],
                'roi': roi,
                'hit_rate': hit_rate
            })
    
    # 人気条件別テスト
    print("\n--- 人気条件別テスト ---")
    pop_conditions = [
        ('pop_1', lambda rd: rd['top1_popularity'] == 1, '1番人気'),
        ('pop_2-3', lambda rd: 2 <= rd['top1_popularity'] <= 3, '2-3番人気'),
        ('pop_4-6', lambda rd: 4 <= rd['top1_popularity'] <= 6, '4-6番人気'),
        ('pop_7+', lambda rd: rd['top1_popularity'] >= 7, '7番人気以上'),
    ]
    
    for form_id, form_name in formations[:10]:  # 主要なフォーメーションのみ
        for cond_id, cond_func, cond_name in pop_conditions:
            stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
            
            for rid, rd in race_data.items():
                if rid not in payout_map:
                    continue
                if not cond_func(rd):
                    continue
                
                cost, ret, hit = simulate_sanrentan(rd, payout_map, rid, form_id)
                if cost > 0:
                    stats['races'] += 1
                    stats['cost'] += cost
                    stats['return'] += ret
                    stats['hits'] += hit
            
            if stats['races'] >= 30 and stats['cost'] > 0:
                roi = stats['return'] / stats['cost'] * 100
                hit_rate = stats['hits'] / stats['races'] * 100
                results.append({
                    'formation': form_id,
                    'name': form_name,
                    'condition': cond_name,
                    'races': stats['races'],
                    'cost': stats['cost'],
                    'roi': roi,
                    'hit_rate': hit_rate
                })
    
    # スコア分布条件別テスト
    print("\n--- スコア分布条件別テスト ---")
    score_conditions = [
        ('gap_small', lambda rd: rd['score_range'] < 0.3, '均衡(gap<0.3)'),
        ('gap_medium', lambda rd: 0.3 <= rd['score_range'] < 0.6, '中差(0.3-0.6)'),
        ('gap_large', lambda rd: rd['score_range'] >= 0.6, '大差(gap≥0.6)'),
    ]
    
    for form_id, form_name in formations[:10]:
        for cond_id, cond_func, cond_name in score_conditions:
            stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
            
            for rid, rd in race_data.items():
                if rid not in payout_map:
                    continue
                if not cond_func(rd):
                    continue
                
                cost, ret, hit = simulate_sanrentan(rd, payout_map, rid, form_id)
                if cost > 0:
                    stats['races'] += 1
                    stats['cost'] += cost
                    stats['return'] += ret
                    stats['hits'] += hit
            
            if stats['races'] >= 30 and stats['cost'] > 0:
                roi = stats['return'] / stats['cost'] * 100
                hit_rate = stats['hits'] / stats['races'] * 100
                results.append({
                    'formation': form_id,
                    'name': form_name,
                    'condition': cond_name,
                    'races': stats['races'],
                    'cost': stats['cost'],
                    'roi': roi,
                    'hit_rate': hit_rate
                })
    
    return results

def main():
    print("\n" + "#"*80)
    print("# 📊 3連単フォーメーション網羅テスト (2024+2025年)")
    print("#"*80)
    
    years = [2024, 2025]
    
    df = load_data(years)
    df = load_model_and_predict(df)
    
    pay_df = load_payouts(years)
    payout_map = build_payout_map(pay_df)
    
    race_data = get_race_data(df)
    logger.info(f"Prepared data for {len(race_data)} races")
    
    results = run_sanrentan_grid_search(race_data, payout_map)
    
    # 結果表示
    results = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'='*90}")
    print("📊 3連単フォーメーション グリッドサーチ結果 (ROI順 Top30)")
    print(f"{'='*90}")
    print(f"{'フォーメーション':<30} | {'条件':<12} | {'Races':>6} | {'点数':>8} | {'ROI':>8} | {'的中率':>7}")
    print("-" * 90)
    
    for r in results[:30]:
        avg_cost = r['cost'] / r['races'] if r['races'] > 0 else 0
        print(f"{r['name']:<30} | {r['condition']:<12} | {r['races']:>6} | {avg_cost/100:>7.0f}点 | {r['roi']:>7.1f}% | {r['hit_rate']:>6.1f}%")
    
    # ROI 100%以上
    over_100 = [r for r in results if r['roi'] >= 100]
    
    print(f"\n{'='*90}")
    print(f"🏆 ROI 100%以上の戦略: {len(over_100)}件")
    print(f"{'='*90}")
    
    for r in sorted(over_100, key=lambda x: x['roi'], reverse=True):
        avg_cost = r['cost'] / r['races'] if r['races'] > 0 else 0
        print(f"  {r['name']} x {r['condition']}: ROI {r['roi']:.1f}%, {r['races']}レース, {avg_cost/100:.0f}点")
    
    # 効率性分析 (ROI/点数)
    print(f"\n{'='*90}")
    print("💡 効率性分析 (ROI÷点数で効率順)")
    print(f"{'='*90}")
    
    for r in results:
        avg_cost = r['cost'] / r['races'] if r['races'] > 0 else 1
        r['efficiency'] = r['roi'] / (avg_cost / 100)
    
    efficiency_sorted = sorted(results, key=lambda x: x['efficiency'], reverse=True)
    
    for r in efficiency_sorted[:15]:
        avg_cost = r['cost'] / r['races'] if r['races'] > 0 else 0
        print(f"  {r['name']} x {r['condition']}: 効率 {r['efficiency']:.1f}, ROI {r['roi']:.1f}%, {avg_cost/100:.0f}点")
    
    # ファイル保存
    with open('reports/sanrentan_formation_grid_search.txt', 'w', encoding='utf-8') as f:
        f.write("=== 3連単フォーメーション網羅テスト (2024+2025年) ===\n\n")
        
        f.write("--- ROI上位30 ---\n")
        for r in results[:30]:
            avg_cost = r['cost'] / r['races'] if r['races'] > 0 else 0
            f.write(f"{r['name']} x {r['condition']}: ROI {r['roi']:.1f}%, {r['races']}レース, {avg_cost/100:.0f}点, Hit {r['hit_rate']:.1f}%\n")
        
        f.write(f"\n--- ROI 100%以上: {len(over_100)}件 ---\n")
        for r in sorted(over_100, key=lambda x: x['roi'], reverse=True):
            f.write(f"{r['name']} x {r['condition']}: ROI {r['roi']:.1f}%\n")
        
        f.write("\n--- 効率性ランキング ---\n")
        for r in efficiency_sorted[:15]:
            f.write(f"{r['name']} x {r['condition']}: 効率 {r['efficiency']:.1f}\n")
    
    print("\n結果を reports/sanrentan_formation_grid_search.txt に保存しました")
    print("\n✅ グリッドサーチ完了!")

if __name__ == "__main__":
    main()

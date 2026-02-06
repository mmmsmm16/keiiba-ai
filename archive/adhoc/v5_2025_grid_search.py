"""
v5_2025モデル専用グリッドサーチ (2025年データのみ)
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

def load_data():
    """2025年データのみロード"""
    data_path = 'data/processed/preprocessed_data.parquet'
    df = pd.read_parquet(data_path)
    df = df[df['year'] == 2025].copy()
    logger.info(f"Loaded {len(df)} rows for 2025 only")
    return df

def load_model_and_predict(df):
    sys.path.append('src')
    from model.ensemble import EnsembleModel
    
    model = EnsembleModel()
    model.load_model('models/ensemble_v5_2025.pkl')  # v5_2025を使用
    
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
    
    df['prob'] = df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(1.0).replace(0, 1.0)
    df['ev'] = df['prob'] * df['odds']
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(99)
    
    return df

def load_payouts():
    """2025年のみ"""
    engine = get_db_engine()
    query = text("SELECT * FROM jvd_hr WHERE kaisai_nen = '2025'")
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
    payout_map = defaultdict(lambda: {'tansho': {}, 'umaren': {}, 'sanrenpuku': {}, 'sanrentan': {}})
    
    for _, row in pay_df.iterrows():
        rid = row['race_id']
        
        for prefix, max_count in [('haraimodoshi_tansho', 3), ('haraimodoshi_umaren', 3),
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

# ランキング方法
def rank_by_score(grp):
    return grp.sort_values('score', ascending=False)

def rank_group_top2fix(grp):
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6:
        return sorted_g
    top2 = sorted_g.head(2)
    rank3_6 = sorted_g.iloc[2:6].sort_values('ev', ascending=False)
    rest = sorted_g.iloc[6:]
    return pd.concat([top2, rank3_6, rest])

def rank_score_then_ev_5(grp):
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) <= 5:
        return sorted_g.sort_values('ev', ascending=False)
    top = sorted_g.head(5).sort_values('ev', ascending=False)
    rest = sorted_g.iloc[5:]
    return pd.concat([top, rest])

def rank_weighted_07(grp):
    grp = grp.copy()
    score_min, score_max = grp['score'].min(), grp['score'].max()
    ev_min, ev_max = grp['ev'].min(), grp['ev'].max()
    if score_max > score_min:
        grp['score_norm'] = (grp['score'] - score_min) / (score_max - score_min)
    else:
        grp['score_norm'] = 0.5
    if ev_max > ev_min:
        grp['ev_norm'] = (grp['ev'] - ev_min) / (ev_max - ev_min)
    else:
        grp['ev_norm'] = 0.5
    grp['weighted'] = 0.7 * grp['score_norm'] + 0.3 * grp['ev_norm']
    return grp.sort_values('weighted', ascending=False)

# 条件判定
def get_race_conditions(grp):
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6:
        return {}
    
    scores = sorted_g['score'].values[:6]
    score_range = scores[0] - scores[5]
    top3_gap = scores[0] - scores[2]
    bottom_gap = scores[2] - scores[5]
    top1_pop = int(sorted_g.iloc[0]['popularity']) if not pd.isna(sorted_g.iloc[0]['popularity']) else 99
    
    if score_range >= 0.5:
        dist_cat = 'large_gap'
    elif score_range >= 0.3:
        dist_cat = 'medium_gap'
    elif score_range >= 0.15:
        dist_cat = 'small_gap'
    else:
        dist_cat = 'balanced'
    
    if top3_gap >= 0.3 and bottom_gap < 0.1:
        structure = 'top3_dominant'
    else:
        structure = 'normal'
    
    if top1_pop >= 7:
        pop_cat = 'longshot'
    elif top1_pop >= 4:
        pop_cat = 'midrange'
    else:
        pop_cat = 'favorite'
    
    return {
        'dist_cat': dist_cat, 'structure': structure, 'pop_cat': pop_cat,
        'top1_pop': top1_pop, 'score_range': score_range,
    }

# ベットシミュレーション
def simulate_bet(horses, payout_map, rid, bet_type):
    if rid not in payout_map:
        return 0, 0, 0
    pm = payout_map[rid]
    
    if bet_type == 'tansho':
        if len(horses) < 1: return 0, 0, 0
        axis = horses[0]
        key = f"{axis:02}"
        ret = pm['tansho'].get(key, 0)
        return 100, ret, 1 if ret > 0 else 0
    
    elif bet_type == 'umaren_3':
        if len(horses) < 4: return 0, 0, 0
        axis = horses[0]
        opps = horses[1:4]
        cost = len(opps) * 100
        ret, hit = 0, 0
        for opp in opps:
            c_sorted = sorted([axis, opp])
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in pm['umaren']:
                ret += pm['umaren'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'sanrenpuku':
        if len(horses) < 6: return 0, 0, 0
        axis = horses[0]
        opps = horses[1:6]
        tickets = [t for t in combinations([axis] + opps, 3) if axis in t]
        cost = len(tickets) * 100
        ret, hit = 0, 0
        for t in tickets:
            key = f"{sorted(t)[0]:02}{sorted(t)[1]:02}{sorted(t)[2]:02}"
            if key in pm['sanrenpuku']:
                ret += pm['sanrenpuku'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'sanrentan_6':
        if len(horses) < 4: return 0, 0, 0
        axis = horses[0]
        opps = horses[1:4]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        cost = len(tickets) * 100
        ret, hit = 0, 0
        for t in tickets:
            key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
            if key in pm['sanrentan']:
                ret += pm['sanrentan'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'sanrentan_20':
        if len(horses) < 6: return 0, 0, 0
        axis = horses[0]
        opps = horses[1:6]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        cost = len(tickets) * 100
        ret, hit = 0, 0
        for t in tickets:
            key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
            if key in pm['sanrentan']:
                ret += pm['sanrentan'][key]
                hit = 1
        return cost, ret, hit
    
    return 0, 0, 0

def run_grid_search(df, payout_map):
    ranking_methods = {
        'score': rank_by_score,
        'group_top2fix': rank_group_top2fix,
        'score_then_ev_5': rank_score_then_ev_5,
        'weighted_0.7': rank_weighted_07,
    }
    betting_strategies = ['tansho', 'umaren_3', 'sanrenpuku', 'sanrentan_6', 'sanrentan_20']
    
    conditions = {
        'all': lambda c: True,
        'top3_dominant': lambda c: c.get('structure') == 'top3_dominant',
        'longshot': lambda c: c.get('pop_cat') == 'longshot',
        'midrange': lambda c: c.get('pop_cat') == 'midrange',
        'favorite': lambda c: c.get('pop_cat') == 'favorite',
        'large_gap': lambda c: c.get('dist_cat') == 'large_gap',
        'medium_gap': lambda c: c.get('dist_cat') == 'medium_gap',
        'small_gap': lambda c: c.get('dist_cat') == 'small_gap',
        'balanced': lambda c: c.get('dist_cat') == 'balanced',
    }
    
    results = []
    
    race_data = {}
    for rid, grp in df.groupby('race_id'):
        if len(grp) < 6: continue
        cond = get_race_conditions(grp)
        if not cond: continue
        race_data[rid] = {'grp': grp, 'cond': cond}
    
    logger.info(f"Processing {len(race_data)} races (2025 only, v5_2025 model)...")
    
    for cond_name, cond_func in conditions.items():
        matching_races = {rid: data for rid, data in race_data.items() if cond_func(data['cond'])}
        if len(matching_races) < 20: continue
        
        for rank_name, rank_func in ranking_methods.items():
            for bet_type in betting_strategies:
                stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
                
                for rid, data in matching_races.items():
                    sorted_grp = rank_func(data['grp'].copy())
                    horses = sorted_grp['horse_number'].astype(int).tolist()
                    cost, ret, hit = simulate_bet(horses, payout_map, rid, bet_type)
                    if cost > 0:
                        stats['races'] += 1
                        stats['cost'] += cost
                        stats['return'] += ret
                        stats['hits'] += hit
                
                if stats['races'] >= 20 and stats['cost'] > 0:
                    roi = stats['return'] / stats['cost'] * 100
                    hit_rate = stats['hits'] / stats['races'] * 100
                    results.append({
                        'condition': cond_name,
                        'ranking': rank_name,
                        'betting': bet_type,
                        'races': stats['races'],
                        'roi': roi,
                        'hit_rate': hit_rate,
                    })
    
    return results

def main():
    print("\n" + "#"*80)
    print("# 🎯 v5_2025モデル × 2025年データ グリッドサーチ")
    print("# (Out-of-Sample: 2025年のみ)")
    print("#"*80)
    
    df = load_data()
    df = load_model_and_predict(df)
    
    pay_df = load_payouts()
    payout_map = build_payout_map(pay_df)
    logger.info(f"Loaded payouts for {len(payout_map)} races")
    
    results = run_grid_search(df, payout_map)
    results = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'='*100}")
    print("📊 v5_2025 × 2025年 グリッドサーチ結果 (ROI上位20)")
    print(f"{'='*100}")
    print(f"{'条件':<18} | {'ランキング':<18} | {'馬券':<14} | {'Races':>6} | {'ROI':>8} | {'的中率':>7}")
    print("-" * 90)
    
    for r in results[:20]:
        print(f"{r['condition']:<18} | {r['ranking']:<18} | {r['betting']:<14} | {r['races']:>6} | {r['roi']:>7.1f}% | {r['hit_rate']:>6.1f}%")
    
    over_100 = [r for r in results if r['roi'] >= 100]
    print(f"\n{'='*100}")
    print(f"🏆 ROI 100%以上の組み合わせ: {len(over_100)}件")
    print(f"{'='*100}")
    for r in over_100:
        print(f"  {r['condition']} × {r['ranking']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース")
    
    # 条件別ベスト
    print(f"\n{'='*100}")
    print("🔍 条件別 ベスト組み合わせ (v5_2025 × 2025年)")
    print(f"{'='*100}")
    cond_best = {}
    for r in results:
        if r['condition'] not in cond_best or r['roi'] > cond_best[r['condition']]['roi']:
            cond_best[r['condition']] = r
    for cond, r in sorted(cond_best.items(), key=lambda x: x[1]['roi'], reverse=True):
        print(f"  {cond}: {r['ranking']} × {r['betting']} → ROI {r['roi']:.1f}% ({r['races']}レース)")
    
    # ファイル保存
    os.makedirs('reports', exist_ok=True)
    with open('reports/v5_2025_grid_search.txt', 'w', encoding='utf-8') as f:
        f.write("=== v5_2025モデル × 2025年データ グリッドサーチ ===\n\n")
        f.write("--- ROI上位20 ---\n")
        for r in results[:20]:
            f.write(f"{r['condition']} × {r['ranking']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース\n")
        f.write(f"\n--- ROI 100%以上: {len(over_100)}件 ---\n")
        for r in over_100:
            f.write(f"{r['condition']} × {r['ranking']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース\n")
        f.write("\n--- 条件別ベスト ---\n")
        for cond, r in sorted(cond_best.items(), key=lambda x: x[1]['roi'], reverse=True):
            f.write(f"{cond}: {r['ranking']} × {r['betting']} → ROI {r['roi']:.1f}%\n")
    
    print("\n結果を reports/v5_2025_grid_search.txt に保存しました")
    print("\n✅ グリッドサーチ完了!")

if __name__ == "__main__":
    main()

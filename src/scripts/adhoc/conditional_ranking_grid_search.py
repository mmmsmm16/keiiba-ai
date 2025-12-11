"""
条件付きランキング × 馬券戦略 グリッドサーチ
- group_top2fix等の新ランキング方法 × スコア分布条件 × 馬券戦略
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

# ============================================================
# データロード関数
# ============================================================

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
    model.load_model('models/ensemble_v5_2025.pkl')
    
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

# ============================================================
# ランキング方法 (上位3つに絞る)
# ============================================================

def rank_by_score(grp):
    return grp.sort_values('score', ascending=False)

def rank_group_top2fix(grp):
    """Top2固定、3-6をEVでre-rank"""
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

# ============================================================
# 条件判定
# ============================================================

def get_race_conditions(grp):
    """レースの条件を計算"""
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6:
        return {}
    
    scores = sorted_g['score'].values[:6]
    
    # スコア分布
    score_range = scores[0] - scores[5]
    top3_gap = scores[0] - scores[2]
    bottom_gap = scores[2] - scores[5]
    
    # Top1の人気
    top1_pop = int(sorted_g.iloc[0]['popularity']) if not pd.isna(sorted_g.iloc[0]['popularity']) else 99
    
    # Top1のオッズ
    top1_odds = sorted_g.iloc[0]['odds'] if not pd.isna(sorted_g.iloc[0]['odds']) else 1.0
    
    # スコア分布カテゴリ
    if score_range >= 0.5:
        dist_cat = 'large_gap'
    elif score_range >= 0.3:
        dist_cat = 'medium_gap'
    elif score_range >= 0.15:
        dist_cat = 'small_gap'
    else:
        dist_cat = 'balanced'
    
    # Top3 vs Bottom3
    if top3_gap >= 0.3 and bottom_gap < 0.1:
        structure = 'top3_dominant'
    elif top3_gap < 0.1 and bottom_gap >= 0.2:
        structure = 'bottom_spread'
    else:
        structure = 'normal'
    
    # 人気カテゴリ
    if top1_pop >= 7:
        pop_cat = 'longshot'  # 7番人気以上
    elif top1_pop >= 4:
        pop_cat = 'midrange'  # 4-6番人気
    else:
        pop_cat = 'favorite'  # 1-3番人気
    
    return {
        'dist_cat': dist_cat,
        'structure': structure,
        'pop_cat': pop_cat,
        'top1_pop': top1_pop,
        'top1_odds': top1_odds,
        'score_range': score_range,
    }

# ============================================================
# 馬券シミュレーション
# ============================================================

def simulate_bet(horses, payout_map, rid, bet_type):
    if rid not in payout_map:
        return 0, 0, 0
    
    pm = payout_map[rid]
    
    if bet_type == 'tansho':
        if len(horses) < 1:
            return 0, 0, 0
        axis = horses[0]
        key = f"{axis:02}"
        ret = pm['tansho'].get(key, 0)
        return 100, ret, 1 if ret > 0 else 0
    
    elif bet_type == 'umaren_3':
        if len(horses) < 4:
            return 0, 0, 0
        axis = horses[0]
        opps = horses[1:4]
        cost = len(opps) * 100
        ret = 0
        hit = 0
        for opp in opps:
            c_sorted = sorted([axis, opp])
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in pm['umaren']:
                ret += pm['umaren'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'sanrenpuku':
        if len(horses) < 6:
            return 0, 0, 0
        axis = horses[0]
        opps = horses[1:6]
        tickets = list(combinations([axis] + opps, 3))
        tickets = [t for t in tickets if axis in t]
        cost = len(tickets) * 100
        ret = 0
        hit = 0
        for t in tickets:
            c_sorted = sorted(t)
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
            if key in pm['sanrenpuku']:
                ret += pm['sanrenpuku'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'sanrentan_6':
        if len(horses) < 4:
            return 0, 0, 0
        axis = horses[0]
        opps = horses[1:4]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        cost = len(tickets) * 100
        ret = 0
        hit = 0
        for t in tickets:
            key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
            if key in pm['sanrentan']:
                ret += pm['sanrentan'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'sanrentan_20':
        if len(horses) < 6:
            return 0, 0, 0
        axis = horses[0]
        opps = horses[1:6]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        cost = len(tickets) * 100
        ret = 0
        hit = 0
        for t in tickets:
            key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
            if key in pm['sanrentan']:
                ret += pm['sanrentan'][key]
                hit = 1
        return cost, ret, hit
    
    return 0, 0, 0

# ============================================================
# 条件付きグリッドサーチ
# ============================================================

def run_conditional_grid_search(df, payout_map):
    """条件付きグリッドサーチ"""
    
    # ベストランキング方法のみ
    ranking_methods = {
        'score': rank_by_score,
        'group_top2fix': rank_group_top2fix,
        'score_then_ev_5': rank_score_then_ev_5,
        'weighted_0.7': rank_weighted_07,
    }
    
    betting_strategies = ['tansho', 'umaren_3', 'sanrenpuku', 'sanrentan_6', 'sanrentan_20']
    
    # 条件フィルター
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
        'longshot+large': lambda c: c.get('pop_cat') == 'longshot' and c.get('dist_cat') in ['large_gap', 'medium_gap'],
        'midrange+top3dom': lambda c: c.get('pop_cat') == 'midrange' and c.get('structure') == 'top3_dominant',
    }
    
    results = []
    
    # レースごとにグループ化とメタデータ計算
    race_data = {}
    for rid, grp in df.groupby('race_id'):
        if len(grp) < 6:
            continue
        cond = get_race_conditions(grp)
        if not cond:
            continue
        race_data[rid] = {'grp': grp, 'cond': cond}
    
    logger.info(f"Processing {len(race_data)} races with conditions...")
    
    for cond_name, cond_func in conditions.items():
        # この条件に該当するレースを抽出
        matching_races = {rid: data for rid, data in race_data.items() if cond_func(data['cond'])}
        
        if len(matching_races) < 30:
            continue
        
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
                
                if stats['races'] >= 30 and stats['cost'] > 0:
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
    print("# 🎯 条件付きランキング × 馬券戦略 グリッドサーチ (2024+2025年)")
    print("# 仮説: group_top2fix × 条件フィルター で ROI 100%超え")
    print("#"*80)
    
    years = [2024, 2025]
    
    df = load_data(years)
    df = load_model_and_predict(df)
    
    pay_df = load_payouts(years)
    payout_map = build_payout_map(pay_df)
    logger.info(f"Loaded payouts for {len(payout_map)} races")
    
    results = run_conditional_grid_search(df, payout_map)
    
    # 結果表示
    results = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'='*100}")
    print("📊 条件付きグリッドサーチ結果 (ROI上位30)")
    print(f"{'='*100}")
    print(f"{'条件':<20} | {'ランキング':<18} | {'馬券':<14} | {'Races':>6} | {'ROI':>8} | {'的中率':>7}")
    print("-" * 95)
    
    for r in results[:30]:
        print(f"{r['condition']:<20} | {r['ranking']:<18} | {r['betting']:<14} | {r['races']:>6} | {r['roi']:>7.1f}% | {r['hit_rate']:>6.1f}%")
    
    # ROI 100%以上
    over_100 = [r for r in results if r['roi'] >= 100]
    
    print(f"\n{'='*100}")
    print(f"🏆 ROI 100%以上の組み合わせ: {len(over_100)}件")
    print(f"{'='*100}")
    
    for r in over_100:
        print(f"  {r['condition']} × {r['ranking']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース, Hit {r['hit_rate']:.1f}%")
    
    # group_top2fix のベスト
    print(f"\n{'='*100}")
    print("📈 group_top2fix の条件別ベスト")
    print(f"{'='*100}")
    
    top2fix_results = [r for r in results if r['ranking'] == 'group_top2fix']
    top2fix_results = sorted(top2fix_results, key=lambda x: x['roi'], reverse=True)
    
    for r in top2fix_results[:15]:
        print(f"  {r['condition']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース")
    
    # 条件別ベスト
    print(f"\n{'='*100}")
    print("🔍 条件別 ベスト組み合わせ")
    print(f"{'='*100}")
    
    cond_best = {}
    for r in results:
        if r['condition'] not in cond_best or r['roi'] > cond_best[r['condition']]['roi']:
            cond_best[r['condition']] = r
    
    for cond, r in sorted(cond_best.items(), key=lambda x: x[1]['roi'], reverse=True):
        print(f"  {cond}: {r['ranking']} × {r['betting']} → ROI {r['roi']:.1f}%")
    
    # ファイル保存
    os.makedirs('reports', exist_ok=True)
    with open('reports/conditional_ranking_grid_search.txt', 'w', encoding='utf-8') as f:
        f.write("=== 条件付きランキング × 馬券戦略 グリッドサーチ (2024+2025年) ===\n\n")
        
        f.write("--- ROI上位30 ---\n")
        for r in results[:30]:
            f.write(f"{r['condition']} × {r['ranking']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース\n")
        
        f.write(f"\n--- ROI 100%以上: {len(over_100)}件 ---\n")
        for r in over_100:
            f.write(f"{r['condition']} × {r['ranking']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース\n")
        
        f.write("\n--- group_top2fix ベスト ---\n")
        for r in top2fix_results[:10]:
            f.write(f"{r['condition']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース\n")
    
    print("\n結果を reports/conditional_ranking_grid_search.txt に保存しました")
    print("\n✅ グリッドサーチ完了!")

if __name__ == "__main__":
    main()

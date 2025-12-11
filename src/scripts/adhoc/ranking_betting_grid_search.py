"""
ランキング方法 × 馬券戦略 グリッドサーチ
- 仮説: スコアが近い馬の中ではEVで並び替えたほうが回収率向上
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
    
    # Softmax prob (per race)
    df['prob'] = df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    
    # EV計算
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(1.0).replace(0, 1.0)
    df['ev'] = df['prob'] * df['odds']
    
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
# ランキング方法 (並び替え関数)
# ============================================================

def rank_by_score(grp):
    """現行: スコア順"""
    return grp.sort_values('score', ascending=False)

def rank_by_ev(grp):
    """期待値順"""
    return grp.sort_values('ev', ascending=False)

def rank_score_then_ev(grp, top_n=3):
    """スコア上位N頭をEVでre-rank"""
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) <= top_n:
        return sorted_g.sort_values('ev', ascending=False)
    
    top = sorted_g.head(top_n).sort_values('ev', ascending=False)
    rest = sorted_g.iloc[top_n:]
    return pd.concat([top, rest])

def rank_ev_weighted(grp, alpha=0.5):
    """スコアとEVの加重合成"""
    grp = grp.copy()
    # normalize score and ev to [0, 1] within race
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
    
    grp['weighted'] = alpha * grp['score_norm'] + (1 - alpha) * grp['ev_norm']
    return grp.sort_values('weighted', ascending=False)

def rank_boost_high_ev(grp, ev_threshold=1.2):
    """EV > threshold の馬がいればTop1と入替"""
    sorted_g = grp.sort_values('score', ascending=False).copy()
    if len(sorted_g) < 2:
        return sorted_g
    
    top1_idx = sorted_g.index[0]
    
    # 2位以下でEV閾値を超える馬を探す
    for i in range(1, len(sorted_g)):
        if sorted_g.iloc[i]['ev'] > ev_threshold:
            # swap
            indices = list(sorted_g.index)
            indices[0], indices[i] = indices[i], indices[0]
            return sorted_g.loc[indices]
    
    return sorted_g

def rank_exclude_pop1(grp):
    """1番人気をTop1から除外"""
    sorted_g = grp.sort_values('score', ascending=False).copy()
    if len(sorted_g) < 2:
        return sorted_g
    
    pop_col = 'popularity'
    if pop_col not in sorted_g.columns:
        return sorted_g
    
    sorted_g['popularity'] = pd.to_numeric(sorted_g['popularity'], errors='coerce').fillna(99)
    
    top1 = sorted_g.iloc[0]
    if top1['popularity'] == 1:
        # 1番人気が1位の場合、2位と入替
        indices = list(sorted_g.index)
        indices[0], indices[1] = indices[1], indices[0]
        return sorted_g.loc[indices]
    
    return sorted_g

# ============================================================
# グループ別並び替え (新規追加)
# ============================================================

def rank_group_ev(grp):
    """Top3グループ内 + 4-6グループ内 それぞれEVで並び替え"""
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6:
        return sorted_g
    
    top3 = sorted_g.head(3).sort_values('ev', ascending=False)
    mid3 = sorted_g.iloc[3:6].sort_values('ev', ascending=False)
    rest = sorted_g.iloc[6:]
    
    return pd.concat([top3, mid3, rest])

def rank_group_ev_top3_only(grp):
    """Top3グループ内のみEVで並び替え、4-6以下は元のスコア順"""
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 3:
        return sorted_g
    
    top3 = sorted_g.head(3).sort_values('ev', ascending=False)
    rest = sorted_g.iloc[3:]
    
    return pd.concat([top3, rest])

def rank_group_ev_456_only(grp):
    """Top3はスコア順固定、4-6グループ内のみEVで並び替え"""
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6:
        return sorted_g
    
    top3 = sorted_g.head(3)  # スコア順のまま
    mid3 = sorted_g.iloc[3:6].sort_values('ev', ascending=False)
    rest = sorted_g.iloc[6:]
    
    return pd.concat([top3, mid3, rest])

def rank_group_ev_top1_fix(grp):
    """Top1固定、2-3グループ内 + 4-6グループ内 それぞれEVで並び替え"""
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6:
        return sorted_g
    
    top1 = sorted_g.head(1)  # 1位固定
    rank2_3 = sorted_g.iloc[1:3].sort_values('ev', ascending=False)
    rank4_6 = sorted_g.iloc[3:6].sort_values('ev', ascending=False)
    rest = sorted_g.iloc[6:]
    
    return pd.concat([top1, rank2_3, rank4_6, rest])

def rank_group_ev_top2_fix(grp):
    """Top2固定、3-6グループ内EVで並び替え"""
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6:
        return sorted_g
    
    top2 = sorted_g.head(2)  # 1-2位固定
    rank3_6 = sorted_g.iloc[2:6].sort_values('ev', ascending=False)
    rest = sorted_g.iloc[6:]
    
    return pd.concat([top2, rank3_6, rest])

# ============================================================
# 馬券シミュレーション
# ============================================================

def simulate_bet(horses, payout_map, rid, bet_type):
    """
    Args:
        horses: 馬番リスト (並び替え済み)
        payout_map: 払戻データ
        rid: レースID
        bet_type: 馬券種類
    Returns:
        (cost, return, hit_flag)
    """
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
    
    elif bet_type == 'umaren_5':
        if len(horses) < 6:
            return 0, 0, 0
        axis = horses[0]
        opps = horses[1:6]
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
        # Top1→Top2-4流し (6点)
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
        # Top1→Top2-6流し (20点)
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
    
    elif bet_type == 'sanrentan_box3':
        # Top1-3 BOX (6点)
        if len(horses) < 3:
            return 0, 0, 0
        top3 = horses[:3]
        tickets = list(permutations(top3, 3))
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
# グリッドサーチ実行
# ============================================================

def run_grid_search(df, payout_map):
    """メイングリッドサーチ"""
    
    ranking_methods = {
        'score': lambda g: rank_by_score(g),
        'ev': lambda g: rank_by_ev(g),
        'score_then_ev_3': lambda g: rank_score_then_ev(g, top_n=3),
        'score_then_ev_5': lambda g: rank_score_then_ev(g, top_n=5),
        'weighted_0.3': lambda g: rank_ev_weighted(g, alpha=0.3),
        'weighted_0.5': lambda g: rank_ev_weighted(g, alpha=0.5),
        'weighted_0.7': lambda g: rank_ev_weighted(g, alpha=0.7),
        'boost_ev_1.2': lambda g: rank_boost_high_ev(g, ev_threshold=1.2),
        'boost_ev_1.5': lambda g: rank_boost_high_ev(g, ev_threshold=1.5),
        'exclude_pop1': lambda g: rank_exclude_pop1(g),
        # 新規: グループ別並び替え
        'group_ev': lambda g: rank_group_ev(g),  # Top3 + 4-6 それぞれEV
        'group_ev_top3': lambda g: rank_group_ev_top3_only(g),  # Top3のみEV
        'group_ev_456': lambda g: rank_group_ev_456_only(g),  # 4-6のみEV
        'group_top1fix': lambda g: rank_group_ev_top1_fix(g),  # Top1固定、2-3 + 4-6 EV
        'group_top2fix': lambda g: rank_group_ev_top2_fix(g),  # Top2固定、3-6 EV
    }
    
    betting_strategies = ['tansho', 'umaren_3', 'umaren_5', 'sanrenpuku', 'sanrentan_6', 'sanrentan_20', 'sanrentan_box3']
    
    results = []
    
    # 全レースを処理
    race_groups = {rid: grp for rid, grp in df.groupby('race_id') if len(grp) >= 6}
    logger.info(f"Processing {len(race_groups)} races...")
    
    for rank_name, rank_func in ranking_methods.items():
        for bet_type in betting_strategies:
            stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
            
            for rid, grp in race_groups.items():
                # ランキング適用
                sorted_grp = rank_func(grp.copy())
                horses = sorted_grp['horse_number'].astype(int).tolist()
                
                # ベットシミュレーション
                cost, ret, hit = simulate_bet(horses, payout_map, rid, bet_type)
                if cost > 0:
                    stats['races'] += 1
                    stats['cost'] += cost
                    stats['return'] += ret
                    stats['hits'] += hit
            
            if stats['races'] >= 50 and stats['cost'] > 0:
                roi = stats['return'] / stats['cost'] * 100
                hit_rate = stats['hits'] / stats['races'] * 100
                results.append({
                    'ranking': rank_name,
                    'betting': bet_type,
                    'races': stats['races'],
                    'roi': roi,
                    'hit_rate': hit_rate,
                    'cost': stats['cost'],
                    'return': stats['return']
                })
    
    return results

def main():
    print("\n" + "#"*80)
    print("# 🎯 ランキング方法 × 馬券戦略 グリッドサーチ (2024+2025年)")
    print("# 仮説: スコアが近い馬の中ではEVで並び替えたほうが回収率向上")
    print("#"*80)
    
    years = [2024, 2025]
    
    # 1. データロード
    df = load_data(years)
    df = load_model_and_predict(df)
    
    # 2. 払戻データ
    pay_df = load_payouts(years)
    payout_map = build_payout_map(pay_df)
    logger.info(f"Loaded payouts for {len(payout_map)} races")
    
    # 3. グリッドサーチ
    results = run_grid_search(df, payout_map)
    
    # 4. 結果表示
    results = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'='*90}")
    print("📊 グリッドサーチ結果 (ROI上位30)")
    print(f"{'='*90}")
    print(f"{'ランキング':<20} | {'馬券':<15} | {'Races':>6} | {'ROI':>8} | {'的中率':>7}")
    print("-" * 80)
    
    for r in results[:30]:
        print(f"{r['ranking']:<20} | {r['betting']:<15} | {r['races']:>6} | {r['roi']:>7.1f}% | {r['hit_rate']:>6.1f}%")
    
    # ROI 100%以上
    over_100 = [r for r in results if r['roi'] >= 100]
    
    print(f"\n{'='*90}")
    print(f"🏆 ROI 100%以上の組み合わせ: {len(over_100)}件")
    print(f"{'='*90}")
    
    for r in over_100:
        print(f"  {r['ranking']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース, Hit {r['hit_rate']:.1f}%")
    
    # ランキング方法別ベスト
    print(f"\n{'='*90}")
    print("📈 ランキング方法別 ベストROI")
    print(f"{'='*90}")
    
    ranking_best = {}
    for r in results:
        if r['ranking'] not in ranking_best or r['roi'] > ranking_best[r['ranking']]['roi']:
            ranking_best[r['ranking']] = r
    
    for rank_name, r in sorted(ranking_best.items(), key=lambda x: x[1]['roi'], reverse=True):
        print(f"  {rank_name}: {r['betting']} → ROI {r['roi']:.1f}%")
    
    # 馬券戦略別ベスト
    print(f"\n{'='*90}")
    print("🎫 馬券戦略別 ベストランキング")
    print(f"{'='*90}")
    
    betting_best = {}
    for r in results:
        if r['betting'] not in betting_best or r['roi'] > betting_best[r['betting']]['roi']:
            betting_best[r['betting']] = r
    
    for bet_name, r in sorted(betting_best.items(), key=lambda x: x[1]['roi'], reverse=True):
        print(f"  {bet_name}: {r['ranking']} → ROI {r['roi']:.1f}%")
    
    # 現行(score)との比較
    print(f"\n{'='*90}")
    print("🔍 現行(score順)との比較")
    print(f"{'='*90}")
    
    score_results = {r['betting']: r for r in results if r['ranking'] == 'score'}
    
    for bet_type in ['tansho', 'umaren_3', 'sanrenpuku', 'sanrentan_6', 'sanrentan_20']:
        if bet_type not in score_results:
            continue
        baseline = score_results[bet_type]['roi']
        
        best_for_bet = [r for r in results if r['betting'] == bet_type]
        if not best_for_bet:
            continue
        best = max(best_for_bet, key=lambda x: x['roi'])
        
        diff = best['roi'] - baseline
        sign = "+" if diff > 0 else ""
        print(f"  {bet_type}: score={baseline:.1f}% → {best['ranking']}={best['roi']:.1f}% ({sign}{diff:.1f}%)")
    
    # ファイル保存
    os.makedirs('reports', exist_ok=True)
    with open('reports/ranking_betting_grid_search.txt', 'w', encoding='utf-8') as f:
        f.write("=== ランキング方法 × 馬券戦略 グリッドサーチ (2024+2025年) ===\n\n")
        
        f.write("--- ROI上位30 ---\n")
        for r in results[:30]:
            f.write(f"{r['ranking']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース, Hit {r['hit_rate']:.1f}%\n")
        
        f.write(f"\n--- ROI 100%以上: {len(over_100)}件 ---\n")
        for r in over_100:
            f.write(f"{r['ranking']} × {r['betting']}: ROI {r['roi']:.1f}%, {r['races']}レース\n")
        
        f.write("\n--- ランキング方法別ベスト ---\n")
        for rank_name, r in sorted(ranking_best.items(), key=lambda x: x[1]['roi'], reverse=True):
            f.write(f"{rank_name}: {r['betting']} → ROI {r['roi']:.1f}%\n")
        
        f.write("\n--- 現行(score)との比較 ---\n")
        for bet_type in ['tansho', 'umaren_3', 'sanrenpuku', 'sanrentan_6', 'sanrentan_20']:
            if bet_type not in score_results:
                continue
            baseline = score_results[bet_type]['roi']
            best_for_bet = [r for r in results if r['betting'] == bet_type]
            if not best_for_bet:
                continue
            best = max(best_for_bet, key=lambda x: x['roi'])
            diff = best['roi'] - baseline
            sign = "+" if diff > 0 else ""
            f.write(f"{bet_type}: score={baseline:.1f}% → {best['ranking']}={best['roi']:.1f}% ({sign}{diff:.1f}%)\n")
    
    print("\n結果を reports/ranking_betting_grid_search.txt に保存しました")
    print("\n✅ グリッドサーチ完了!")

if __name__ == "__main__":
    main()

"""
スコア分布パターン別グリッドサーチ
- Top1-6のスコアが均衡 vs 上位/下位が離れている
- 仮説: 離れている場合は単勝や絞った買い目が有効
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

def get_race_data_with_distribution(df):
    """レースごとのスコア分布を計算"""
    race_data = {}
    
    for rid, grp in df.groupby('race_id'):
        sorted_g = grp.sort_values('score', ascending=False)
        if len(sorted_g) < 6:
            continue
        
        top6 = sorted_g.head(6)
        scores = top6['score'].values
        
        # スコア分布指標
        score_std = np.std(scores)  # 標準偏差
        score_range = scores[0] - scores[5]  # Top1 - Top6の差
        top3_gap = scores[0] - scores[2]  # Top1 - Top3の差
        bottom_gap = scores[2] - scores[5]  # Top3 - Top6の差
        
        # 上位集中度 (Top3のスコア合計 / Top6のスコア合計)
        top3_concentration = scores[:3].sum() / scores.sum() if scores.sum() > 0 else 0.5
        
        top1 = sorted_g.iloc[0]
        
        race_data[rid] = {
            'score_std': score_std,
            'score_range': score_range,
            'top3_gap': top3_gap,
            'bottom_gap': bottom_gap,
            'top3_concentration': top3_concentration,
            'top1_odds': top1['odds'] if not pd.isna(top1['odds']) else 0,
            'top1_rank': top1['rank'],
            'top1_popularity': int(top1['popularity']) if not pd.isna(top1['popularity']) else 99,
            'horses': sorted_g['horse_number'].astype(int).tolist()
        }
    
    return race_data

def simulate_bet(rd, payout_map, rid, bet_type, opp_count=5):
    """馬券シミュレーション"""
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
        tickets = [t for t in tickets if axis in t]
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
    
    # 絞った買い目 (Top3のみ)
    elif bet_type == 'sanrentan_top3':
        if len(h_nums) < 3:
            return 0, 0, 0
        axis = h_nums[0]
        opps = h_nums[1:3]
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
    
    elif bet_type == 'umaren_top3':
        if len(h_nums) < 3:
            return 0, 0, 0
        axis = h_nums[0]
        opps = h_nums[1:3]
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
    
    return 0, 0, 0

def run_distribution_grid_search(race_data, payout_map):
    """スコア分布パターン別グリッドサーチ"""
    
    # 分布カテゴリ
    def dist_category(rd):
        """スコア分布をカテゴライズ"""
        range_val = rd['score_range']
        
        # Top1-6の差で分類
        if range_val >= 0.5:
            return '大差 (range≥0.5)'
        elif range_val >= 0.3:
            return '中差 (0.3-0.5)'
        elif range_val >= 0.15:
            return '小差 (0.15-0.3)'
        else:
            return '均衡 (range<0.15)'
    
    def top3_vs_bottom_category(rd):
        """上位3頭と下位3頭の差をカテゴライズ"""
        top3_gap = rd['top3_gap']
        bottom_gap = rd['bottom_gap']
        
        if top3_gap >= 0.3 and bottom_gap < 0.1:
            return 'Top3優勢・下位団子'
        elif top3_gap < 0.1 and bottom_gap >= 0.2:
            return 'Top3団子・下位離散'
        elif top3_gap >= 0.2 and bottom_gap >= 0.2:
            return '全体離散'
        else:
            return '全体均衡'
    
    bet_types = ['tansho', 'umaren', 'umaren_top3', 'wide', 'sanrenpuku', 'sanrentan', 'sanrentan_top3']
    
    results = []
    
    # パターン1: score_range別
    for bet_type in bet_types:
        for dist_cat in ['大差 (range≥0.5)', '中差 (0.3-0.5)', '小差 (0.15-0.3)', '均衡 (range<0.15)']:
            stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
            
            for rid, rd in race_data.items():
                if rid not in payout_map:
                    continue
                if dist_category(rd) != dist_cat:
                    continue
                
                cost, ret, hit = simulate_bet(rd, payout_map, rid, bet_type)
                if cost > 0:
                    stats['races'] += 1
                    stats['cost'] += cost
                    stats['return'] += ret
                    stats['hits'] += hit
            
            if stats['races'] >= 30 and stats['cost'] > 0:
                roi = stats['return'] / stats['cost'] * 100
                results.append({
                    'bet_type': bet_type,
                    'condition': dist_cat,
                    'category': 'score_range',
                    'races': stats['races'],
                    'roi': roi,
                    'hit_rate': stats['hits'] / stats['races'] * 100
                })
    
    # パターン2: Top3 vs Bottom3 パターン別
    for bet_type in bet_types:
        for pattern in ['Top3優勢・下位団子', 'Top3団子・下位離散', '全体離散', '全体均衡']:
            stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
            
            for rid, rd in race_data.items():
                if rid not in payout_map:
                    continue
                if top3_vs_bottom_category(rd) != pattern:
                    continue
                
                cost, ret, hit = simulate_bet(rd, payout_map, rid, bet_type)
                if cost > 0:
                    stats['races'] += 1
                    stats['cost'] += cost
                    stats['return'] += ret
                    stats['hits'] += hit
            
            if stats['races'] >= 30 and stats['cost'] > 0:
                roi = stats['return'] / stats['cost'] * 100
                results.append({
                    'bet_type': bet_type,
                    'condition': pattern,
                    'category': 'top3_vs_bottom',
                    'races': stats['races'],
                    'roi': roi,
                    'hit_rate': stats['hits'] / stats['races'] * 100
                })
    
    return results

def main():
    print("\n" + "#"*80)
    print("# 📊 スコア分布パターン別 グリッドサーチ (2024+2025年)")
    print("# 仮説: 上位/下位が離れている場合は単勝や絞った買い目が有効")
    print("#"*80)
    
    years = [2024, 2025]
    
    df = load_data(years)
    df = load_model_and_predict(df)
    
    pay_df = load_payouts(years)
    payout_map = build_payout_map(pay_df)
    
    race_data = get_race_data_with_distribution(df)
    logger.info(f"Prepared data for {len(race_data)} races")
    
    # 分布の統計を表示
    ranges = [rd['score_range'] for rd in race_data.values()]
    print(f"\n📈 スコア分布統計:")
    print(f"   Top1-6差: 平均 {np.mean(ranges):.3f}, 中央値 {np.median(ranges):.3f}")
    print(f"   最小 {np.min(ranges):.3f}, 最大 {np.max(ranges):.3f}")
    
    results = run_distribution_grid_search(race_data, payout_map)
    
    # 結果表示
    results = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'='*80}")
    print("📊 スコア分布パターン別 グリッドサーチ結果 (ROI順)")
    print(f"{'='*80}")
    print(f"{'券種':<16} | {'条件':<20} | {'Races':>6} | {'ROI':>8} | {'的中率':>7}")
    print("-" * 75)
    
    for r in results[:30]:
        print(f"{r['bet_type']:<16} | {r['condition']:<20} | {r['races']:>6} | {r['roi']:>7.1f}% | {r['hit_rate']:>6.1f}%")
    
    # ROI 100%以上
    over_100 = [r for r in results if r['roi'] >= 100]
    
    print(f"\n{'='*80}")
    print(f"🏆 ROI 100%以上の戦略: {len(over_100)}件")
    print(f"{'='*80}")
    
    for r in over_100:
        print(f"  {r['bet_type']} x {r['condition']}: ROI {r['roi']:.1f}%, {r['races']}レース")
    
    # 仮説検証
    print(f"\n{'='*80}")
    print("🔍 仮説検証: 「上位/下位が離れている場合は絞った買い目が有効」")
    print(f"{'='*80}")
    
    # 大差のレースでの比較
    large_gap = [r for r in results if '大差' in r['condition'] or 'Top3優勢' in r['condition']]
    large_gap = sorted(large_gap, key=lambda x: x['roi'], reverse=True)
    
    print("\n【大差/Top3優勢レースでのベスト券種】")
    for r in large_gap[:10]:
        print(f"  {r['bet_type']}: ROI {r['roi']:.1f}%, {r['condition']}")
    
    # 均衡レースでの比較
    balanced = [r for r in results if '均衡' in r['condition']]
    balanced = sorted(balanced, key=lambda x: x['roi'], reverse=True)
    
    print("\n【均衡レースでのベスト券種】")
    for r in balanced[:10]:
        print(f"  {r['bet_type']}: ROI {r['roi']:.1f}%, {r['condition']}")
    
    # ファイル保存
    with open('reports/score_distribution_grid_search.txt', 'w', encoding='utf-8') as f:
        f.write("=== スコア分布パターン別 グリッドサーチ (2024+2025年) ===\n\n")
        
        f.write("--- ROI上位30 ---\n")
        for r in results[:30]:
            f.write(f"{r['bet_type']} x {r['condition']}: ROI {r['roi']:.1f}%, {r['races']}レース, Hit {r['hit_rate']:.1f}%\n")
        
        f.write(f"\n--- ROI 100%以上: {len(over_100)}件 ---\n")
        for r in over_100:
            f.write(f"{r['bet_type']} x {r['condition']}: ROI {r['roi']:.1f}%, {r['races']}レース\n")
        
        f.write("\n--- 仮説検証 ---\n")
        f.write("【大差/Top3優勢レースでのベスト】\n")
        for r in large_gap[:5]:
            f.write(f"  {r['bet_type']}: ROI {r['roi']:.1f}%\n")
        f.write("【均衡レースでのベスト】\n")
        for r in balanced[:5]:
            f.write(f"  {r['bet_type']}: ROI {r['roi']:.1f}%\n")
    
    print("\n結果を reports/score_distribution_grid_search.txt に保存しました")
    print("\n✅ グリッドサーチ完了!")

if __name__ == "__main__":
    main()

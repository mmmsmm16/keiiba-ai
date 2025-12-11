"""
V13 堅実戦略グリッドサーチ
- Option C対象外レース（Top1が1-6番人気）のみを対象
- 2024年でパターン発見 → 2025年でバックテスト予定
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

# v7モデルを使用 (TabNetなしでfeature mismatchを回避)
MODEL_PATH = 'experiments/v7_ensemble_full/models/lgbm.pkl'
DATASET_PATH = 'experiments/v7_ensemble_full/data/lgbm_datasets.pkl'

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_predictions_from_db(year):
    data_path = 'data/processed/preprocessed_data.parquet'
    df = pd.read_parquet(data_path)
    df = df[df['year'] == year].copy()
    
    # JRA Only
    if 'venue' in df.columns or 'race_id' in df.columns:
        jra_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        df['venue_code'] = df['race_id'].astype(str).str[4:6]
        df = df[df['venue_code'].isin(jra_codes)].copy()
    
    logger.info(f"Loaded {len(df)} rows for year {year} (JRA only)")
    return df

def load_model_and_predict(df):
    # Fix import path
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from model.lgbm import KeibaLGBM
    
    model = KeibaLGBM()
    model.load_model(MODEL_PATH)
    
    import pickle
    with open(DATASET_PATH, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    # 欠損カラムは0埋め
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    
    X = df[feature_cols]
    scores = model.predict(X)
    df['score'] = scores
    return df

def load_payouts(year):
    engine = get_db_engine()
    query = text(f"SELECT * FROM jvd_hr WHERE kaisai_nen = '{year}'")
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
    
    # スコア差計算
    def get_score_gap(grp):
        sorted_scores = grp.sort_values('score', ascending=False)
        if len(sorted_scores) >= 6:
            return sorted_scores.iloc[0]['score'] - sorted_scores.iloc[5]['score']
        elif len(sorted_scores) >= 2:
            return sorted_scores.iloc[0]['score'] - sorted_scores.iloc[-1]['score']
        return 0
    
    score_gaps = df.groupby('race_id').apply(get_score_gap).reset_index()
    score_gaps.columns = ['race_id', 'score_gap']
    df = df.merge(score_gaps, on='race_id', how='left')
    
    return df

def get_race_data(df, filter_non_option_c=True):
    """レースごとのデータを取得（Option C対象外のみ）"""
    race_data = {}
    
    for rid, grp in df.groupby('race_id'):
        sorted_g = grp.sort_values('score', ascending=False)
        if len(sorted_g) < 6:
            continue
        
        top1 = sorted_g.iloc[0]
        
        pop = top1.get('popularity', 99)
        if pd.isna(pop): pop = 99
        
        score_gap = sorted_g.iloc[0].get('score_gap', 0)
        if pd.isna(score_gap): score_gap = 0
        
        # Option C条件: pop >= 7 OR score_gap < 0.3 → 三連単対象
        # Option C対象外: pop < 7 AND score_gap >= 0.3 → 今回のターゲット
        is_option_c_target = (pop >= 7) or (score_gap < 0.3)
        
        if filter_non_option_c and is_option_c_target:
            continue  # Option C対象レースはスキップ
        
        race_data[rid] = {
            'top1_popularity': int(pop),
            'score_gap': score_gap,
            'top1_odds': top1['odds'] if not pd.isna(top1['odds']) else 0,
            'top1_rank': top1['rank'],
            'top1_prob': top1.get('prob', 0),
            'horses': sorted_g['horse_number'].astype(int).tolist(),
            'scores': sorted_g['score'].tolist(),
            'odds_list': sorted_g['odds'].fillna(0).tolist()
        }
    
    return race_data

def simulate_bet(race_data, payout_map, rid, bet_type, n_opps=3):
    """特定の馬券を賭けた場合のコストとリターンを計算"""
    rd = race_data[rid]
    h_nums = rd['horses']
    
    if bet_type == 'tansho':
        cost = 100
        axis = h_nums[0]
        key = f"{axis:02}"
        ret = payout_map.get(rid, {}).get('tansho', {}).get(key, 0)
        hit = 1 if ret > 0 else 0
        return cost, ret, hit
    
    elif bet_type == 'umaren':
        if len(h_nums) < n_opps + 1:
            return 0, 0, 0
        axis = h_nums[0]
        opps = h_nums[1:n_opps+1]
        cost = len(opps) * 100
        ret = 0
        hit = 0
        for opp in opps:
            c_sorted = sorted([axis, opp])
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in payout_map.get(rid, {}).get('umaren', {}):
                ret += payout_map[rid]['umaren'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'umaren_box':
        if len(h_nums) < n_opps:
            return 0, 0, 0
        top_n = h_nums[:n_opps]
        combos = list(combinations(top_n, 2))
        cost = len(combos) * 100
        ret = 0
        hit = 0
        for c in combos:
            c_sorted = sorted(c)
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in payout_map.get(rid, {}).get('umaren', {}):
                ret += payout_map[rid]['umaren'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'wide':
        if len(h_nums) < n_opps + 1:
            return 0, 0, 0
        axis = h_nums[0]
        opps = h_nums[1:n_opps+1]
        cost = len(opps) * 100
        ret = 0
        hit = 0
        for opp in opps:
            c_sorted = sorted([axis, opp])
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in payout_map.get(rid, {}).get('wide', {}):
                ret += payout_map[rid]['wide'][key]
                hit = 1
        return cost, ret, hit
    
    elif bet_type == 'wide_box':
        if len(h_nums) < n_opps:
            return 0, 0, 0
        top_n = h_nums[:n_opps]
        combos = list(combinations(top_n, 2))
        cost = len(combos) * 100
        ret = 0
        hit = 0
        for c in combos:
            c_sorted = sorted(c)
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in payout_map.get(rid, {}).get('wide', {}):
                ret += payout_map[rid]['wide'][key]
                hit = 1
        return cost, ret, hit
    
    return 0, 0, 0

def run_grid_search(race_data, payout_map):
    """堅実戦略のグリッドサーチ"""
    results = []
    
    # 馬券種類と点数のバリエーション
    bet_configs = [
        ('tansho', None),
        ('umaren', 2),
        ('umaren', 3),
        ('umaren', 4),
        ('umaren_box', 3),
        ('umaren_box', 4),
        ('wide', 2),
        ('wide', 3),
        ('wide_box', 3),
        ('wide_box', 4),
    ]
    
    # 人気フィルター（Option C対象外なので既に pop < 7）
    pop_filters = [
        ('全体', lambda p: True),
        ('1番人気', lambda p: p == 1),
        ('2-3番人気', lambda p: 2 <= p <= 3),
        ('4-6番人気', lambda p: 4 <= p <= 6),
        ('1-2番人気', lambda p: 1 <= p <= 2),
        ('1-3番人気', lambda p: 1 <= p <= 3),
    ]
    
    # スコア差フィルター
    gap_filters = [
        ('全gap', lambda g: True),
        ('gap≥0.4', lambda g: g >= 0.4),
        ('gap≥0.5', lambda g: g >= 0.5),
        ('gap≥0.6', lambda g: g >= 0.6),
        ('gap 0.3-0.5', lambda g: 0.3 <= g < 0.5),
    ]
    
    # Top1のオッズフィルター
    odds_filters = [
        ('全odds', lambda o: True),
        ('odds≤3.0', lambda o: o <= 3.0),
        ('odds≤5.0', lambda o: o <= 5.0),
        ('odds>3.0', lambda o: o > 3.0),
    ]
    
    for bet_type, n_opps in bet_configs:
        for pop_name, pop_fn in pop_filters:
            for gap_name, gap_fn in gap_filters:
                for odds_name, odds_fn in odds_filters:
                    stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
                    
                    for rid, rd in race_data.items():
                        if rid not in payout_map:
                            continue
                        
                        # フィルター適用
                        if not pop_fn(rd['top1_popularity']):
                            continue
                        if not gap_fn(rd['score_gap']):
                            continue
                        if not odds_fn(rd['top1_odds']):
                            continue
                        
                        cost, ret, hit = simulate_bet(race_data, payout_map, rid, bet_type, n_opps or 3)
                        
                        if cost > 0:
                            stats['races'] += 1
                            stats['cost'] += cost
                            stats['return'] += ret
                            stats['hits'] += hit
                    
                    if stats['races'] >= 20 and stats['cost'] > 0:
                        roi = stats['return'] / stats['cost'] * 100
                        hit_rate = stats['hits'] / stats['races'] * 100
                        
                        bet_label = f"{bet_type}_{n_opps}" if n_opps else bet_type
                        condition = f"{pop_name} x {gap_name} x {odds_name}"
                        
                        results.append({
                            'bet_type': bet_label,
                            'condition': condition,
                            'races': stats['races'],
                            'cost': stats['cost'],
                            'return': stats['return'],
                            'roi': roi,
                            'hit_rate': hit_rate,
                            'profit': stats['return'] - stats['cost']
                        })
    
    return results

def main():
    print("\n" + "#"*80)
    print("# 📊 V13 堅実戦略グリッドサーチ (2024年)")
    print("# 対象: Option C対象外レース (pop < 7 AND gap >= 0.3)")
    print("#"*80)
    
    year = 2024
    
    print("\n1. データ読み込み...")
    df = load_predictions_from_db(year)
    
    print("2. モデル予測...")
    df = load_model_and_predict(df)
    
    print("3. 前処理...")
    df = preprocess_data(df)
    
    print("4. 払戻データ読み込み...")
    pay_df = load_payouts(year)
    payout_map = build_payout_map(pay_df)
    logger.info(f"Built payout map for {len(payout_map)} races")
    
    print("5. レースデータ抽出（Option C対象外のみ）...")
    race_data = get_race_data(df, filter_non_option_c=True)
    logger.info(f"Option C対象外レース: {len(race_data)} races")
    
    # 参考: Option C対象レース数も表示
    race_data_all = get_race_data(df, filter_non_option_c=False)
    option_c_count = len(race_data_all) - len(race_data)
    logger.info(f"Option C対象レース: {option_c_count} races")
    
    print("\n6. グリッドサーチ実行...")
    results = run_grid_search(race_data, payout_map)
    
    # 結果をROI順にソート
    results = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    # 結果表示
    print(f"\n{'='*100}")
    print("📊 グリッドサーチ結果 (ROI上位30)")
    print(f"{'='*100}")
    print(f"{'馬券種類':<15} | {'条件':<40} | {'Races':>6} | {'ROI':>8} | {'的中率':>7} | {'利益':>10}")
    print("-" * 100)
    
    for r in results[:30]:
        print(f"{r['bet_type']:<15} | {r['condition']:<40} | {r['races']:>6} | {r['roi']:>7.1f}% | {r['hit_rate']:>6.1f}% | {r['profit']:>+10.0f}")
    
    # ROI 90%以上 & 的中率25%以上
    high_quality = [r for r in results if r['roi'] >= 90 and r['hit_rate'] >= 25]
    
    print(f"\n{'='*100}")
    print(f"🏆 堅実戦略候補 (ROI≥90% & 的中率≥25%): {len(high_quality)}件")
    print(f"{'='*100}")
    
    for r in high_quality[:20]:
        print(f"  {r['bet_type']} x {r['condition']}")
        print(f"    → ROI {r['roi']:.1f}%, 的中率 {r['hit_rate']:.1f}%, {r['races']}レース, 利益 {r['profit']:+,.0f}円")
        print()
    
    # ROI 100%以上
    over_100 = [r for r in results if r['roi'] >= 100]
    
    print(f"\n{'='*100}")
    print(f"💰 ROI 100%以上: {len(over_100)}件")
    print(f"{'='*100}")
    
    for r in over_100[:10]:
        print(f"  {r['bet_type']} x {r['condition']}: ROI {r['roi']:.1f}%, {r['races']}レース")
    
    # ファイル保存
    os.makedirs('reports', exist_ok=True)
    with open('reports/v13_stable_grid_search_2024.txt', 'w', encoding='utf-8') as f:
        f.write("=== V13 堅実戦略グリッドサーチ (2024年) ===\n")
        f.write(f"対象: Option C対象外レース (pop < 7 AND gap >= 0.3)\n")
        f.write(f"レース数: {len(race_data)}\n\n")
        
        f.write("--- ROI上位30 ---\n")
        for r in results[:30]:
            f.write(f"{r['bet_type']} x {r['condition']}: ROI {r['roi']:.1f}%, Hit {r['hit_rate']:.1f}%, {r['races']}races, Profit {r['profit']:+,.0f}円\n")
        
        f.write(f"\n--- 堅実戦略候補 (ROI≥90% & 的中率≥25%) ---\n")
        for r in high_quality[:20]:
            f.write(f"{r['bet_type']} x {r['condition']}: ROI {r['roi']:.1f}%, Hit {r['hit_rate']:.1f}%, {r['races']}races\n")
        
        f.write(f"\n--- ROI 100%以上: {len(over_100)}件 ---\n")
        for r in over_100:
            f.write(f"{r['bet_type']} x {r['condition']}: ROI {r['roi']:.1f}%, {r['races']}races\n")
    
    print("\n結果を reports/v13_stable_grid_search_2024.txt に保存しました")
    print("\n✅ グリッドサーチ完了!")

if __name__ == "__main__":
    main()

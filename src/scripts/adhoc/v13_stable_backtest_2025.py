"""
V13 堅実戦略バックテスト（2025年）
- 2024年で発見した有望戦略を2025年データで検証
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

# v7モデルを使用
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
    jra_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    df['venue_code'] = df['race_id'].astype(str).str[4:6]
    df = df[df['venue_code'].isin(jra_codes)].copy()
    
    logger.info(f"Loaded {len(df)} rows for year {year} (JRA only)")
    return df

def load_model_and_predict(df):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from model.lgbm import KeibaLGBM
    
    model = KeibaLGBM()
    model.load_model(MODEL_PATH)
    
    import pickle
    with open(DATASET_PATH, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
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
    """レースごとのデータを取得"""
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
        
        is_option_c_target = (pop >= 7) or (score_gap < 0.3)
        
        if filter_non_option_c and is_option_c_target:
            continue
        
        race_data[rid] = {
            'top1_popularity': int(pop),
            'score_gap': score_gap,
            'top1_odds': top1['odds'] if not pd.isna(top1['odds']) else 0,
            'horses': sorted_g['horse_number'].astype(int).tolist(),
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
    
    return 0, 0, 0

def backtest_strategy(race_data, payout_map, strategy):
    """特定の戦略をバックテスト"""
    stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
    
    for rid, rd in race_data.items():
        if rid not in payout_map:
            continue
        
        # 条件チェック
        if not strategy['pop_fn'](rd['top1_popularity']):
            continue
        if not strategy['gap_fn'](rd['score_gap']):
            continue
        if not strategy['odds_fn'](rd['top1_odds']):
            continue
        
        cost, ret, hit = simulate_bet(race_data, payout_map, rid, 
                                       strategy['bet_type'], strategy.get('n_opps', 3))
        
        if cost > 0:
            stats['races'] += 1
            stats['cost'] += cost
            stats['return'] += ret
            stats['hits'] += hit
    
    if stats['cost'] > 0:
        roi = stats['return'] / stats['cost'] * 100
        hit_rate = stats['hits'] / stats['races'] * 100 if stats['races'] > 0 else 0
        profit = stats['return'] - stats['cost']
    else:
        roi = 0
        hit_rate = 0
        profit = 0
    
    return {**stats, 'roi': roi, 'hit_rate': hit_rate, 'profit': profit}

def main():
    print("\n" + "#"*80)
    print("# 📊 V13 堅実戦略バックテスト (2025年)")
    print("# 2024年で発見した戦略を2025年で検証")
    print("#"*80)
    
    year = 2025
    
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
    
    # 2024年で発見した有望戦略
    strategies = [
        {
            'name': '戦略A: 馬連Box4 × 4-6番人気 × gap≥0.5',
            'bet_type': 'umaren_box',
            'n_opps': 4,
            'pop_fn': lambda p: 4 <= p <= 6,
            'gap_fn': lambda g: g >= 0.5,
            'odds_fn': lambda o: True,
            '2024_roi': 119.9,
            '2024_hit_rate': 30.4,
        },
        {
            'name': '戦略B: 馬連Box3 × 1番人気 × gap 0.3-0.5 × odds≤3.0',
            'bet_type': 'umaren_box',
            'n_opps': 3,
            'pop_fn': lambda p: p == 1,
            'gap_fn': lambda g: 0.3 <= g < 0.5,
            'odds_fn': lambda o: o <= 3.0,
            '2024_roi': 146.7,
            '2024_hit_rate': 23.3,
        },
        {
            'name': '戦略C: 単勝 × 4-6番人気 × gap≥0.6',
            'bet_type': 'tansho',
            'n_opps': None,
            'pop_fn': lambda p: 4 <= p <= 6,
            'gap_fn': lambda g: g >= 0.6,
            'odds_fn': lambda o: True,
            '2024_roi': 133.4,
            '2024_hit_rate': 13.5,
        },
        {
            'name': '戦略D: 馬連2点流し × 4-6番人気 × gap≥0.5',
            'bet_type': 'umaren',
            'n_opps': 2,
            'pop_fn': lambda p: 4 <= p <= 6,
            'gap_fn': lambda g: g >= 0.5,
            'odds_fn': lambda o: True,
            '2024_roi': 119.5,
            '2024_hit_rate': 11.8,
        },
        {
            'name': '戦略E: 馬連3点流し × 2-3番人気 × gap≥0.6 × odds≤3.0',
            'bet_type': 'umaren',
            'n_opps': 3,
            'pop_fn': lambda p: 2 <= p <= 3,
            'gap_fn': lambda g: g >= 0.6,
            'odds_fn': lambda o: o <= 3.0,
            '2024_roi': 107.9,
            '2024_hit_rate': 32.7,
        },
    ]
    
    print("\n" + "="*80)
    print("📊 バックテスト結果 (2025年)")
    print("="*80)
    
    results = []
    for s in strategies:
        result = backtest_strategy(race_data, payout_map, s)
        result['name'] = s['name']
        result['2024_roi'] = s['2024_roi']
        result['2024_hit_rate'] = s['2024_hit_rate']
        results.append(result)
        
        print(f"\n{s['name']}")
        print(f"  2024年: ROI {s['2024_roi']:.1f}%, 的中率 {s['2024_hit_rate']:.1f}%")
        print(f"  2025年: ROI {result['roi']:.1f}%, 的中率 {result['hit_rate']:.1f}%, {result['races']}レース")
        
        if result['roi'] >= 100:
            print(f"  ✅ 2025年もROI 100%以上！ 利益 {result['profit']:+,.0f}円")
        elif result['roi'] >= 90:
            print(f"  ⚠️ ROI 90-100%: 利益 {result['profit']:+,.0f}円")
        else:
            print(f"  ❌ ROI 90%未満: 損失 {result['profit']:+,.0f}円")
    
    # サマリー
    print("\n" + "="*80)
    print("📋 サマリー")
    print("="*80)
    
    over_100 = [r for r in results if r['roi'] >= 100]
    over_90 = [r for r in results if 90 <= r['roi'] < 100]
    under_90 = [r for r in results if r['roi'] < 90]
    
    print(f"  ✅ ROI 100%以上: {len(over_100)}件")
    for r in over_100:
        print(f"     - {r['name']}: ROI {r['roi']:.1f}%")
    
    print(f"  ⚠️ ROI 90-100%: {len(over_90)}件")
    for r in over_90:
        print(f"     - {r['name']}: ROI {r['roi']:.1f}%")
    
    print(f"  ❌ ROI 90%未満: {len(under_90)}件")
    for r in under_90:
        print(f"     - {r['name']}: ROI {r['roi']:.1f}%")
    
    # ファイル保存
    os.makedirs('reports', exist_ok=True)
    with open('reports/v13_stable_backtest_2025.txt', 'w', encoding='utf-8') as f:
        f.write("=== V13 堅実戦略バックテスト (2025年) ===\n\n")
        for r in results:
            f.write(f"{r['name']}\n")
            f.write(f"  2024: ROI {r['2024_roi']:.1f}%, Hit {r['2024_hit_rate']:.1f}%\n")
            f.write(f"  2025: ROI {r['roi']:.1f}%, Hit {r['hit_rate']:.1f}%, {r['races']}races, Profit {r['profit']:+,.0f}円\n\n")
    
    print("\n結果を reports/v13_stable_backtest_2025.txt に保存しました")
    print("\n✅ バックテスト完了!")

if __name__ == "__main__":
    main()

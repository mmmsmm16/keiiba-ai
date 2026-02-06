"""
拡張グリッドサーチ
1. レース条件別 (芝 vs ダート, 距離, クラス)
2. 競馬場別
4. 人気順位別
5. AIスコア差別 (Top1 - Top2)
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

def load_model_and_predict(df, model_name='ensemble', version='v4_2025'):
    sys.path.append('src')
    from model.ensemble import EnsembleModel
    
    model = EnsembleModel()
    model.load_model(f'models/ensemble_{version}.pkl')
    
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

def preprocess_data(df):
    df = df.copy()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    df['prob'] = df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    df['ev'] = df['prob'] * df['odds'].fillna(0)
    
    # スコア差 (Top1 - Top2)
    df['score_max'] = df.groupby('race_id')['score'].transform('max')
    df['score_second'] = df.groupby('race_id')['score'].transform(lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else x.max())
    df['score_gap'] = df['score_max'] - df['score_second']
    
    return df

def get_race_features(df):
    """レースごとの特徴量を取得"""
    race_features = {}
    
    for rid, grp in df.groupby('race_id'):
        sorted_g = grp.sort_values('score', ascending=False)
        top1 = sorted_g.iloc[0] if len(sorted_g) > 0 else None
        top2 = sorted_g.iloc[1] if len(sorted_g) > 1 else None
        
        if top1 is None:
            continue
        
        # 馬場 (芝/ダート)
        surface = top1.get('surface', '')
        if pd.isna(surface): surface = ''
        
        # 距離カテゴリ
        dist = top1.get('distance', 0)
        if pd.isna(dist): dist = 0
        if dist <= 1400:
            dist_cat = 'sprint'
        elif dist <= 2000:
            dist_cat = 'mile'
        else:
            dist_cat = 'long'
        
        # 競馬場
        venue = str(rid)[4:6] if len(str(rid)) >= 6 else ''
        
        # Top1の人気
        pop = top1.get('popularity', 99)
        if pd.isna(pop): pop = 99
        
        # スコア差
        score_gap = top1.get('score_gap', 0)
        if pd.isna(score_gap): score_gap = 0
        
        race_features[rid] = {
            'surface': surface,
            'dist_cat': dist_cat,
            'venue': venue,
            'top1_popularity': int(pop),
            'score_gap': score_gap,
            'top1_odds': top1['odds'] if not pd.isna(top1['odds']) else 0,
            'top1_ev': top1['ev'],
            'top1_rank': top1['rank'],
            'top1_horse': int(top1['horse_number']),
            'horses': sorted_g['horse_number'].astype(int).tolist()
        }
    
    return race_features

def run_segment_analysis(race_features, payout_map, segment_name, segment_func):
    """セグメント別ROI分析"""
    segments = defaultdict(lambda: {'races': 0, 'cost': 0, 'return': 0, 'hits': 0})
    
    for rid, rf in race_features.items():
        if rid not in payout_map:
            continue
        
        seg = segment_func(rf)
        if seg is None:
            continue
        
        # 単勝でテスト
        odds = rf['top1_odds']
        actual_rank = rf['top1_rank']
        
        cost = 100
        ret = odds * 100 if actual_rank == 1 else 0
        
        segments[seg]['races'] += 1
        segments[seg]['cost'] += cost
        segments[seg]['return'] += ret
        segments[seg]['hits'] += 1 if actual_rank == 1 else 0
    
    return segments

def print_segment_results(title, segments, min_races=30):
    """セグメント結果表示"""
    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}")
    
    results = []
    for seg, s in segments.items():
        if s['races'] >= min_races:
            roi = s['return'] / s['cost'] * 100 if s['cost'] > 0 else 0
            hit_rate = s['hits'] / s['races'] * 100 if s['races'] > 0 else 0
            results.append({
                'segment': seg,
                'races': s['races'],
                'roi': roi,
                'hit_rate': hit_rate,
                'profit': s['return'] - s['cost']
            })
    
    results = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    print(f"{'セグメント':<20} | {'Races':>6} | {'ROI':>8} | {'的中率':>7} | {'利益':>10}")
    print("-" * 70)
    
    for r in results:
        profit_str = f"¥{r['profit']:+,.0f}"
        print(f"{str(r['segment']):<20} | {r['races']:>6} | {r['roi']:>7.1f}% | {r['hit_rate']:>6.1f}% | {profit_str:>10}")
    
    if results:
        best = results[0]
        print(f"\n🏆 ベスト: {best['segment']} (ROI {best['roi']:.1f}%, {best['races']}レース)")
    
    return results

def main():
    print("\n" + "#"*80)
    print("# 📊 拡張グリッドサーチ (2024+2025年)")
    print("# 切り口: 馬場/距離/競馬場/人気/スコア差")
    print("#"*80)
    
    years = [2024, 2025]
    
    df = load_predictions_from_db(years)
    df = load_model_and_predict(df, 'ensemble', 'v4_2025')
    df = preprocess_data(df)
    
    pay_df = load_payouts(years)
    payout_map = build_payout_map(pay_df)
    logger.info(f"Built payout map for {len(payout_map)} races")
    
    race_features = get_race_features(df)
    logger.info(f"Extracted features for {len(race_features)} races")
    
    all_results = {}
    
    # 1. 馬場別 (芝 vs ダート)
    segments = run_segment_analysis(
        race_features, payout_map, 
        "馬場別",
        lambda rf: rf['surface'] if rf['surface'] in ['芝', 'ダート', 'Turf', 'Dirt'] else None
    )
    all_results['surface'] = print_segment_results("馬場別 (単勝Top1)", segments)
    
    # 2. 距離別
    segments = run_segment_analysis(
        race_features, payout_map,
        "距離別",
        lambda rf: rf['dist_cat']
    )
    all_results['distance'] = print_segment_results("距離別 (単勝Top1)", segments)
    
    # 3. 競馬場別
    venue_names = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
        '05': '東京', '06': '中山', '07': '中京', '08': '京都',
        '09': '阪神', '10': '小倉'
    }
    segments = run_segment_analysis(
        race_features, payout_map,
        "競馬場別",
        lambda rf: venue_names.get(rf['venue'], rf['venue'])
    )
    all_results['venue'] = print_segment_results("競馬場別 (単勝Top1)", segments)
    
    # 4. Top1人気別
    def pop_category(rf):
        pop = rf['top1_popularity']
        if pop == 1: return '1番人気'
        elif pop <= 3: return '2-3番人気'
        elif pop <= 6: return '4-6番人気'
        elif pop <= 10: return '7-10番人気'
        else: return '11番人気以下'
    
    segments = run_segment_analysis(
        race_features, payout_map,
        "Top1人気別",
        pop_category
    )
    all_results['popularity'] = print_segment_results("Top1人気別 (単勝Top1)", segments)
    
    # 5. スコア差別
    def score_gap_category(rf):
        gap = rf['score_gap']
        if gap >= 0.5: return 'Gap≥0.5 (大差)'
        elif gap >= 0.3: return 'Gap 0.3-0.5'
        elif gap >= 0.1: return 'Gap 0.1-0.3'
        else: return 'Gap<0.1 (僅差)'
    
    segments = run_segment_analysis(
        race_features, payout_map,
        "スコア差別",
        score_gap_category
    )
    all_results['score_gap'] = print_segment_results("スコア差別 (単勝Top1)", segments)
    
    # 複合条件テスト (最も有望な組み合わせ)
    print("\n" + "="*70)
    print("📊 複合条件テスト (有望な組み合わせ)")
    print("="*70)
    
    combo_segments = defaultdict(lambda: {'races': 0, 'cost': 0, 'return': 0, 'hits': 0})
    
    for rid, rf in race_features.items():
        if rid not in payout_map:
            continue
        
        surface = rf['surface']
        gap_cat = score_gap_category(rf)
        pop = rf['top1_popularity']
        odds = rf['top1_odds']
        
        # 複合キー
        key = f"{surface}_{gap_cat}_Pop{pop if pop <= 3 else '4+'}"
        
        cost = 100
        ret = odds * 100 if rf['top1_rank'] == 1 else 0
        
        combo_segments[key]['races'] += 1
        combo_segments[key]['cost'] += cost
        combo_segments[key]['return'] += ret
        combo_segments[key]['hits'] += 1 if rf['top1_rank'] == 1 else 0
    
    all_results['combo'] = print_segment_results("複合条件 (単勝Top1)", combo_segments, min_races=20)
    
    # 結果をファイルに保存
    with open('reports/extended_grid_search.txt', 'w', encoding='utf-8') as f:
        f.write("=== 拡張グリッドサーチ結果 (2024+2025年) ===\n\n")
        
        for category, results in all_results.items():
            f.write(f"\n--- {category} ---\n")
            if results:
                for r in results[:10]:
                    f.write(f"{r['segment']}: ROI {r['roi']:.1f}%, {r['races']}レース, Hit {r['hit_rate']:.1f}%\n")
    
    print("\n結果を reports/extended_grid_search.txt に保存しました")
    print("\n✅ 拡張グリッドサーチ完了!")

if __name__ == "__main__":
    main()

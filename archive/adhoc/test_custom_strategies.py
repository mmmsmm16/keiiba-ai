"""
高速版 網羅的戦略探索スクリプト
- 複数年データ(2023+2024)で大きなサンプル数
- ベクトル化処理で高速化
- 的中率を必ず表示
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
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_predictions_from_db(years=[2023, 2024]):
    """DBから予測データをロード（要: 前処理済みデータ）"""
    data_path = 'data/processed/preprocessed_data.parquet'
    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        return None
    
    df = pd.read_parquet(data_path)
    df = df[df['year'].isin(years)].copy()
    logger.info(f"Loaded {len(df)} rows for years {years}")
    return df

def load_model_and_predict(df, model_name='ensemble', version='v4_2025'):
    """モデルをロードして予測"""
    sys.path.append('src')
    
    from model.ensemble import EnsembleModel
    from model.lgbm import KeibaLGBM
    
    model_dir = 'models'
    
    if model_name == 'ensemble':
        from model.ensemble import EnsembleModel
        model = EnsembleModel()
        path = os.path.join(model_dir, f'ensemble_{version}.pkl')
    else:
        model = KeibaLGBM()
        path = os.path.join(model_dir, f'lgbm_{version}.pkl')
    
    model.load_model(path)
    
    # 特徴量取得
    feature_cols = None
    try:
        if hasattr(model.model, 'feature_name'):
            feature_cols = model.model.feature_name()
        elif hasattr(model.model, 'feature_names_'):
            feature_cols = model.model.feature_names_
    except:
        pass
    
    if not feature_cols:
        import pickle
        with open('data/processed/lgbm_datasets.pkl', 'rb') as f:
            datasets = pickle.load(f)
        feature_cols = datasets['train']['X'].columns.tolist()
    
    # 欠損カラム補完
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    
    X = df[feature_cols]
    scores = model.predict(X)
    df['score'] = scores
    
    return df

def load_payouts(years=[2023, 2024]):
    """払戻データをロード"""
    engine = get_db_engine()
    years_str = ",".join([f"'{y}'" for y in years])
    query = text(f"SELECT * FROM jvd_hr WHERE kaisai_nen IN ({years_str})")
    
    df = pd.read_sql(query, engine)
    
    # race_id 構築
    df['race_id'] = (
        df['kaisai_nen'].astype(str) +
        df['keibajo_code'].astype(str) +
        df['kaisai_kai'].astype(str) +
        df['kaisai_nichime'].astype(str) +
        df['race_bango'].astype(str)
    )
    
    logger.info(f"Loaded {len(df)} payout records for years {years}")
    return df

def build_payout_map(pay_df):
    """払戻マップを高速構築"""
    payout_map = defaultdict(lambda: {'tansho': {}, 'umaren': {}, 'wide': {}, 'sanrenpuku': {}, 'sanrentan': {}})
    
    for _, row in pay_df.iterrows():
        rid = row['race_id']
        
        for prefix, max_count in [('haraimodoshi_tansho', 3), ('haraimodoshi_umaren', 3), 
                                   ('haraimodoshi_wide', 7), ('haraimodoshi_sanrenpuku', 3), 
                                   ('haraimodoshi_sanrentan', 6)]:
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
    """データ前処理"""
    df = df.copy()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    
    # レース内での予測順位
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    # Softmax確率
    df['prob'] = df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    
    # 期待値 (EV = prob * odds)
    df['ev'] = df['prob'] * df['odds'].fillna(0)
    
    return df

def fast_grid_search(df, payout_map, bet_type='tansho'):
    """高速グリッドサーチ"""
    start_time = time.time()
    
    # パラメータグリッド（条件を絞って高速化）
    min_probs = [0.0, 0.15, 0.20, 0.25]
    min_evs = [0.0, 1.0, 1.2, 1.5]
    odds_ranges = [
        (1.0, 3.0),    # 本命
        (3.0, 10.0),   # 中人気
        (10.0, 20.0),  # 中穴
        (20.0, 50.0),  # 大穴
        (10.0, 999),   # 穴馬全体
        (1.0, 999),    # 全体
    ]
    
    if bet_type in ['sanrentan', 'sanrenpuku', 'umaren']:
        opp_counts = [5, 6]
    else:
        opp_counts = [1]
    
    results = []
    
    # レースごとにTop1情報を事前計算
    race_top1 = df[df['pred_rank'] == 1].set_index('race_id')[['prob', 'ev', 'odds', 'rank', 'horse_number']].to_dict('index')
    
    # レースごとのTop N馬番リストを事前計算
    race_horses = {}
    for rid, grp in df.groupby('race_id'):
        sorted_g = grp.sort_values('score', ascending=False)
        race_horses[rid] = sorted_g['horse_number'].astype(int).tolist()
    
    total_combos = len(min_probs) * len(min_evs) * len(odds_ranges) * len(opp_counts)
    combo_count = 0
    
    for min_prob, min_ev, (min_odds, max_odds), opp_count in product(min_probs, min_evs, odds_ranges, opp_counts):
        combo_count += 1
        
        cost, ret, hits, races = 0, 0, 0, 0
        
        for rid, top1 in race_top1.items():
            if rid not in payout_map:
                continue
            
            prob = top1['prob']
            ev = top1['ev']
            odds = top1['odds'] if not pd.isna(top1['odds']) else 0
            actual_rank = top1['rank']
            
            # フィルター条件
            if prob < min_prob: continue
            if ev < min_ev: continue
            if odds < min_odds or odds > max_odds: continue
            
            if bet_type == 'tansho':
                # 単勝
                cost += 100
                races += 1
                if actual_rank == 1:
                    ret += odds * 100
                    hits += 1
                    
            elif bet_type == 'sanrentan':
                # 3連単ながし
                if rid not in race_horses: continue
                h_nums = race_horses[rid]
                if len(h_nums) < opp_count + 1: continue
                
                axis = h_nums[0]
                opps = h_nums[1:opp_count+1]
                
                tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
                race_cost = len(tickets) * 100
                race_ret = 0
                hit_flag = 0
                
                for t in tickets:
                    key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                    if key in payout_map[rid]['sanrentan']:
                        race_ret += payout_map[rid]['sanrentan'][key]
                        hit_flag = 1
                
                cost += race_cost
                ret += race_ret
                if hit_flag: hits += 1
                races += 1
                
            elif bet_type == 'umaren':
                # 馬連ながし
                if rid not in race_horses: continue
                h_nums = race_horses[rid]
                if len(h_nums) < opp_count + 1: continue
                
                axis = h_nums[0]
                opps = h_nums[1:opp_count+1]
                
                race_cost = len(opps) * 100
                race_ret = 0
                hit_flag = 0
                
                for opp in opps:
                    c_sorted = sorted([axis, opp])
                    key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
                    if key in payout_map[rid]['umaren']:
                        race_ret += payout_map[rid]['umaren'][key]
                        hit_flag = 1
                
                cost += race_cost
                ret += race_ret
                if hit_flag: hits += 1
                races += 1
                
            elif bet_type == 'sanrenpuku':
                # 3連複ながし
                if rid not in race_horses: continue
                h_nums = race_horses[rid]
                if len(h_nums) < 6: continue
                
                axis = h_nums[0]
                opps = h_nums[1:6]
                
                tickets = [(axis, o1, o2) for o1, o2 in combinations(opps, 2)]
                race_cost = len(tickets) * 100
                race_ret = 0
                hit_flag = 0
                
                for t in tickets:
                    c_sorted = sorted(t)
                    key = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
                    if key in payout_map[rid]['sanrenpuku']:
                        race_ret += payout_map[rid]['sanrenpuku'][key]
                        hit_flag = 1
                
                cost += race_cost
                ret += race_ret
                if hit_flag: hits += 1
                races += 1
        
        if races >= 30:  # 最低サンプル数
            roi = ret / cost * 100 if cost > 0 else 0
            hit_rate = hits / races * 100 if races > 0 else 0
            results.append({
                'min_prob': min_prob, 
                'min_ev': min_ev, 
                'min_odds': min_odds, 
                'max_odds': max_odds,
                'opp_count': opp_count if bet_type != 'tansho' else '-',
                'roi': roi, 
                'hit_rate': hit_rate, 
                'races': races,
                'profit': ret - cost
            })
    
    elapsed = time.time() - start_time
    logger.info(f"{bet_type} grid search completed in {elapsed:.1f}s ({total_combos} combinations)")
    
    return sorted(results, key=lambda x: x['roi'], reverse=True)

def print_results(title, results, top_n=15):
    """結果表示"""
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print(f"{'='*80}")
    
    if not results:
        print("⚠️ 条件を満たす結果がありませんでした")
        return
    
    # ヘッダー
    print(f"{'Prob':>5} | {'EV':>4} | {'Odds':>10} | {'Opps':>4} | {'ROI':>8} | {'的中率':>7} | {'Races':>6} | {'Profit':>10}")
    print("-" * 80)
    
    for r in results[:top_n]:
        odds_range = f"{r['min_odds']:.0f}-{r['max_odds']:.0f}"
        profit_str = f"¥{r['profit']:,.0f}"
        print(f"{r['min_prob']:>5.2f} | {r['min_ev']:>4.1f} | {odds_range:>10} | {str(r['opp_count']):>4} | {r['roi']:>7.1f}% | {r['hit_rate']:>6.1f}% | {r['races']:>6} | {profit_str:>10}")
    
    # ベスト
    if results:
        best = results[0]
        print(f"\n🏆 ベスト: ROI {best['roi']:.1f}%, 的中率 {best['hit_rate']:.1f}%, {best['races']}レース")

def find_over_100_strategies(all_results):
    """ROI 100%超え戦略をまとめる"""
    print("\n" + "="*80)
    print("🎯 ROI 100%超え戦略まとめ")
    print("="*80)
    
    over_100 = []
    for bet_type, results in all_results.items():
        for r in results:
            if r['roi'] >= 100:
                over_100.append({
                    'bet_type': bet_type,
                    **r
                })
    
    over_100 = sorted(over_100, key=lambda x: x['roi'], reverse=True)
    
    if not over_100:
        print("⚠️ ROI 100%超えの戦略は見つかりませんでした")
        return
    
    # ヘッダー
    print(f"{'券種':>8} | {'Prob':>5} | {'EV':>4} | {'Odds':>10} | {'ROI':>8} | {'的中率':>7} | {'Races':>6}")
    print("-" * 70)
    
    for r in over_100:
        odds_range = f"{r['min_odds']:.0f}-{r['max_odds']:.0f}"
        print(f"{r['bet_type']:>8} | {r['min_prob']:>5.2f} | {r['min_ev']:>4.1f} | {odds_range:>10} | {r['roi']:>7.1f}% | {r['hit_rate']:>6.1f}% | {r['races']:>6}")
    
    print(f"\n📊 合計 {len(over_100)} 件のROI 100%超え戦略を発見!")

def main():
    print("\n" + "#"*80)
    print("# 📊 高速版 網羅的グリッドサーチ (2023+2024年)")
    print("# 条件: AIスコア(Prob) × 期待値(EV) × オッズ範囲")
    print("#"*80)
    
    # データロード
    years = [2024, 2025]
    logger.info(f"Loading data for years: {years}")
    
    df = load_predictions_from_db(years)
    if df is None:
        return
    
    # モデル予測
    logger.info("Loading model and predicting...")
    df = load_model_and_predict(df, 'ensemble', 'v4_2025')
    
    # 前処理
    df = preprocess_data(df)
    logger.info(f"Preprocessed: {len(df)} rows, {df['race_id'].nunique()} races")
    
    # 払戻データロード
    pay_df = load_payouts(years)
    payout_map = build_payout_map(pay_df)
    logger.info(f"Built payout map for {len(payout_map)} races")
    
    # グリッドサーチ実行
    all_results = {}
    
    for bet_type in ['tansho', 'umaren', 'sanrenpuku', 'sanrentan']:
        results = fast_grid_search(df, payout_map, bet_type)
        all_results[bet_type] = results
        print_results(f"{bet_type.upper()} グリッドサーチ結果", results)
    
    # まとめ
    find_over_100_strategies(all_results)
    
    print("\n✅ グリッドサーチ完了!")

if __name__ == "__main__":
    main()

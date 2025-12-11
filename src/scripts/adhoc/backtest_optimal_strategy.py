"""
新戦略バックテスト (2025年YTD)
OptimalStrategy v2 の実際のパフォーマンスを検証
"""
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from scipy.special import softmax
from sqlalchemy import create_engine, text

sys.path.append('src')
from inference.optimal_strategy import OptimalStrategy

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
    data_path = 'data/processed/preprocessed_data.parquet'
    df = pd.read_parquet(data_path)
    df = df[df['year'] == 2025].copy()
    logger.info(f"Loaded {len(df)} rows for 2025")
    return df

def load_model_and_predict(df):
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
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(99)
    
    return df

def load_payouts():
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
    payout_map = defaultdict(lambda: {'sanrenpuku': {}, 'sanrentan': {}})
    
    for _, row in pay_df.iterrows():
        rid = row['race_id']
        
        for prefix, max_count in [('haraimodoshi_sanrenpuku', 3), ('haraimodoshi_sanrentan', 6)]:
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

def run_backtest(df, payout_map):
    """新戦略でバックテスト"""
    strategy = OptimalStrategy()
    
    # 戦略別の集計
    results_by_strategy = defaultdict(lambda: {'races': 0, 'cost': 0, 'return': 0, 'hits': 0})
    results_by_condition = defaultdict(lambda: {'races': 0, 'cost': 0, 'return': 0, 'hits': 0})
    
    race_groups = {rid: grp for rid, grp in df.groupby('race_id') if len(grp) >= 6}
    logger.info(f"Processing {len(race_groups)} races for 2025 YTD...")
    
    for rid, grp in race_groups.items():
        if rid not in payout_map:
            continue
        
        sorted_g = grp.sort_values('score', ascending=False)
        
        horse_numbers = sorted_g['horse_number'].astype(int).tolist()
        scores = sorted_g['score'].tolist()
        popularities = sorted_g['popularity'].fillna(99).astype(int).tolist()
        odds_list = sorted_g['odds'].fillna(1.0).tolist()
        probs = sorted_g['prob'].tolist()
        
        # 戦略を適用
        rec = strategy.analyze_race(horse_numbers, scores, popularities, odds_list, probs)
        
        if rec.bet_type == 'skip':
            continue
        
        # 条件名を取得
        condition = strategy._classify_race(scores, popularities)
        
        # コスト計算
        cost = len(rec.tickets) * 100
        
        # 払戻計算
        ret = 0
        hit = 0
        pm = payout_map[rid]
        
        for ticket in rec.tickets:
            if rec.bet_type == 'sanrentan':
                key = f"{ticket[0]:02}{ticket[1]:02}{ticket[2]:02}"
                if key in pm['sanrentan']:
                    ret += pm['sanrentan'][key]
                    hit = 1
            
            elif rec.bet_type == 'sanrenpuku':
                sorted_ticket = sorted(ticket)
                key = f"{sorted_ticket[0]:02}{sorted_ticket[1]:02}{sorted_ticket[2]:02}"
                if key in pm['sanrenpuku']:
                    ret += pm['sanrenpuku'][key]
                    hit = 1
        
        # 集計
        results_by_strategy[rec.strategy_name]['races'] += 1
        results_by_strategy[rec.strategy_name]['cost'] += cost
        results_by_strategy[rec.strategy_name]['return'] += ret
        results_by_strategy[rec.strategy_name]['hits'] += hit
        
        results_by_condition[condition]['races'] += 1
        results_by_condition[condition]['cost'] += cost
        results_by_condition[condition]['return'] += ret
        results_by_condition[condition]['hits'] += hit
    
    return dict(results_by_strategy), dict(results_by_condition)

def main():
    print("\n" + "#"*80)
    print("# 🎯 新戦略バックテスト (2025年YTD)")
    print("# OptimalStrategy v2 の実際のパフォーマンス検証")
    print("#"*80)
    
    df = load_data()
    df = load_model_and_predict(df)
    
    pay_df = load_payouts()
    payout_map = build_payout_map(pay_df)
    logger.info(f"Loaded payouts for {len(payout_map)} races")
    
    results_by_strategy, results_by_condition = run_backtest(df, payout_map)
    
    # 結果表示
    print(f"\n{'='*80}")
    print("📊 戦略別パフォーマンス")
    print(f"{'='*80}")
    print(f"{'戦略名':<20} | {'レース':>6} | {'コスト':>10} | {'払戻':>10} | {'ROI':>8} | {'的中':>5}")
    print("-" * 75)
    
    total_cost = 0
    total_return = 0
    total_hits = 0
    total_races = 0
    
    for name, stats in sorted(results_by_strategy.items(), key=lambda x: x[1]['return'] / x[1]['cost'] * 100 if x[1]['cost'] > 0 else 0, reverse=True):
        if stats['cost'] > 0:
            roi = stats['return'] / stats['cost'] * 100
            hit_rate = stats['hits'] / stats['races'] * 100
            print(f"{name:<20} | {stats['races']:>6} | {stats['cost']:>10,} | {stats['return']:>10,} | {roi:>7.1f}% | {hit_rate:>4.1f}%")
            total_cost += stats['cost']
            total_return += stats['return']
            total_hits += stats['hits']
            total_races += stats['races']
    
    print("-" * 75)
    if total_cost > 0:
        overall_roi = total_return / total_cost * 100
        overall_hit = total_hits / total_races * 100 if total_races > 0 else 0
        print(f"{'合計':<20} | {total_races:>6} | {total_cost:>10,} | {total_return:>10,} | {overall_roi:>7.1f}% | {overall_hit:>4.1f}%")
    
    # 条件別
    print(f"\n{'='*80}")
    print("📈 条件別パフォーマンス")
    print(f"{'='*80}")
    print(f"{'条件':<15} | {'レース':>6} | {'コスト':>10} | {'払戻':>10} | {'ROI':>8} | {'期待ROI':>8}")
    print("-" * 75)
    
    expected_roi_map = {
        'longshot': 232.8,
        'balanced': 210.4,
        'midrange': 168.6,
        'small_gap': 137.2,
        'top3_dominant': 119.6,
    }
    
    for cond, stats in sorted(results_by_condition.items(), key=lambda x: x[1]['return'] / x[1]['cost'] * 100 if x[1]['cost'] > 0 else 0, reverse=True):
        if stats['cost'] > 0:
            roi = stats['return'] / stats['cost'] * 100
            expected = expected_roi_map.get(cond, '-')
            expected_str = f"{expected:.0f}%" if isinstance(expected, float) else expected
            print(f"{cond:<15} | {stats['races']:>6} | {stats['cost']:>10,} | {stats['return']:>10,} | {roi:>7.1f}% | {expected_str:>8}")
    
    # ファイル保存
    os.makedirs('reports', exist_ok=True)
    with open('reports/optimal_strategy_backtest_2025.txt', 'w', encoding='utf-8') as f:
        f.write("=== 新戦略バックテスト (2025年YTD) ===\n\n")
        
        f.write("--- 戦略別パフォーマンス ---\n")
        for name, stats in results_by_strategy.items():
            if stats['cost'] > 0:
                roi = stats['return'] / stats['cost'] * 100
                f.write(f"{name}: {stats['races']}レース, ROI {roi:.1f}%\n")
        
        f.write(f"\n--- 合計 ---\n")
        if total_cost > 0:
            f.write(f"レース: {total_races}, コスト: {total_cost:,}, 払戻: {total_return:,}, ROI: {overall_roi:.1f}%\n")
        
        f.write("\n--- 条件別パフォーマンス ---\n")
        for cond, stats in results_by_condition.items():
            if stats['cost'] > 0:
                roi = stats['return'] / stats['cost'] * 100
                expected = expected_roi_map.get(cond, '-')
                f.write(f"{cond}: {stats['races']}レース, ROI {roi:.1f}% (期待: {expected}%)\n")
    
    print("\n結果を reports/optimal_strategy_backtest_2025.txt に保存しました")
    print("\n✅ バックテスト完了!")

if __name__ == "__main__":
    main()

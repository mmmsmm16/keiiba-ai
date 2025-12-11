"""
2025年分割検証 (Hold-out Validation)
- Optimization: 2025/01/01 - 2025/09/30
- Backtest:     2025/10/01 - 2025/12/31
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
    """2025年データをロードして分割"""
    data_path = 'data/processed/preprocessed_data.parquet'
    df = pd.read_parquet(data_path)
    df = df[df['year'] == 2025].copy()
    
    # date列をdatetime型に変換
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        # race_idから日付パースが必要だが、date列がある前提
        pass
        
    split_date = pd.Timestamp('2025-09-30')
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()
    
    logger.info(f"Optimization (Jan-Sep): {len(train_df)} rows")
    logger.info(f"Backtest (Oct-Dec): {len(test_df)} rows")
    
    return train_df, test_df

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

# ランキング方法 (共通)
def rank_by_score(grp):
    return grp.sort_values('score', ascending=False)
def rank_group_top2fix(grp):
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6: return sorted_g
    top2 = sorted_g.head(2)
    rank3_6 = sorted_g.iloc[2:6].sort_values('ev', ascending=False)
    rest = sorted_g.iloc[6:]
    return pd.concat([top2, rank3_6, rest])
def rank_score_then_ev_5(grp):
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) <= 5: return sorted_g.sort_values('ev', ascending=False)
    top = sorted_g.head(5).sort_values('ev', ascending=False)
    rest = sorted_g.iloc[5:]
    return pd.concat([top, rest])
def rank_weighted_07(grp):
    grp = grp.copy()
    s_min, s_max = grp['score'].min(), grp['score'].max()
    e_min, e_max = grp['ev'].min(), grp['ev'].max()
    grp['score_norm'] = (grp['score'] - s_min) / (s_max - s_min) if s_max > s_min else 0.5
    grp['ev_norm'] = (grp['ev'] - e_min) / (e_max - e_min) if e_max > e_min else 0.5
    grp['weighted'] = 0.7 * grp['score_norm'] + 0.3 * grp['ev_norm']
    return grp.sort_values('weighted', ascending=False)

# 条件判定 (共通)
def get_race_conditions(grp):
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6: return None
    
    scores = sorted_g['score'].values
    score_range = scores[0] - scores[5]
    top3_gap = scores[0] - scores[2]
    bottom_gap = scores[2] - scores[5]
    top1_pop = int(sorted_g.iloc[0]['popularity']) if not pd.isna(sorted_g.iloc[0]['popularity']) else 99
    
    if score_range >= 0.5: dist_cat = 'large_gap'
    elif score_range >= 0.3: dist_cat = 'medium_gap'
    elif score_range >= 0.15: dist_cat = 'small_gap'
    else: dist_cat = 'balanced'
    
    if top3_gap >= 0.3 and bottom_gap < 0.1: structure = 'top3_dominant'
    else: structure = 'normal'
    
    if top1_pop >= 7: pop_cat = 'longshot'
    elif top1_pop >= 4: pop_cat = 'midrange'
    else: pop_cat = 'favorite'
    
    return {
        'dist_cat': dist_cat, 'structure': structure, 'pop_cat': pop_cat, 'top1_pop': top1_pop
    }

# ベットシミュレーション (簡略化: 必要な戦略のみ)
def simulate_bet(horses, payout_map, rid, bet_type):
    if rid not in payout_map: return 0, 0, 0
    pm = payout_map[rid]
    
    cost, ret, hit = 0, 0, 0
    
    if bet_type == 'tansho':
        key = f"{horses[0]:02}"
        if key in pm['tansho']: return 100, pm['tansho'][key], 1
        return 100, 0, 0
        
    elif bet_type == 'umaren_3':
        axis = horses[0]
        opps = horses[1:4]
        cost = 300
        for opp in opps:
            pair = sorted([axis, opp])
            key = f"{pair[0]:02}{pair[1]:02}"
            if key in pm['umaren']:
                ret += pm['umaren'][key]
                hit = 1
        return cost, ret, hit

    elif bet_type == 'sanrenpuku':
        axis = horses[0]
        tickets = [t for t in combinations([axis] + horses[1:6], 3) if axis in t]
        cost = len(tickets) * 100
        for t in tickets:
            key = f"{sorted(t)[0]:02}{sorted(t)[1]:02}{sorted(t)[2]:02}"
            if key in pm['sanrenpuku']:
                ret += pm['sanrenpuku'][key]
                hit = 1
        return cost, ret, hit
        
    elif bet_type == 'sanrentan_6':
        axis = horses[0]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(horses[1:4], 2)]
        cost = 600
        for t in tickets:
            key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
            if key in pm['sanrentan']:
                ret += pm['sanrentan'][key]
                hit = 1
        return cost, ret, hit

    elif bet_type == 'sanrentan_20':
        axis = horses[0]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(horses[1:6], 2)]
        cost = 2000
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
    
    race_data = {}
    for rid, grp in df.groupby('race_id'):
        if len(grp) < 6: continue
        cond = get_race_conditions(grp)
        if cond: race_data[rid] = {'grp': grp, 'cond': cond}

    results = []
    
    # 全条件探索
    for cond_key in ['midrange', 'top3_dominant', 'balanced', 'small_gap', 'longshot', 'large_gap', 'medium_gap', 'favorite']:
        matching_races = {}
        for rid, data in race_data.items():
            c = data['cond']
            match = False
            if cond_key == 'midrange' and c['pop_cat'] == 'midrange': match = True
            elif cond_key == 'longshot' and c['pop_cat'] == 'longshot': match = True
            elif cond_key == 'favorite' and c['pop_cat'] == 'favorite': match = True
            elif cond_key == 'top3_dominant' and c['structure'] == 'top3_dominant': match = True
            elif cond_key == 'balanced' and c['dist_cat'] == 'balanced': match = True
            elif cond_key == 'small_gap' and c['dist_cat'] == 'small_gap': match = True
            elif cond_key == 'medium_gap' and c['dist_cat'] == 'medium_gap': match = True
            elif cond_key == 'large_gap' and c['dist_cat'] == 'large_gap': match = True
            
            if match: matching_races[rid] = data
            
        if len(matching_races) < 10: continue

        for rank_name, rank_func in ranking_methods.items():
            for bet_type in betting_strategies:
                stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
                for rid, data in matching_races.items():
                    sorted_grp = rank_func(data['grp'].copy())
                    horses = sorted_grp['horse_number'].astype(int).tolist()
                    c, r, h = simulate_bet(horses, payout_map, rid, bet_type)
                    stats['races'] += 1
                    stats['cost'] += c
                    stats['return'] += r
                    stats['hits'] += h
                
                if stats['cost'] > 0:
                    roi = stats['return'] / stats['cost'] * 100
                    results.append({
                        'condition': cond_key,
                        'ranking': rank_name,
                        'betting': bet_type,
                        'roi': roi,
                        'races': stats['races'],
                        'stats': stats
                    })
    
    return results

def main():
    print("\n=== Hold-out Validation (2025 Split) ===")
    train_df, test_df = load_data()
    
    # 予測
    logger.info("Predicting for Train set...")
    train_df = load_model_and_predict(train_df)
    logger.info("Predicting for Test set...")
    test_df = load_model_and_predict(test_df)
    
    pay_df = load_payouts()
    payout_map = build_payout_map(pay_df)
    
    # 1. 最適化フェーズ (Jan-Sep)
    logger.info("Running Optimization on Jan-Sep data...")
    train_results = run_grid_search(train_df, payout_map)
    
    # ROI 100%超え & レース数20以上の良戦略を抽出
    best_strategies = {}
    for r in train_results:
        if r['roi'] > 110 and r['races'] >= 20: # 閾値を少し厳しく(110%)
             # 条件ごとにベスト1を選択
             cond = r['condition']
             if cond not in best_strategies or r['roi'] > best_strategies[cond]['roi']:
                 best_strategies[cond] = r
    
    print(f"\nTraining Results (Jan-Sep) - Best Strategies:")
    for cond, r in best_strategies.items():
        print(f"  {cond}: {r['ranking']} - {r['betting']} (ROI: {r['roi']:.1f}%, {r['races']} races)")

    # 2. バックテストフェーズ (Oct-Dec)
    logger.info("Running Backtest on Oct-Dec data...")
    test_results = run_grid_search(test_df, payout_map)
    
    print(f"\nBacktest Results (Oct-Dec) - Validation:")
    print(f"{'Condition':<15} | {'Strategy':<30} | {'Train ROI':>10} | {'Test ROI':>10} | {'Test Races':>10}")
    print("-" * 90)
    
    test_lookup = {(r['condition'], r['ranking'], r['betting']): r for r in test_results}
    
    total_cost = 0
    total_return = 0
    
    for cond, train_r in best_strategies.items():
        key = (cond, train_r['ranking'], train_r['betting'])
        if key in test_lookup:
            test_r = test_lookup[key]
            
            strat_name = f"{train_r['ranking'][:10]}.. - {train_r['betting']}"
            print(f"{cond:<15} | {strat_name:<30} | {train_r['roi']:>9.1f}% | {test_r['roi']:>9.1f}% | {test_r['races']:>10}")
            
            total_cost += test_r['stats']['cost']
            total_return += test_r['stats']['return']
        else:
            print(f"{cond:<15} | {strat_name:<30} | {train_r['roi']:>9.1f}% | {'N/A':>10} | {'0':>10}")

    if total_cost > 0:
        overall_roi = total_return / total_cost * 100
        print("-" * 90)
        print(f"{'OVERALL':<48} | {'':>10} | {overall_roi:>9.1f}% |")

    # Save report
    os.makedirs('reports', exist_ok=True)
    with open('reports/holdout_validation_2025.txt', 'w') as f:
        f.write("Hold-out Validation 2025 (Jan-Sep Train -> Oct-Dec Test)\n")
        f.write(f"Overall Test ROI: {overall_roi:.1f}%\n" if total_cost > 0 else "Overall Test ROI: N/A\n")

if __name__ == "__main__":
    main()

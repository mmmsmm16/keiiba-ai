"""
最終統合戦略シミュレーション

ロジック:
1. Top3平均オッズ 10-30倍 -> 馬連流し (Top1 - Top2~5)
2. スコア差 < 0.02 -> 三連単1着固定 (Top1 -> Top2-5 -> Top2-5)
3. スコア差 0.02-0.05 -> 馬単1着固定 (Top1 -> Top2-5)
4. 上記以外 -> スキップ
"""
import os
import sys
import pickle
import json
import logging
from datetime import datetime
from itertools import combinations, permutations

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FOLDS = [
    {"valid_year": 2021, "name": "fold1"},
    {"valid_year": 2022, "name": "fold2"},
    {"valid_year": 2023, "name": "fold3"},
    {"valid_year": 2024, "name": "fold4"},
]

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_payout_data(years):
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

def build_payout_map(payout_df):
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'umaren': {}, 'umatan': {}, 'sanrentan': {}, 'wide': {}}
        
        # 馬連
        for i in range(1, 4):
            k_comb, k_pay = f'haraimodoshi_umaren_{i}a', f'haraimodoshi_umaren_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['umaren'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        # 馬単
        for i in range(1, 7):
            k_comb, k_pay = f'haraimodoshi_umatan_{i}a', f'haraimodoshi_umatan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['umatan'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        # 三連単
        for i in range(1, 7):
            k_comb, k_pay = f'haraimodoshi_sanrentan_{i}a', f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['sanrentan'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        # ワイド
        for i in range(1, 8):
            k_comb, k_pay = f'haraimodoshi_wide_{i}a', f'haraimodoshi_wide_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['wide'][str(row[k_comb])] = int(row[k_pay])
                except: pass
    return payout_map

def run_simulation(all_race_data, all_payout_maps):
    results = []
    
    for fold_name, race_data in all_race_data.items():
        payout_map = all_payout_maps[fold_name]
        
        total_bet = 0
        total_return = 0
        races_bet = 0
        races_skipped = 0
        
        rule_stats = {
            'rule1_mid_odds_umaren': {'bet': 0, 'return': 0, 'count': 0},
            'rule2_low_gap_sanrentan': {'bet': 0, 'return': 0, 'count': 0},
            'rule3_mid_gap_umatan': {'bet': 0, 'return': 0, 'count': 0},
            'rule4_large_field_wide_form': {'bet': 0, 'return': 0, 'count': 0},
        }
        
        for race_id, info in race_data.items():
            if race_id not in payout_map:
                continue
            
            payouts = payout_map[race_id]
            bet_amount = 0
            win_amount = 0
            rule_hit = None
            
            # Top3平均オッズ
            avg_top3_odds = info['avg_top3_odds']
            score_gap = info['score_gap']
            top_horses = info['horses']
            
            if avg_top3_odds is not None and 10 <= avg_top3_odds <= 30:
                # Rule 1: 馬連流し (Top1 - Top2~5)
                rule_hit = 'rule1_mid_odds_umaren'
                combos = [(top_horses[0], o) for o in top_horses[1:5]]
                bet_amount = len(combos) * 100
                for c in combos:
                    c_sorted = sorted(c)
                    key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
                    if key in payouts['umaren']:
                        win_amount += payouts['umaren'][key]
            
            elif score_gap < 0.02:
                # Rule 2: 三連単1着固定 (Top1 -> Top2-5 -> Top2-5)
                rule_hit = 'rule2_low_gap_sanrentan'
                formations = [(top_horses[0], o1, o2) for o1, o2 in permutations(top_horses[1:5], 2)]
                bet_amount = len(formations) * 100
                for p in formations:
                    key = f"{p[0]:02}{p[1]:02}{p[2]:02}"
                    if key in payouts['sanrentan']:
                        win_amount += payouts['sanrentan'][key]

            elif 0.02 <= score_gap < 0.05:
                # Rule 3: 馬単1着固定 (Top1 -> Top2-5)
                rule_hit = 'rule3_mid_gap_umatan'
                combos = [(top_horses[0], o) for o in top_horses[1:5]]
                bet_amount = len(combos) * 100
                for p in combos:
                    key = f"{p[0]:02}{p[1]:02}"
                    if key in payouts['umatan']:
                        win_amount += payouts['umatan'][key]
            
            elif len(info['horses']) >= 15: # 多頭数
                # Rule 4: ワイドフォーメーション (1,2 -> 3,4,5)
                rule_hit = 'rule4_large_field_wide_form'
                # 1,2着目から3,4,5着目へ流し
                combos = [(a, b) for a in top_horses[:2] for b in top_horses[2:5]] if len(top_horses) >= 5 else []
                bet_amount = len(combos) * 100
                for c in combos:
                    c_sorted = sorted(c)
                    key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
                    if key in payouts['wide']:
                        win_amount += payouts['wide'][key]

            else:
                # スキップ
                races_skipped += 1
                continue
            
            total_bet += bet_amount
            total_return += win_amount
            races_bet += 1
            if rule_hit:
                rule_stats[rule_hit]['bet'] += bet_amount
                rule_stats[rule_hit]['return'] += win_amount
                rule_stats[rule_hit]['count'] += 1
        
        roi = total_return / total_bet * 100 if total_bet > 0 else 0
        results.append({
            'fold': fold_name,
            'roi': roi,
            'bet': total_bet,
            'return': total_return,
            'races_bet': races_bet,
            'races_skipped': races_skipped,
            'total_races': races_bet + races_skipped,
            'rule_stats': rule_stats
        })
    
    return results

def main():
    cv_dir = os.path.join(project_root, 'experiments', 'v23_regression_cv')
    
    logger.info("=" * 60)
    logger.info("最終統合戦略シミュレーション")
    logger.info("=" * 60)
    
    parquet_path = os.path.join(project_root, 'data/processed/preprocessed_data_v10_leakfix.parquet')
    df = pd.read_parquet(parquet_path)
    
    pickle_path = os.path.join(project_root, 'data/processed/lgbm_datasets_v10_leakfix.pkl')
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    all_race_data = {}
    all_payout_maps = {}
    
    for fold in FOLDS:
        fold_name = fold['name']
        valid_year = fold['valid_year']
        
        meta_path = os.path.join(cv_dir, fold_name, 'meta_v23.pkl')
        if not os.path.exists(meta_path):
            continue
        
        with open(os.path.join(cv_dir, fold_name, 'lgbm_v23.pkl'), 'rb') as f:
            lgbm = pickle.load(f)
        with open(os.path.join(cv_dir, fold_name, 'catboost_v23.pkl'), 'rb') as f:
            catboost = pickle.load(f)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        valid_df = df[df['year'] == valid_year].copy()
        for col in feature_cols:
            if col not in valid_df.columns:
                valid_df[col] = 0
        if 'venue' in valid_df.columns:
            valid_df = valid_df[valid_df['venue'].isin([f"{i:02}" for i in range(1, 11)])]
        
        X = valid_df[feature_cols]
        valid_df['score'] = meta.predict(np.column_stack([lgbm.predict(X), catboost.predict(X)]))
        
        payout_df = load_payout_data([valid_year])
        
        # プリコンピュート (必要な情報のみ)
        race_data = {}
        for race_id, group in valid_df.groupby('race_id'):
            sorted_group = group.nlargest(6, 'score')
            top_horses = sorted_group['horse_number'].astype(int).tolist()
            top_scores = sorted_group['score'].tolist()
            if len(top_horses) < 3: continue
            
            top3_odds = sorted_group['odds'].head(3).tolist() if 'odds' in sorted_group.columns else []
            avg_top3_odds = np.mean([o for o in top3_odds if not pd.isna(o)]) if top3_odds else None
            
            race_data[race_id] = {
                'horses': top_horses,
                'score_gap': top_scores[0] - top_scores[1] if len(top_scores) >= 2 else 0,
                'avg_top3_odds': avg_top3_odds
            }
        
        all_race_data[fold_name] = race_data
        all_payout_maps[fold_name] = build_payout_map(payout_df)
    
    results = run_simulation(all_race_data, all_payout_maps)
    
    logger.info("\n=== シミュレーション結果 ===")
    
    total_bet = 0
    total_return = 0
    
    for r in results:
        logger.info(f"\n{r['fold']} ({FOLDS[next(i for i, f in enumerate(FOLDS) if f['name'] == r['fold'])]['valid_year']})")
        logger.info(f"  ROI: {r['roi']:.1f}%")
        logger.info(f"  投資: {r['bet']:,}円")
        logger.info(f"  回収: {r['return']:,}円")
        logger.info(f"  購入: {r['races_bet']}R (スキップ: {r['races_skipped']}R, 率: {r['races_skipped']/r['total_races']:.1%})")
        
        logger.info("  [ルール別]")
        for rule, stats in r['rule_stats'].items():
            roi = stats['return'] / stats['bet'] * 100 if stats['bet'] > 0 else 0
            logger.info(f"    {rule}: ROI={roi:.1f}% ({stats['count']}R)")
        
        total_bet += r['bet']
        total_return += r['return']
    
    avg_roi = total_return / total_bet * 100 if total_bet > 0 else 0
    logger.info(f"\n全期間平均ROI: {avg_roi:.1f}%")
    logger.info(f"総投資: {total_bet:,}円")
    logger.info(f"総回収: {total_return:,}円")
    
    # 保存
    output_path = os.path.join(cv_dir, 'final_simulation_v23.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
            'total_summary': {
                'roi': avg_roi,
                'bet': total_bet,
                'return': total_return
            }
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n結果を保存: {output_path}")

if __name__ == "__main__":
    main()

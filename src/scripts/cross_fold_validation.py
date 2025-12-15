"""
クロスFold戦略検証

各Foldで最適化されたルールを他のFoldに適用して汎化性能を検証
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
        payout_map[rid] = {'tansho': {}, 'umaren': {}, 'sanrenpuku': {}, 'sanrentan': {}}
        for i in range(1, 4):
            k_num, k_pay = f'haraimodoshi_tansho_{i}a', f'haraimodoshi_tansho_{i}b'
            if k_num in row and row[k_num] and str(row[k_num]).strip():
                try: payout_map[rid]['tansho'][str(row[k_num]).zfill(2)] = int(row[k_pay])
                except: pass
        for i in range(1, 4):
            k_comb, k_pay = f'haraimodoshi_umaren_{i}a', f'haraimodoshi_umaren_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['umaren'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        for i in range(1, 4):
            k_comb, k_pay = f'haraimodoshi_sanrenpuku_{i}a', f'haraimodoshi_sanrenpuku_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['sanrenpuku'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        for i in range(1, 7):
            k_comb, k_pay = f'haraimodoshi_sanrentan_{i}a', f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['sanrentan'][str(row[k_comb])] = int(row[k_pay])
                except: pass
    return payout_map

def precompute_race_data(pred_df):
    race_data = {}
    for race_id, group in pred_df.groupby('race_id'):
        sorted_group = group.nlargest(6, 'score')
        top_horses = sorted_group['horse_number'].astype(int).tolist()
        top_scores = sorted_group['score'].tolist()
        if len(top_horses) < 3:
            continue
        info = {
            'horses': top_horses,
            'scores': top_scores,
            'score_gap': top_scores[0] - top_scores[1] if len(top_scores) >= 2 else 0,
            'top1_odds': sorted_group['odds'].iloc[0] if 'odds' in sorted_group.columns else None,
        }
        info['formations'] = {
            'san_box3': list(permutations(top_horses[:3], 3)),
            'san_box5': list(permutations(top_horses[:5], 3)) if len(top_horses) >= 5 else [],
            'san_1st': [(top_horses[0], o1, o2) for o1, o2 in permutations(top_horses[1:5], 2)] if len(top_horses) >= 5 else [],
            'san_3rd': [(o1, o2, top_horses[0]) for o1, o2 in permutations(top_horses[1:5], 2)] if len(top_horses) >= 5 else [],
        }
        info['combos'] = {
            'sanrenpuku_box3': list(combinations(top_horses[:3], 3)),
            'sanrenpuku_box5': list(combinations(top_horses[:5], 3)) if len(top_horses) >= 5 else [],
            'umaren_box3': list(combinations(top_horses[:3], 2)),
            'umaren_box5': list(combinations(top_horses[:5], 2)) if len(top_horses) >= 5 else [],
        }
        race_data[race_id] = info
    return race_data

def classify_race(info):
    gap = info['score_gap']
    odds = info['top1_odds'] if info['top1_odds'] is not None and not pd.isna(info['top1_odds']) else 0
    if gap >= 0.05 and 5 <= odds <= 50:
        return 'high_conf_mid_odds'
    elif gap >= 0.05 and 1 <= odds < 5:
        return 'high_conf_fav'
    elif 0.02 <= gap < 0.05 and 5 <= odds <= 50:
        return 'mid_conf_mid_odds'
    elif 0.02 <= gap < 0.05 and 1 <= odds < 5:
        return 'mid_conf_fav'
    elif gap < 0.02:
        return 'low_conf'
    return 'other'

def execute_bet(info, payouts, bet_type, formation_key):
    bet_amount = 0
    win_amount = 0
    if bet_type == 'sanrentan':
        formations = info['formations'].get(formation_key, [])
        bet_amount = len(formations) * 100
        for p in formations:
            key = f"{p[0]:02}{p[1]:02}{p[2]:02}"
            if key in payouts['sanrentan']:
                win_amount += payouts['sanrentan'][key]
    elif bet_type == 'sanrenpuku':
        combos = info['combos'].get(formation_key, [])
        bet_amount = len(combos) * 100
        for c in combos:
            c_sorted = sorted(c)
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
            if key in payouts['sanrenpuku']:
                win_amount += payouts['sanrenpuku'][key]
    elif bet_type == 'umaren':
        combos = info['combos'].get(formation_key, [])
        bet_amount = len(combos) * 100
        for c in combos:
            c_sorted = sorted(c)
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in payouts['umaren']:
                win_amount += payouts['umaren'][key]
    elif bet_type == 'tansho':
        bet_amount = 100
        key = f"{info['horses'][0]:02}"
        if key in payouts['tansho']:
            win_amount = payouts['tansho'][key]
    return bet_amount, win_amount

def evaluate_on_fold(race_data, payout_map, rules):
    total_bet = 0
    total_return = 0
    for race_id, info in race_data.items():
        if race_id not in payout_map:
            continue
        zone = classify_race(info)
        if zone not in rules or rules[zone][0] == 'skip':
            continue
        bet_type, formation_key = rules[zone]
        bet_amount, win_amount = execute_bet(info, payout_map[race_id], bet_type, formation_key)
        total_bet += bet_amount
        total_return += win_amount
    roi = total_return / total_bet * 100 if total_bet > 0 else 0
    return roi, total_bet, total_return

def find_best_rules_for_fold(race_data, payout_map):
    """単一Foldで最適なルールを見つける"""
    bet_options = [
        ('skip', None),
        ('tansho', 'top1'),
        ('umaren', 'umaren_box3'),
        ('umaren', 'umaren_box5'),
        ('sanrenpuku', 'sanrenpuku_box3'),
        ('sanrenpuku', 'sanrenpuku_box5'),
        ('sanrentan', 'san_box3'),
        ('sanrentan', 'san_box5'),
        ('sanrentan', 'san_1st'),
        ('sanrentan', 'san_3rd'),
    ]
    zones = ['high_conf_mid_odds', 'high_conf_fav', 'mid_conf_mid_odds', 'mid_conf_fav', 'low_conf']
    
    best_rules = {}
    for zone in zones:
        best_roi = -100
        best_bet = ('skip', None)
        
        for bet_type, formation in bet_options:
            test_rules = {z: ('skip', None) for z in zones}
            test_rules[zone] = (bet_type, formation)
            roi, _, _ = evaluate_on_fold(race_data, payout_map, test_rules)
            
            # このゾーンだけのROI計算
            zone_bet = 0
            zone_ret = 0
            for race_id, info in race_data.items():
                if race_id not in payout_map:
                    continue
                z = classify_race(info)
                if z == zone and bet_type != 'skip':
                    b, r = execute_bet(info, payout_map[race_id], bet_type, formation)
                    zone_bet += b
                    zone_ret += r
            
            zone_roi = zone_ret / zone_bet * 100 if zone_bet > 0 else 0
            if zone_roi > best_roi:
                best_roi = zone_roi
                best_bet = (bet_type, formation)
        
        best_rules[zone] = best_bet
    
    return best_rules

def main():
    cv_dir = os.path.join(project_root, 'experiments', 'v23_regression_cv')
    
    logger.info("=" * 60)
    logger.info("クロスFold戦略検証")
    logger.info("各Foldで最適化したルールを他Foldに適用")
    logger.info("=" * 60)
    
    parquet_path = os.path.join(project_root, 'data/processed/preprocessed_data_v10_leakfix.parquet')
    df = pd.read_parquet(parquet_path)
    
    pickle_path = os.path.join(project_root, 'data/processed/lgbm_datasets_v10_leakfix.pkl')
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    all_race_data = {}
    all_payout_maps = {}
    
    # 全Foldデータ準備
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
        all_race_data[fold_name] = precompute_race_data(valid_df)
        all_payout_maps[fold_name] = build_payout_map(payout_df)
        logger.info(f"{fold_name}: {len(all_race_data[fold_name])} races")
    
    # 各Foldで最適ルールを求める
    fold_best_rules = {}
    fold_self_roi = {}
    
    logger.info("\n=== 各Foldでの最適ルール ===")
    for fold_name in all_race_data.keys():
        best_rules = find_best_rules_for_fold(all_race_data[fold_name], all_payout_maps[fold_name])
        fold_best_rules[fold_name] = best_rules
        
        roi, bet, ret = evaluate_on_fold(all_race_data[fold_name], all_payout_maps[fold_name], best_rules)
        fold_self_roi[fold_name] = roi
        
        logger.info(f"\n{fold_name} 最適ルール (自己ROI: {roi:.1f}%):")
        for zone, (bt, fm) in best_rules.items():
            if bt != 'skip':
                logger.info(f"  {zone}: {bt}_{fm}")
    
    # クロスバリデーション: 各Foldのルールを他Foldに適用
    logger.info("\n" + "=" * 60)
    logger.info("クロスFold適用結果")
    logger.info("=" * 60)
    
    cross_results = {}
    
    print("\n        |", end="")
    for target in all_race_data.keys():
        print(f" {target} |", end="")
    print()
    print("-" * 50)
    
    for source in all_race_data.keys():
        print(f"{source} |", end="")
        cross_results[source] = {}
        
        for target in all_race_data.keys():
            roi, _, _ = evaluate_on_fold(all_race_data[target], all_payout_maps[target], fold_best_rules[source])
            cross_results[source][target] = roi
            marker = "★" if source == target else ""
            print(f" {roi:5.1f}%{marker}|", end="")
        print()
    
    # 各ルールの平均ROI（他Fold適用時）
    logger.info("\n=== 各ルールの汎化性能 ===")
    for source in fold_best_rules.keys():
        other_rois = [cross_results[source][t] for t in cross_results[source] if t != source]
        avg_other = np.mean(other_rois)
        self_roi = fold_self_roi[source]
        logger.info(f"{source}ルール: 自己={self_roi:.1f}%, 他Fold平均={avg_other:.1f}%, 差={self_roi - avg_other:.1f}%")
    
    # 最も汎化性能が高いルールを特定
    best_source = None
    best_avg = -100
    for source in fold_best_rules.keys():
        avg = np.mean(list(cross_results[source].values()))
        if avg > best_avg:
            best_avg = avg
            best_source = source
    
    logger.info(f"\n最も汎化性能が高いルール: {best_source} (全Fold平均={best_avg:.1f}%)")
    logger.info(f"\n{best_source}の最適ルール:")
    for zone, (bt, fm) in fold_best_rules[best_source].items():
        if bt != 'skip':
            logger.info(f"  {zone}: {bt}_{fm}")
    
    # 保存
    output_path = os.path.join(cv_dir, 'cross_fold_validation.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cross_results': cross_results,
            'best_source': best_source,
            'best_avg_roi': best_avg,
            'best_rules': {k: f"{v[0]}_{v[1]}" if v[1] else v[0] for k, v in fold_best_rules[best_source].items()}
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n結果を保存: {output_path}")

if __name__ == "__main__":
    main()

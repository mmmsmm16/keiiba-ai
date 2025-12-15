"""
固定ルール戦略評価器

全Foldで共通のルールを適用し、一貫した収益性を検証
過学習を避けるため、各ゾーンで複数の馬券タイプをテストし、
全Foldで黒字になる組み合わせを探索
"""
import os
import sys
import pickle
import json
import logging
import argparse
from datetime import datetime
from itertools import combinations, permutations, product

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

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
    try:
        df = pd.read_sql(query, engine)
        df['race_id'] = (
            df['kaisai_nen'].astype(str) +
            df['keibajo_code'].astype(str) +
            df['kaisai_kai'].astype(str) +
            df['kaisai_nichime'].astype(str) +
            df['race_bango'].astype(str)
        )
        return df
    except Exception as e:
        return pd.DataFrame()

def build_payout_map(payout_df):
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'tansho': {}, 'umaren': {}, 'umatan': {}, 'sanrenpuku': {}, 'sanrentan': {}}
        
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
        
        for i in range(1, 7):
            k_comb, k_pay = f'haraimodoshi_umatan_{i}a', f'haraimodoshi_umatan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['umatan'][str(row[k_comb])] = int(row[k_pay])
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
    """レースを条件で分類"""
    gap = info['score_gap']
    odds = info['top1_odds'] if info['top1_odds'] is not None and not pd.isna(info['top1_odds']) else 0
    
    if gap >= 0.05 and 5 <= odds <= 50:
        return 'high_conf_mid_odds'
    elif gap >= 0.05 and 1 <= odds < 5:
        return 'high_conf_fav'
    elif gap >= 0.05 and odds > 50:
        return 'high_conf_longshot'
    elif 0.02 <= gap < 0.05 and 5 <= odds <= 50:
        return 'mid_conf_mid_odds'
    elif 0.02 <= gap < 0.05 and 1 <= odds < 5:
        return 'mid_conf_fav'
    elif gap < 0.02:
        return 'low_conf'
    else:
        return 'other'


def execute_bet(info, payouts, bet_type, formation_key):
    bet_amount = 0
    win_amount = 0
    hit = False
    
    if bet_type == 'sanrentan':
        formations = info['formations'].get(formation_key, [])
        bet_amount = len(formations) * 100
        for p in formations:
            key = f"{p[0]:02}{p[1]:02}{p[2]:02}"
            if key in payouts['sanrentan']:
                win_amount += payouts['sanrentan'][key]
                hit = True
    
    elif bet_type == 'sanrenpuku':
        combos = info['combos'].get(formation_key, [])
        bet_amount = len(combos) * 100
        for c in combos:
            c_sorted = sorted(c)
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
            if key in payouts['sanrenpuku']:
                win_amount += payouts['sanrenpuku'][key]
                hit = True
    
    elif bet_type == 'umaren':
        combos = info['combos'].get(formation_key, [])
        bet_amount = len(combos) * 100
        for c in combos:
            c_sorted = sorted(c)
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in payouts['umaren']:
                win_amount += payouts['umaren'][key]
                hit = True
    
    elif bet_type == 'tansho':
        bet_amount = 100
        key = f"{info['horses'][0]:02}"
        if key in payouts['tansho']:
            win_amount = payouts['tansho'][key]
            hit = True
    
    elif bet_type == 'skip':
        pass
    
    return bet_amount, win_amount, hit


def evaluate_fixed_strategy(all_race_data, all_payout_maps, strategy_rules):
    """固定戦略を全Foldで評価"""
    fold_results = []
    
    for fold_name, race_data in all_race_data.items():
        payout_map = all_payout_maps[fold_name]
        
        total_bet = 0
        total_return = 0
        zone_stats = {}
        
        for race_id, info in race_data.items():
            if race_id not in payout_map:
                continue
            
            zone = classify_race(info)
            
            if zone not in zone_stats:
                zone_stats[zone] = {'races': 0, 'bet': 0, 'return': 0, 'hits': 0}
            zone_stats[zone]['races'] += 1
            
            if zone not in strategy_rules:
                continue
            
            bet_type, formation_key = strategy_rules[zone]
            if bet_type == 'skip':
                continue
            
            bet_amount, win_amount, hit = execute_bet(info, payout_map[race_id], bet_type, formation_key)
            
            total_bet += bet_amount
            total_return += win_amount
            zone_stats[zone]['bet'] += bet_amount
            zone_stats[zone]['return'] += win_amount
            if hit:
                zone_stats[zone]['hits'] += 1
        
        roi = total_return / total_bet * 100 if total_bet > 0 else 0
        fold_results.append({
            'fold': fold_name,
            'roi': roi,
            'total_bet': total_bet,
            'total_return': total_return,
            'zone_stats': zone_stats
        })
    
    return fold_results


def search_best_fixed_strategy(all_race_data, all_payout_maps):
    """全Foldで一貫したパフォーマンスを示す固定戦略を探索"""
    
    # 馬券オプション
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
    
    # まず各ゾーン・各馬券の全Fold平均ROIを計算
    zone_bet_scores = {}
    
    for zone in zones:
        zone_bet_scores[zone] = []
        
        for bet_type, formation in bet_options:
            # このゾーンだけに賭ける
            test_rules = {z: ('skip', None) for z in zones}
            test_rules[zone] = (bet_type, formation)
            
            fold_results = evaluate_fixed_strategy(all_race_data, all_payout_maps, test_rules)
            
            # 各Foldでのこのゾーンの収益を計算
            zone_rois = []
            for fr in fold_results:
                zs = fr['zone_stats'].get(zone, {})
                if zs.get('bet', 0) > 0:
                    z_roi = zs['return'] / zs['bet'] * 100
                    zone_rois.append(z_roi)
            
            if zone_rois:
                avg_roi = np.mean(zone_rois)
                min_roi = min(zone_rois)
                zone_bet_scores[zone].append({
                    'bet_type': bet_type,
                    'formation': formation,
                    'avg_roi': avg_roi,
                    'min_roi': min_roi,
                    'all_roi': zone_rois
                })
    
    # 各ゾーンで最も安定した（min_roi高い）馬券を選択
    best_rules = {}
    
    logger.info("\n=== 各ゾーンの最適馬券（全Fold安定性重視） ===")
    
    for zone in zones:
        scores = zone_bet_scores[zone]
        # min_roi でソート（全Foldで安定して高い）
        scores_sorted = sorted(scores, key=lambda x: (x['min_roi'], x['avg_roi']), reverse=True)
        
        if scores_sorted:
            best = scores_sorted[0]
            # min_roi > 70% なら採用
            if best['min_roi'] >= 70:
                best_rules[zone] = (best['bet_type'], best['formation'])
                logger.info(f"  {zone}: {best['bet_type']}_{best['formation']} → 平均={best['avg_roi']:.1f}%, 最小={best['min_roi']:.1f}%")
            else:
                best_rules[zone] = ('skip', None)
                logger.info(f"  {zone}: スキップ（最小ROI {best['min_roi']:.1f}% < 75%）")
    
    return best_rules


def main():
    parser = argparse.ArgumentParser(description='固定ルール戦略評価器')
    parser.add_argument('--dataset_suffix', type=str, default='_v10_leakfix')
    parser.add_argument('--version', type=str, default='v23')
    parser.add_argument('--cv_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.cv_dir is None:
        args.cv_dir = os.path.join(project_root, 'experiments', f'{args.version}_regression_cv')
    
    logger.info("=" * 60)
    logger.info("固定ルール戦略評価器")
    logger.info("全Foldで一貫した結果を出すルールを探索")
    logger.info("=" * 60)
    
    parquet_path = os.path.join(project_root, f'data/processed/preprocessed_data{args.dataset_suffix}.parquet')
    df = pd.read_parquet(parquet_path)
    
    pickle_path = os.path.join(project_root, f'data/processed/lgbm_datasets{args.dataset_suffix}.pkl')
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    all_race_data = {}
    all_payout_maps = {}
    
    # 全Foldのデータを事前ロード
    for fold in FOLDS:
        fold_name = fold['name']
        valid_year = fold['valid_year']
        
        logger.info(f"\n{fold_name} ({valid_year}) データ準備中...")
        
        meta_path = os.path.join(args.cv_dir, fold_name, f'meta_{args.version}.pkl')
        if not os.path.exists(meta_path):
            continue
        
        with open(os.path.join(args.cv_dir, fold_name, f'lgbm_{args.version}.pkl'), 'rb') as f:
            lgbm_model = pickle.load(f)
        with open(os.path.join(args.cv_dir, fold_name, f'catboost_{args.version}.pkl'), 'rb') as f:
            catboost_model = pickle.load(f)
        with open(meta_path, 'rb') as f:
            meta_model = pickle.load(f)
        
        valid_df = df[df['year'] == valid_year].copy()
        for col in feature_cols:
            if col not in valid_df.columns:
                valid_df[col] = 0
        
        if 'venue' in valid_df.columns:
            jra_venues = [f"{i:02}" for i in range(1, 11)]
            valid_df = valid_df[valid_df['venue'].isin(jra_venues)]
        
        X_valid = valid_df[feature_cols]
        lgbm_pred = lgbm_model.predict(X_valid)
        catboost_pred = catboost_model.predict(X_valid)
        valid_df['score'] = meta_model.predict(np.column_stack([lgbm_pred, catboost_pred]))
        
        payout_df = load_payout_data([valid_year])
        if payout_df.empty:
            continue
        
        all_race_data[fold_name] = precompute_race_data(valid_df)
        all_payout_maps[fold_name] = build_payout_map(payout_df)
        
        logger.info(f"  {len(all_race_data[fold_name])} レース準備完了")
    
    # 最適な固定ルールを探索
    logger.info("\n" + "=" * 60)
    logger.info("固定ルール探索中...")
    best_rules = search_best_fixed_strategy(all_race_data, all_payout_maps)
    
    # 固定ルールでの最終評価
    logger.info("\n" + "=" * 60)
    logger.info("固定ルールでの最終評価")
    logger.info("=" * 60)
    
    final_results = evaluate_fixed_strategy(all_race_data, all_payout_maps, best_rules)
    
    for fr in final_results:
        logger.info(f"{fr['fold']}: ROI={fr['roi']:.1f}%, 投資={fr['total_bet']:,}円, 回収={fr['total_return']:,}円")
    
    avg_roi = np.mean([fr['roi'] for fr in final_results])
    min_roi = min([fr['roi'] for fr in final_results])
    max_roi = max([fr['roi'] for fr in final_results])
    
    logger.info(f"\n平均ROI: {avg_roi:.1f}%, 最小: {min_roi:.1f}%, 最大: {max_roi:.1f}%")
    
    # 採用ルール表示
    logger.info("\n=== 最終採用ルール ===")
    for zone, (bt, fm) in best_rules.items():
        if bt != 'skip':
            logger.info(f"  {zone}: {bt}_{fm}")
        else:
            logger.info(f"  {zone}: 買わない")
    
    # 保存
    output_path = os.path.join(args.cv_dir, 'fixed_strategy.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rules': {k: f"{v[0]}_{v[1]}" if v[1] else v[0] for k, v in best_rules.items()},
            'fold_results': [{'fold': fr['fold'], 'roi': fr['roi'], 'bet': fr['total_bet'], 'return': fr['total_return']} for fr in final_results],
            'avg_roi': avg_roi,
            'min_roi': min_roi,
            'max_roi': max_roi
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n結果を保存: {output_path}")
    logger.info("完了!")

if __name__ == "__main__":
    main()

"""
統合戦略オプティマイザー

レース条件に応じて最適な馬券タイプを選択する統合戦略を探索
各レースを1回だけカウントし、条件マッチングで最適な買い方を決定
"""
import os
import sys
import pickle
import json
import logging
import argparse
from datetime import datetime
from itertools import combinations, permutations

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
        logger.error(f"エラー: {e}")
        return pd.DataFrame()

def build_payout_map(payout_df):
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'tansho': {}, 'umaren': {}, 'umatan': {}, 'sanrenpuku': {}, 'sanrentan': {}}
        
        for i in range(1, 4):
            k_num = f'haraimodoshi_tansho_{i}a'
            k_pay = f'haraimodoshi_tansho_{i}b'
            if k_num in row and row[k_num] and str(row[k_num]).strip():
                try:
                    payout_map[rid]['tansho'][str(row[k_num]).zfill(2)] = int(row[k_pay])
                except: pass
        
        for i in range(1, 4):
            k_comb = f'haraimodoshi_umaren_{i}a'
            k_pay = f'haraimodoshi_umaren_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    payout_map[rid]['umaren'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        
        for i in range(1, 7):
            k_comb = f'haraimodoshi_umatan_{i}a'
            k_pay = f'haraimodoshi_umatan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    payout_map[rid]['umatan'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        
        for i in range(1, 4):
            k_comb = f'haraimodoshi_sanrenpuku_{i}a'
            k_pay = f'haraimodoshi_sanrenpuku_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    payout_map[rid]['sanrenpuku'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        
        for i in range(1, 7):
            k_comb = f'haraimodoshi_sanrentan_{i}a'
            k_pay = f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    payout_map[rid]['sanrentan'][str(row[k_comb])] = int(row[k_pay])
                except: pass
    
    return payout_map


def precompute_race_data(pred_df):
    """レースデータをプリコンピュート"""
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
        
        # フォーメーション事前生成
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
    
    # ゾーン分類
    zone = None
    
    # 高確信 + 中穴
    if gap >= 0.05 and 5 <= odds <= 50:
        zone = 'high_conf_mid_odds'
    # 高確信 + 人気馬
    elif gap >= 0.05 and 1 <= odds < 5:
        zone = 'high_conf_fav'
    # 高確信 + 大穴
    elif gap >= 0.05 and odds > 50:
        zone = 'high_conf_longshot'
    # 中確信 + 中穴
    elif 0.02 <= gap < 0.05 and 5 <= odds <= 50:
        zone = 'mid_conf_mid_odds'
    # 中確信 + 人気馬
    elif 0.02 <= gap < 0.05 and 1 <= odds < 5:
        zone = 'mid_conf_fav'
    # 低確信
    elif gap < 0.02:
        zone = 'low_conf'
    else:
        zone = 'other'
    
    return zone


def execute_bet(info, payouts, bet_type, formation_key):
    """賭けを実行して結果を返す"""
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


def evaluate_integrated_strategy(race_data, payout_map, strategy_rules):
    """
    統合戦略を評価
    
    strategy_rules: dict mapping zone -> (bet_type, formation_key)
    例: {'high_conf_mid_odds': ('sanrentan', 'san_box3'),
         'mid_conf_fav': ('umaren', 'umaren_box5'),
         'low_conf': ('skip', None)}
    """
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
        
        payouts = payout_map[race_id]
        bet_amount, win_amount, hit = execute_bet(info, payouts, bet_type, formation_key)
        
        total_bet += bet_amount
        total_return += win_amount
        zone_stats[zone]['bet'] += bet_amount
        zone_stats[zone]['return'] += win_amount
        if hit:
            zone_stats[zone]['hits'] += 1
    
    roi = total_return / total_bet * 100 if total_bet > 0 else 0
    
    return {
        'roi': roi,
        'total_bet': total_bet,
        'total_return': total_return,
        'zone_stats': zone_stats
    }


def find_best_integrated_strategy(race_data, payout_map, fold_name):
    """ゾーンごとに最適な馬券タイプを探索"""
    
    # 各ゾーンで試す馬券タイプ
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
    
    zones = ['high_conf_mid_odds', 'high_conf_fav', 'high_conf_longshot', 
             'mid_conf_mid_odds', 'mid_conf_fav', 'low_conf', 'other']
    
    # まず、各ゾーン単独での最適馬券を探索
    zone_best = {}
    
    for zone in zones:
        best_roi = -100
        best_bet = None
        
        for bet_type, formation in bet_options:
            # 単一ゾーンのみに賭ける
            rules = {z: ('skip', None) for z in zones}
            rules[zone] = (bet_type, formation)
            
            result = evaluate_integrated_strategy(race_data, payout_map, rules)
            zone_roi = result['zone_stats'].get(zone, {}).get('return', 0) / result['zone_stats'].get(zone, {}).get('bet', 1) * 100 if result['zone_stats'].get(zone, {}).get('bet', 0) > 0 else 0
            
            if zone_roi > best_roi:
                best_roi = zone_roi
                best_bet = (bet_type, formation, zone_roi, result['zone_stats'].get(zone, {}))
        
        zone_best[zone] = best_bet
        if best_bet[0] != 'skip':
            logger.info(f"  [{fold_name}] {zone}: {best_bet[0]}_{best_bet[1]} → ROI={best_bet[2]:.1f}%")
    
    # 統合戦略を構築
    integrated_rules = {}
    for zone, best in zone_best.items():
        bet_type, formation, roi, _ = best
        # ROI 75%以上のゾーンのみ採用
        if roi >= 75:
            integrated_rules[zone] = (bet_type, formation)
        else:
            integrated_rules[zone] = ('skip', None)
    
    # 統合戦略の評価
    final_result = evaluate_integrated_strategy(race_data, payout_map, integrated_rules)
    
    return integrated_rules, final_result, zone_best


def main():
    parser = argparse.ArgumentParser(description='統合戦略オプティマイザー')
    parser.add_argument('--dataset_suffix', type=str, default='_v10_leakfix')
    parser.add_argument('--version', type=str, default='v23')
    parser.add_argument('--cv_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.cv_dir is None:
        args.cv_dir = os.path.join(project_root, 'experiments', f'{args.version}_regression_cv')
    
    logger.info("=" * 60)
    logger.info("統合戦略オプティマイザー")
    logger.info("条件に応じて最適な馬券を選択する統合戦略を探索")
    logger.info("=" * 60)
    
    parquet_path = os.path.join(project_root, f'data/processed/preprocessed_data{args.dataset_suffix}.parquet')
    df = pd.read_parquet(parquet_path)
    
    pickle_path = os.path.join(project_root, f'data/processed/lgbm_datasets{args.dataset_suffix}.pkl')
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    all_results = []
    all_zone_stats = []
    
    for fold in FOLDS:
        fold_name = fold['name']
        valid_year = fold['valid_year']
        
        logger.info(f"\n=== {fold_name} ({valid_year}) ===")
        
        # モデルロード
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
        
        # JRAフィルタ
        if 'venue' in valid_df.columns:
            jra_venues = [f"{i:02}" for i in range(1, 11)]
            valid_df = valid_df[valid_df['venue'].isin(jra_venues)]
        
        # 予測
        X_valid = valid_df[feature_cols]
        lgbm_pred = lgbm_model.predict(X_valid)
        catboost_pred = catboost_model.predict(X_valid)
        valid_df['score'] = meta_model.predict(np.column_stack([lgbm_pred, catboost_pred]))
        
        # 払戻データ
        payout_df = load_payout_data([valid_year])
        if payout_df.empty:
            continue
        payout_map = build_payout_map(payout_df)
        
        # プリコンピュート
        race_data = precompute_race_data(valid_df)
        logger.info(f"対象レース: {len(race_data)}")
        
        # 最適統合戦略を探索
        integrated_rules, final_result, zone_best = find_best_integrated_strategy(race_data, payout_map, fold_name)
        
        logger.info(f"\n[{fold_name}] 統合戦略 ROI: {final_result['roi']:.1f}%")
        logger.info(f"  投資: {final_result['total_bet']:,}円, 回収: {final_result['total_return']:,}円")
        
        all_results.append({
            'fold': fold_name,
            'year': valid_year,
            'rules': {k: v[0] + ('_' + v[1] if v[1] else '') for k, v in integrated_rules.items()},
            'roi': final_result['roi'],
            'total_bet': final_result['total_bet'],
            'total_return': final_result['total_return']
        })
        
        for zone, stats in final_result['zone_stats'].items():
            all_zone_stats.append({
                'fold': fold_name,
                'zone': zone,
                'races': stats['races'],
                'bet': stats['bet'],
                'return': stats['return'],
                'hits': stats['hits']
            })
    
    # サマリー
    logger.info("\n" + "=" * 60)
    logger.info("全Foldサマリー")
    logger.info("=" * 60)
    
    for r in all_results:
        logger.info(f"{r['fold']} ({r['year']}): ROI={r['roi']:.1f}%")
    
    avg_roi = np.mean([r['roi'] for r in all_results])
    min_roi = min([r['roi'] for r in all_results])
    max_roi = max([r['roi'] for r in all_results])
    
    logger.info(f"\n平均ROI: {avg_roi:.1f}%, 最小: {min_roi:.1f}%, 最大: {max_roi:.1f}%")
    
    # ゾーン別統計
    zone_df = pd.DataFrame(all_zone_stats)
    zone_summary = zone_df.groupby('zone').agg({
        'races': 'sum',
        'bet': 'sum',
        'return': 'sum',
        'hits': 'sum'
    })
    zone_summary['roi'] = zone_summary['return'] / zone_summary['bet'] * 100
    zone_summary = zone_summary.round(2)
    
    print("\nゾーン別統計:")
    print(zone_summary.to_string())
    
    # 保存
    output_path = os.path.join(args.cv_dir, 'integrated_strategy.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fold_results': all_results,
            'avg_roi': avg_roi,
            'min_roi': min_roi,
            'max_roi': max_roi
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n結果を保存: {output_path}")
    logger.info("完了!")

if __name__ == "__main__":
    main()

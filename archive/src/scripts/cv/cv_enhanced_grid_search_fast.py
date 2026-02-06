"""
拡張版 戦略グリッドサーチ (最適化版)

ベクトル化とプリコンピュートにより高速化
"""
import os
import sys
import pickle
import json
import logging
import argparse
from datetime import datetime
from itertools import combinations, permutations

# パス設定
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
        logger.error(f"払戻データロードエラー: {e}")
        return pd.DataFrame()

def build_payout_map(payout_df):
    """払戻マップを構築"""
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'tansho': {}, 'umaren': {}, 'umatan': {}, 'sanrenpuku': {}, 'sanrentan': {}}
        
        for i in range(1, 4):
            k_num = f'haraimodoshi_tansho_{i}a'
            k_pay = f'haraimodoshi_tansho_{i}b'
            if k_num in row and row[k_num] and str(row[k_num]).strip():
                try:
                    num = str(row[k_num]).zfill(2)
                    payout_map[rid]['tansho'][num] = int(row[k_pay])
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
    """
    レースごとのデータをプリコンピュート
    ループを減らすため事前計算
    """
    race_data = {}
    
    # グループ化して一括処理
    for race_id, group in pred_df.groupby('race_id'):
        sorted_group = group.nlargest(6, 'score')
        top_horses = sorted_group['horse_number'].astype(int).tolist()
        top_scores = sorted_group['score'].tolist()
        
        if len(top_horses) < 3:
            continue
        
        race_info = {
            'horses': top_horses,
            'scores': top_scores,
            'score_gap': top_scores[0] - top_scores[1] if len(top_scores) >= 2 else 0,
            'top1_odds': sorted_group['odds'].iloc[0] if 'odds' in sorted_group.columns else None,
        }
        
        # フォーメーション事前生成
        race_info['formations'] = {
            'san_box3': list(permutations(top_horses[:3], 3)),
            'san_box5': list(permutations(top_horses[:5], 3)) if len(top_horses) >= 5 else [],
            'san_1st': [(top_horses[0], o1, o2) for o1, o2 in permutations(top_horses[1:5], 2)] if len(top_horses) >= 5 else [],
            'san_2nd': [(f, top_horses[0], t) for f in top_horses[1:4] for t in top_horses[1:5] if f != t] if len(top_horses) >= 5 else [],
            'san_3rd': [(o1, o2, top_horses[0]) for o1, o2 in permutations(top_horses[1:5], 2)] if len(top_horses) >= 5 else [],
            'san_f1_23_345': [(top_horses[0], s, t) for s in top_horses[1:3] for t in top_horses[2:5] if s != t],
            'uma_1st': [(top_horses[0], o) for o in top_horses[1:5]] if len(top_horses) >= 5 else [],
            'uma_2nd': [(o, top_horses[0]) for o in top_horses[1:4]] if len(top_horses) >= 4 else [],
            'uma_box3': list(permutations(top_horses[:3], 2)),
        }
        
        # 三連複用
        race_info['combos'] = {
            'sanrenpuku_box3': list(combinations(top_horses[:3], 3)),
            'sanrenpuku_box5': list(combinations(top_horses[:5], 3)) if len(top_horses) >= 5 else [],
            'umaren_box3': list(combinations(top_horses[:3], 2)),
            'umaren_box5': list(combinations(top_horses[:5], 2)) if len(top_horses) >= 5 else [],
        }
        
        race_data[race_id] = race_info
    
    return race_data


def evaluate_fast(race_data, payout_map, bet_type, formation_key, skip_cond):
    """高速評価"""
    total_bet = 0
    total_return = 0
    hits = 0
    races = 0
    skipped = 0
    
    for race_id, info in race_data.items():
        if race_id not in payout_map:
            continue
        
        # スキップ判定
        should_skip = False
        
        if skip_cond.get('min_score_gap') is not None:
            if info['score_gap'] < skip_cond['min_score_gap']:
                should_skip = True
        
        top1_odds = info['top1_odds']
        if top1_odds is not None and not pd.isna(top1_odds):
            if skip_cond.get('max_top1_odds') is not None and top1_odds > skip_cond['max_top1_odds']:
                should_skip = True
            if skip_cond.get('min_top1_odds') is not None and top1_odds < skip_cond['min_top1_odds']:
                should_skip = True
        
        if should_skip:
            skipped += 1
            continue
        
        races += 1
        payouts = payout_map[race_id]
        
        if bet_type == 'sanrentan':
            formations = info['formations'].get(formation_key, [])
            total_bet += len(formations) * 100
            for p in formations:
                key = f"{p[0]:02}{p[1]:02}{p[2]:02}"
                if key in payouts['sanrentan']:
                    total_return += payouts['sanrentan'][key]
                    hits += 1
        
        elif bet_type == 'sanrenpuku':
            combos = info['combos'].get(formation_key, [])
            total_bet += len(combos) * 100
            for c in combos:
                c_sorted = sorted(c)
                key = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
                if key in payouts['sanrenpuku']:
                    total_return += payouts['sanrenpuku'][key]
                    hits += 1
        
        elif bet_type == 'umatan':
            formations = info['formations'].get(formation_key, [])
            total_bet += len(formations) * 100
            for p in formations:
                key = f"{p[0]:02}{p[1]:02}"
                if key in payouts['umatan']:
                    total_return += payouts['umatan'][key]
                    hits += 1
        
        elif bet_type == 'umaren':
            combos = info['combos'].get(formation_key, [])
            total_bet += len(combos) * 100
            for c in combos:
                c_sorted = sorted(c)
                key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
                if key in payouts['umaren']:
                    total_return += payouts['umaren'][key]
                    hits += 1
        
        elif bet_type == 'tansho':
            total_bet += 100
            key = f"{info['horses'][0]:02}"
            if key in payouts['tansho']:
                total_return += payouts['tansho'][key]
                hits += 1
    
    roi = total_return / total_bet * 100 if total_bet > 0 else 0
    hit_rate = hits / races if races > 0 else 0
    
    return {
        'roi': roi,
        'hit_rate': hit_rate,
        'total_bet': total_bet,
        'total_return': total_return,
        'races': races,
        'skipped': skipped
    }


def run_fast_grid_search(race_data, payout_map, fold_name):
    """高速グリッドサーチ"""
    results = []
    
    # 買い方定義
    bet_formations = [
        ('sanrentan', 'san_box3', '三連単Box3'),
        ('sanrentan', 'san_box5', '三連単Box5'),
        ('sanrentan', 'san_1st', '三連単1着固定'),
        ('sanrentan', 'san_2nd', '三連単2着固定'),
        ('sanrentan', 'san_3rd', '三連単3着固定'),
        ('sanrentan', 'san_f1_23_345', '三連単F_1-23-345'),
        ('sanrenpuku', 'sanrenpuku_box3', '三連複Box3'),
        ('sanrenpuku', 'sanrenpuku_box5', '三連複Box5'),
        ('umatan', 'uma_1st', '馬単1着固定'),
        ('umatan', 'uma_2nd', '馬単2着固定'),
        ('umatan', 'uma_box3', '馬単Box3'),
        ('umaren', 'umaren_box3', '馬連Box3'),
        ('umaren', 'umaren_box5', '馬連Box5'),
        ('tansho', 'top1', '単勝'),
    ]
    
    # スキップ条件
    skip_conditions = [
        ('none', {}),
        ('gap_002', {'min_score_gap': 0.02}),
        ('gap_005', {'min_score_gap': 0.05}),
        ('gap_010', {'min_score_gap': 0.10}),
        ('odds_3_30', {'min_top1_odds': 3.0, 'max_top1_odds': 30.0}),
        ('odds_5_50', {'min_top1_odds': 5.0, 'max_top1_odds': 50.0}),
        ('gap_005_odds_5_50', {'min_score_gap': 0.05, 'min_top1_odds': 5.0, 'max_top1_odds': 50.0}),
    ]
    
    logger.info(f"[{fold_name}] 評価中: {len(bet_formations)}買い方 x {len(skip_conditions)}条件")
    
    for bet_type, formation_key, name in bet_formations:
        for skip_name, skip_cond in skip_conditions:
            strategy_name = f"{name}_{skip_name}"
            result = evaluate_fast(race_data, payout_map, bet_type, formation_key, skip_cond)
            result['strategy'] = strategy_name
            result['bet_type'] = bet_type
            result['formation'] = formation_key
            result['skip_condition'] = skip_name
            result['fold'] = fold_name
            results.append(result)
            
            if result['roi'] > 90:
                logger.info(f"  ★ {strategy_name}: ROI={result['roi']:.1f}%, Hit={result['hit_rate']:.2%}, Bet={result['races']}R")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='拡張戦略グリッドサーチ (最適化版)')
    parser.add_argument('--dataset_suffix', type=str, default='_v10_leakfix')
    parser.add_argument('--version', type=str, default='v23')
    parser.add_argument('--cv_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.cv_dir is None:
        args.cv_dir = os.path.join(project_root, 'experiments', f'{args.version}_regression_cv')
    
    logger.info("=" * 60)
    logger.info("拡張戦略グリッドサーチ (最適化版)")
    logger.info("=" * 60)
    
    parquet_path = os.path.join(project_root, f'data/processed/preprocessed_data{args.dataset_suffix}.parquet')
    logger.info(f"データロード中: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    pickle_path = os.path.join(project_root, f'data/processed/lgbm_datasets{args.dataset_suffix}.pkl')
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    all_results = []
    
    for fold in FOLDS:
        fold_name = fold['name']
        valid_year = fold['valid_year']
        
        logger.info(f"\n=== {fold_name} ({valid_year}) ===")
        
        # モデルロード
        lgbm_path = os.path.join(args.cv_dir, fold_name, f'lgbm_{args.version}.pkl')
        catboost_path = os.path.join(args.cv_dir, fold_name, f'catboost_{args.version}.pkl')
        meta_path = os.path.join(args.cv_dir, fold_name, f'meta_{args.version}.pkl')
        
        if not os.path.exists(meta_path):
            continue
        
        with open(lgbm_path, 'rb') as f:
            lgbm_model = pickle.load(f)
        with open(catboost_path, 'rb') as f:
            catboost_model = pickle.load(f)
        with open(meta_path, 'rb') as f:
            meta_model = pickle.load(f)
        
        # 検証データ
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
        meta_X = np.column_stack([lgbm_pred, catboost_pred])
        valid_df['score'] = meta_model.predict(meta_X)
        
        logger.info(f"データ: {len(valid_df)} レコード")
        
        # 払戻データ
        payout_df = load_payout_data([valid_year])
        if payout_df.empty:
            continue
        payout_map = build_payout_map(payout_df)
        
        # プリコンピュート
        logger.info("レースデータをプリコンピュート中...")
        race_data = precompute_race_data(valid_df)
        logger.info(f"対象レース: {len(race_data)}")
        
        # グリッドサーチ
        results = run_fast_grid_search(race_data, payout_map, fold_name)
        all_results.extend(results)
    
    # 結果集計
    logger.info("\n" + "=" * 60)
    logger.info("全Fold結果サマリー")
    logger.info("=" * 60)
    
    results_df = pd.DataFrame(all_results)
    
    # 戦略ごとの統計
    summary = results_df.groupby('strategy').agg({
        'roi': ['mean', 'std', 'min', 'max'],
        'hit_rate': 'mean',
        'races': 'mean'
    }).round(2)
    
    summary_sorted = summary.sort_values(('roi', 'mean'), ascending=False)
    
    print("\n戦略別平均ROI (上位30):")
    print(summary_sorted.head(30).to_string())
    
    # 一貫して良好
    consistent = []
    for strategy in results_df['strategy'].unique():
        strat_df = results_df[results_df['strategy'] == strategy]
        min_roi = strat_df['roi'].min()
        avg_roi = strat_df['roi'].mean()
        if min_roi >= 80:
            consistent.append({
                'strategy': strategy,
                'avg_roi': avg_roi,
                'min_roi': min_roi,
                'max_roi': strat_df['roi'].max(),
                'avg_hit_rate': strat_df['hit_rate'].mean(),
                'avg_races': strat_df['races'].mean()
            })
    
    logger.info(f"\n=== 一貫して良好 (全Fold ROI >= 80%): {len(consistent)} 件 ===")
    for s in sorted(consistent, key=lambda x: -x['avg_roi'])[:20]:
        logger.info(f"  {s['strategy']}: 平均={s['avg_roi']:.1f}%, 最小={s['min_roi']:.1f}%, 的中={s['avg_hit_rate']:.2%}, {s['avg_races']:.0f}R/年")
    
    # 保存
    output_path = os.path.join(args.cv_dir, 'enhanced_grid_search.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': all_results,
            'consistent_strategies': consistent
        }, f, indent=2)
    
    logger.info(f"\n結果を保存: {output_path}")
    logger.info("完了!")

if __name__ == "__main__":
    main()

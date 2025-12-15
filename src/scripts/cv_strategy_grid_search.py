"""
時系列CV用 戦略グリッドサーチ

各Foldで戦略ROIを評価し、全Foldで一貫して良い戦略を特定する。
最終的に2025年でバックテストを行う。
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
from scipy.special import softmax
from sqlalchemy import create_engine, text

from model.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fold定義
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
    """払戻データをロード"""
    logger.info(f"払戻データ(jvd_hr)をロード中... Years={years}")
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
        logger.info(f"払戻データロード完了: {len(df)} 件")
        return df
    except Exception as e:
        logger.error(f"払戻データロードエラー: {e}")
        return pd.DataFrame()

def build_payout_map(payout_df):
    """払戻マップを構築"""
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'tansho': {}, 'umaren': {}, 'sanrenpuku': {}, 'sanrentan': {}}
        
        # 単勝
        for i in range(1, 4):
            k_num = f'haraimodoshi_tansho_{i}a'
            k_pay = f'haraimodoshi_tansho_{i}b'
            if k_num in row and row[k_num] and str(row[k_num]).strip():
                try:
                    num = str(row[k_num]).zfill(2)
                    pay = int(row[k_pay])
                    payout_map[rid]['tansho'][num] = pay
                except: pass
        
        # 馬連
        for i in range(1, 4):
            k_comb = f'haraimodoshi_umaren_{i}a'
            k_pay = f'haraimodoshi_umaren_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['umaren'][str(row[k_comb])] = pay
                except: pass
        
        # 三連複
        for i in range(1, 4):
            k_comb = f'haraimodoshi_sanrenpuku_{i}a'
            k_pay = f'haraimodoshi_sanrenpuku_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['sanrenpuku'][str(row[k_comb])] = pay
                except: pass
        
        # 三連単
        for i in range(1, 7):
            k_comb = f'haraimodoshi_sanrentan_{i}a'
            k_pay = f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['sanrentan'][str(row[k_comb])] = pay
                except: pass
    
    return payout_map

def evaluate_strategy(pred_df, payout_map, strategy_name, top_n=5, bet_type='sanrentan'):
    """戦略を評価してROIを計算"""
    total_bet = 0
    total_return = 0
    hits = 0
    races = 0
    
    for race_id, group in pred_df.groupby('race_id'):
        if race_id not in payout_map:
            continue
        
        # スコア上位N頭を取得
        top_horses = group.nlargest(top_n, 'score')
        horse_nums = top_horses['horse_number'].astype(int).tolist()
        
        if len(horse_nums) < 3:
            continue
        
        races += 1
        
        if bet_type == 'sanrentan_box':
            # 三連単ボックス
            perms = list(permutations(horse_nums[:top_n], 3))
            bet_count = len(perms)
            total_bet += bet_count * 100
            
            for p in perms:
                perm_str = f"{p[0]:02}{p[1]:02}{p[2]:02}"
                if perm_str in payout_map[race_id]['sanrentan']:
                    total_return += payout_map[race_id]['sanrentan'][perm_str]
                    hits += 1
        
        elif bet_type == 'sanrenpuku_box':
            # 三連複ボックス
            combs = list(combinations(horse_nums[:top_n], 3))
            bet_count = len(combs)
            total_bet += bet_count * 100
            
            for c in combs:
                c_sorted = sorted(c)
                comb_str = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
                if comb_str in payout_map[race_id]['sanrenpuku']:
                    total_return += payout_map[race_id]['sanrenpuku'][comb_str]
                    hits += 1
        
        elif bet_type == 'sanrentan_nagashi':
            # 三連単流し (1着軸)
            if len(horse_nums) < 2:
                continue
            axis = horse_nums[0]
            opps = horse_nums[1:top_n]
            opp_perms = list(permutations(opps, 2))
            bet_count = len(opp_perms)
            total_bet += bet_count * 100
            
            for o1, o2 in opp_perms:
                perm_str = f"{axis:02}{o1:02}{o2:02}"
                if perm_str in payout_map[race_id]['sanrentan']:
                    total_return += payout_map[race_id]['sanrentan'][perm_str]
                    hits += 1
        
        elif bet_type == 'umaren_box':
            # 馬連ボックス
            combs = list(combinations(horse_nums[:top_n], 2))
            bet_count = len(combs)
            total_bet += bet_count * 100
            
            for c in combs:
                c_sorted = sorted(c)
                comb_str = f"{c_sorted[0]:02}{c_sorted[1]:02}"
                if comb_str in payout_map[race_id]['umaren']:
                    total_return += payout_map[race_id]['umaren'][comb_str]
                    hits += 1
        
        elif bet_type == 'tansho':
            # 単勝 (Top1)
            top1 = horse_nums[0]
            total_bet += 100
            num_str = f"{top1:02}"
            if num_str in payout_map[race_id]['tansho']:
                total_return += payout_map[race_id]['tansho'][num_str]
                hits += 1
    
    roi = total_return / total_bet * 100 if total_bet > 0 else 0
    hit_rate = hits / races if races > 0 else 0
    
    return {
        'strategy': strategy_name,
        'bet_type': bet_type,
        'top_n': top_n,
        'roi': roi,
        'hit_rate': hit_rate,
        'total_bet': total_bet,
        'total_return': total_return,
        'races': races,
        'hits': hits
    }

def run_grid_search(pred_df, payout_map, fold_name):
    """グリッドサーチを実行"""
    results = []
    
    # JRAレースのみにフィルタ (競馬場 01-10)
    if 'venue' in pred_df.columns:
        jra_venues = [f"{i:02}" for i in range(1, 11)]
        pred_df_jra = pred_df[pred_df['venue'].isin(jra_venues)].copy()
        logger.info(f"[{fold_name}] JRAフィルタ: {len(pred_df)} → {len(pred_df_jra)} レコード")
    else:
        # venueがない場合はrace_idから判定
        pred_df_jra = pred_df.copy()
        logger.warning(f"[{fold_name}] venue列がありません、全データを使用")
    
    # 基本戦略定義
    base_strategies = [
        ('tansho_top1', 'tansho', 1),
        ('umaren_box3', 'umaren_box', 3),
        ('umaren_box5', 'umaren_box', 5),
        ('sanrenpuku_box3', 'sanrenpuku_box', 3),
        ('sanrenpuku_box5', 'sanrenpuku_box', 5),
        ('sanrentan_box3', 'sanrentan_box', 3),
        ('sanrentan_box5', 'sanrentan_box', 5),
        ('sanrentan_nagashi5', 'sanrentan_nagashi', 5),
        ('sanrentan_nagashi6', 'sanrentan_nagashi', 6),
    ]
    
    # 閾値条件定義
    threshold_conditions = [
        {'name': 'all', 'min_odds': None, 'max_odds': None, 'min_score_diff': None},
        {'name': 'odds_3_30', 'min_odds': 3.0, 'max_odds': 30.0, 'min_score_diff': None},
        {'name': 'odds_5_50', 'min_odds': 5.0, 'max_odds': 50.0, 'min_score_diff': None},
        {'name': 'score_diff_02', 'min_odds': None, 'max_odds': None, 'min_score_diff': 0.2},
        {'name': 'score_diff_05', 'min_odds': None, 'max_odds': None, 'min_score_diff': 0.5},
    ]
    
    for cond in threshold_conditions:
        # 条件でフィルタ
        filtered_df = pred_df_jra.copy()
        
        # Top1のオッズでフィルタ
        if cond['min_odds'] is not None or cond['max_odds'] is not None:
            def filter_by_top1_odds(group, min_o, max_o):
                top1 = group.nlargest(1, 'score')
                if top1.empty:
                    return False
                odds = top1['odds'].values[0]
                if pd.isna(odds):
                    return False
                if min_o is not None and odds < min_o:
                    return False
                if max_o is not None and odds > max_o:
                    return False
                return True
            
            valid_races = []
            for race_id, group in filtered_df.groupby('race_id'):
                if filter_by_top1_odds(group, cond['min_odds'], cond['max_odds']):
                    valid_races.append(race_id)
            filtered_df = filtered_df[filtered_df['race_id'].isin(valid_races)]
        
        # スコア差でフィルタ (Top1とTop2の差が閾値以上)
        if cond['min_score_diff'] is not None:
            def filter_by_score_diff(group, min_diff):
                top2 = group.nlargest(2, 'score')
                if len(top2) < 2:
                    return False
                diff = top2['score'].iloc[0] - top2['score'].iloc[1]
                return diff >= min_diff
            
            valid_races = []
            for race_id, group in filtered_df.groupby('race_id'):
                if filter_by_score_diff(group, cond['min_score_diff']):
                    valid_races.append(race_id)
            filtered_df = filtered_df[filtered_df['race_id'].isin(valid_races)]
        
        cond_name = cond['name']
        n_races = filtered_df['race_id'].nunique()
        
        if n_races < 10:
            logger.info(f"[{fold_name}] 条件 {cond_name}: レース数が少なすぎます ({n_races})")
            continue
        
        logger.info(f"[{fold_name}] 条件 {cond_name}: {n_races} レース")
        
        for name, bet_type, top_n in base_strategies:
            strategy_name = f"{name}_{cond_name}"
            result = evaluate_strategy(filtered_df, payout_map, strategy_name, top_n, bet_type)
            result['fold'] = fold_name
            result['condition'] = cond_name
            results.append(result)
            
            if result['roi'] > 50:  # 注目すべき結果のみ表示
                logger.info(f"  ★ {strategy_name}: ROI={result['roi']:.1f}%, Hit={result['hit_rate']:.2%}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='時系列CV戦略グリッドサーチ')
    parser.add_argument('--dataset_suffix', type=str, default='_v10_leakfix')
    parser.add_argument('--version', type=str, default='v22')
    parser.add_argument('--cv_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.cv_dir is None:
        args.cv_dir = os.path.join(project_root, 'experiments', f'{args.version}_timeseries_cv')
    
    logger.info("=" * 60)
    logger.info("時系列CV 戦略グリッドサーチ開始")
    logger.info("=" * 60)
    
    # データロード
    parquet_path = os.path.join(project_root, f'data/processed/preprocessed_data{args.dataset_suffix}.parquet')
    logger.info(f"前処理済みデータをロード中: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # 特徴量リスト
    pickle_path = os.path.join(project_root, f'data/processed/lgbm_datasets{args.dataset_suffix}.pkl')
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    all_results = []
    
    for fold in FOLDS:
        fold_name = fold['name']
        valid_year = fold['valid_year']
        
        logger.info(f"\n=== {fold_name} ({valid_year}) 評価中 ===")
        
        # モデルロード
        ensemble_path = os.path.join(args.cv_dir, fold_name, f'ensemble_{args.version}.pkl')
        if not os.path.exists(ensemble_path):
            logger.warning(f"モデルが見つかりません: {ensemble_path}")
            continue
        
        ensemble = EnsembleModel()
        ensemble.load_model(ensemble_path)
        
        # 検証データ
        valid_df = df[df['year'] == valid_year].copy()
        for col in feature_cols:
            if col not in valid_df.columns:
                valid_df[col] = 0
        
        # 予測
        X_valid = valid_df[feature_cols]
        scores = ensemble.predict(X_valid)
        valid_df['score'] = scores
        
        # 払戻データ
        payout_df = load_payout_data([valid_year])
        if payout_df.empty:
            logger.warning(f"{valid_year}年の払戻データがありません")
            continue
        
        payout_map = build_payout_map(payout_df)
        
        # グリッドサーチ
        results = run_grid_search(valid_df, payout_map, fold_name)
        all_results.extend(results)
    
    # 結果を集計
    logger.info("\n" + "=" * 60)
    logger.info("全Fold結果サマリー")
    logger.info("=" * 60)
    
    results_df = pd.DataFrame(all_results)
    
    # 戦略ごとにFold平均を計算
    summary = results_df.groupby('strategy').agg({
        'roi': ['mean', 'std', 'min', 'max'],
        'hit_rate': 'mean'
    }).round(2)
    
    print("\n戦略別平均ROI:")
    print(summary.to_string())
    
    # 全Foldで一貫してROI > 100%の戦略を特定
    consistent_strategies = []
    for strategy in results_df['strategy'].unique():
        strat_df = results_df[results_df['strategy'] == strategy]
        min_roi = strat_df['roi'].min()
        avg_roi = strat_df['roi'].mean()
        if min_roi >= 80:  # 全Foldで80%以上
            consistent_strategies.append({
                'strategy': strategy,
                'avg_roi': avg_roi,
                'min_roi': min_roi,
                'max_roi': strat_df['roi'].max()
            })
    
    logger.info(f"\n一貫して良好な戦略 (全Fold ROI >= 80%): {len(consistent_strategies)} 件")
    for s in sorted(consistent_strategies, key=lambda x: -x['avg_roi']):
        logger.info(f"  {s['strategy']}: 平均={s['avg_roi']:.1f}%, 最小={s['min_roi']:.1f}%, 最大={s['max_roi']:.1f}%")
    
    # 結果保存
    output_path = os.path.join(args.cv_dir, 'strategy_grid_search.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': all_results,
            'consistent_strategies': consistent_strategies
        }, f, indent=2)
    
    logger.info(f"\n結果を保存: {output_path}")
    logger.info("グリッドサーチ完了!")

if __name__ == "__main__":
    main()

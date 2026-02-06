"""
拡張版 戦略グリッドサーチ (v23回帰モデル)

新機能:
1. スキップ条件 - レースを買わない条件を探索
2. フォーメーション - 1着固定、2着固定、3着固定など
"""
import os
import sys
import pickle
import json
import logging
import argparse
from datetime import datetime
from itertools import combinations, permutations, product

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
        
        # 馬単
        for i in range(1, 7):
            k_comb = f'haraimodoshi_umatan_{i}a'
            k_pay = f'haraimodoshi_umatan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['umatan'][str(row[k_comb])] = pay
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


def generate_formations(top_horses, formation_type):
    """
    フォーメーションを生成
    
    formation_type:
    - 'box3': 3頭ボックス
    - 'box5': 5頭ボックス
    - 'nagashi_1st': 1着固定流し (Top1軸)
    - 'nagashi_12': 1-2着固定 (Top1-Top2軸)
    - 'form_1_23_345': 1着Top1, 2着Top2-3, 3着Top3-5
    - 'form_12_123_12345': 1着Top1-2, 2着Top1-3, 3着Top1-5
    """
    formations = []
    
    if len(top_horses) < 3:
        return formations
    
    if formation_type == 'box3':
        formations = list(permutations(top_horses[:3], 3))
    
    elif formation_type == 'box5':
        formations = list(permutations(top_horses[:5], 3))
    
    elif formation_type == 'nagashi_1st':
        # 1着固定: Top1を1着に、2-3着はTop2-5から
        axis = top_horses[0]
        others = top_horses[1:5]
        for o1, o2 in permutations(others, 2):
            formations.append((axis, o1, o2))
    
    elif formation_type == 'nagashi_2nd':
        # 2着固定: Top1を2着に、1着はTop2-4, 3着はTop2-5
        axis = top_horses[0]
        first_cands = top_horses[1:4]
        third_cands = top_horses[1:5]
        for f in first_cands:
            for t in third_cands:
                if f != t:
                    formations.append((f, axis, t))
    
    elif formation_type == 'nagashi_3rd':
        # 3着固定: Top1を3着に、1-2着はTop2-5から
        axis = top_horses[0]
        others = top_horses[1:5]
        for o1, o2 in permutations(others, 2):
            formations.append((o1, o2, axis))
    
    elif formation_type == 'form_1_23_345':
        # 1着Top1, 2着Top2-3, 3着Top3-5
        first = [top_horses[0]]
        second = top_horses[1:3]
        third = top_horses[2:5]
        for f in first:
            for s in second:
                for t in third:
                    if f != s and s != t and f != t:
                        formations.append((f, s, t))
    
    elif formation_type == 'form_12_23_345':
        # 1着Top1-2, 2着Top2-3, 3着Top3-5
        first = top_horses[:2]
        second = top_horses[1:3]
        third = top_horses[2:5]
        for f in first:
            for s in second:
                for t in third:
                    if f != s and s != t and f != t:
                        formations.append((f, s, t))
    
    elif formation_type == 'form_1_234_2345':
        # 1着Top1, 2着Top2-4, 3着Top2-5
        first = [top_horses[0]]
        second = top_horses[1:4]
        third = top_horses[1:5]
        for f in first:
            for s in second:
                for t in third:
                    if f != s and s != t and f != t:
                        formations.append((f, s, t))
    
    return formations


def generate_umatan_formations(top_horses, formation_type):
    """馬単フォーメーション"""
    formations = []
    
    if len(top_horses) < 2:
        return formations
    
    if formation_type == '1st_fix':
        # 1着固定: Top1を1着に
        axis = top_horses[0]
        for o in top_horses[1:5]:
            formations.append((axis, o))
    
    elif formation_type == '2nd_fix':
        # 2着固定: Top1を2着に
        axis = top_horses[0]
        for o in top_horses[1:4]:
            formations.append((o, axis))
    
    elif formation_type == 'box3':
        for h1, h2 in permutations(top_horses[:3], 2):
            formations.append((h1, h2))
    
    elif formation_type == 'box5':
        for h1, h2 in permutations(top_horses[:5], 2):
            formations.append((h1, h2))
    
    return formations


def evaluate_strategy_with_skip(pred_df, payout_map, bet_type, formation_type, skip_condition):
    """
    スキップ条件付きで戦略を評価
    
    skip_condition: dict with keys:
    - min_top1_score: Top1のスコアがこれ未満ならスキップ
    - min_score_gap: Top1-Top2のスコア差がこれ未満ならスキップ
    - max_top1_odds: Top1のオッズがこれ超ならスキップ
    - min_top1_odds: Top1のオッズがこれ未満ならスキップ
    """
    total_bet = 0
    total_return = 0
    hits = 0
    races = 0
    skipped = 0
    
    for race_id, group in pred_df.groupby('race_id'):
        if race_id not in payout_map:
            continue
        
        # スコア順にソート
        group_sorted = group.nlargest(6, 'score')
        top_horses = group_sorted['horse_number'].astype(int).tolist()
        top_scores = group_sorted['score'].tolist()
        
        if len(top_horses) < 3:
            continue
        
        # スキップ条件チェック
        should_skip = False
        
        if skip_condition.get('min_top1_score') is not None:
            if top_scores[0] < skip_condition['min_top1_score']:
                should_skip = True
        
        if skip_condition.get('min_score_gap') is not None and len(top_scores) >= 2:
            gap = top_scores[0] - top_scores[1]
            if gap < skip_condition['min_score_gap']:
                should_skip = True
        
        if 'odds' in group_sorted.columns:
            top1_odds = group_sorted['odds'].iloc[0]
            if not pd.isna(top1_odds):
                if skip_condition.get('max_top1_odds') is not None:
                    if top1_odds > skip_condition['max_top1_odds']:
                        should_skip = True
                if skip_condition.get('min_top1_odds') is not None:
                    if top1_odds < skip_condition['min_top1_odds']:
                        should_skip = True
        
        if should_skip:
            skipped += 1
            continue
        
        races += 1
        
        # 三連単
        if bet_type == 'sanrentan':
            formations = generate_formations(top_horses, formation_type)
            bet_count = len(formations)
            total_bet += bet_count * 100
            
            for p in formations:
                perm_str = f"{p[0]:02}{p[1]:02}{p[2]:02}"
                if perm_str in payout_map[race_id]['sanrentan']:
                    total_return += payout_map[race_id]['sanrentan'][perm_str]
                    hits += 1
        
        # 三連複
        elif bet_type == 'sanrenpuku':
            if formation_type in ['box3', 'box5']:
                n = 3 if formation_type == 'box3' else 5
                combs = list(combinations(top_horses[:n], 3))
            else:
                # フォーメーションからユニークな組み合わせを抽出
                formations = generate_formations(top_horses, formation_type)
                combs = list(set([tuple(sorted(f)) for f in formations]))
            
            bet_count = len(combs)
            total_bet += bet_count * 100
            
            for c in combs:
                c_sorted = sorted(c)
                comb_str = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
                if comb_str in payout_map[race_id]['sanrenpuku']:
                    total_return += payout_map[race_id]['sanrenpuku'][comb_str]
                    hits += 1
        
        # 馬単
        elif bet_type == 'umatan':
            formations = generate_umatan_formations(top_horses, formation_type)
            bet_count = len(formations)
            total_bet += bet_count * 100
            
            for p in formations:
                perm_str = f"{p[0]:02}{p[1]:02}"
                if perm_str in payout_map[race_id]['umatan']:
                    total_return += payout_map[race_id]['umatan'][perm_str]
                    hits += 1
        
        # 馬連
        elif bet_type == 'umaren':
            if formation_type in ['box3', 'box5']:
                n = 3 if formation_type == 'box3' else 5
                combs = list(combinations(top_horses[:n], 2))
            else:
                formations = generate_umatan_formations(top_horses, formation_type)
                combs = list(set([tuple(sorted(f)) for f in formations]))
            
            bet_count = len(combs)
            total_bet += bet_count * 100
            
            for c in combs:
                c_sorted = sorted(c)
                comb_str = f"{c_sorted[0]:02}{c_sorted[1]:02}"
                if comb_str in payout_map[race_id]['umaren']:
                    total_return += payout_map[race_id]['umaren'][comb_str]
                    hits += 1
        
        # 単勝
        elif bet_type == 'tansho':
            top1 = top_horses[0]
            total_bet += 100
            num_str = f"{top1:02}"
            if num_str in payout_map[race_id]['tansho']:
                total_return += payout_map[race_id]['tansho'][num_str]
                hits += 1
    
    roi = total_return / total_bet * 100 if total_bet > 0 else 0
    hit_rate = hits / races if races > 0 else 0
    
    return {
        'roi': roi,
        'hit_rate': hit_rate,
        'total_bet': total_bet,
        'total_return': total_return,
        'races': races,
        'hits': hits,
        'skipped': skipped
    }


def run_enhanced_grid_search(pred_df, payout_map, fold_name):
    """拡張グリッドサーチ"""
    results = []
    
    # JRAフィルタ
    if 'venue' in pred_df.columns:
        jra_venues = [f"{i:02}" for i in range(1, 11)]
        pred_df_jra = pred_df[pred_df['venue'].isin(jra_venues)].copy()
        logger.info(f"[{fold_name}] JRAフィルタ: {len(pred_df)} → {len(pred_df_jra)} レコード")
    else:
        pred_df_jra = pred_df.copy()
    
    # 買い方定義
    bet_formations = [
        # 三連単
        ('sanrentan', 'box3', '三連単Box3'),
        ('sanrentan', 'box5', '三連単Box5'),
        ('sanrentan', 'nagashi_1st', '三連単1着固定'),
        ('sanrentan', 'nagashi_2nd', '三連単2着固定'),
        ('sanrentan', 'nagashi_3rd', '三連単3着固定'),
        ('sanrentan', 'form_1_23_345', '三連単F_1-23-345'),
        ('sanrentan', 'form_12_23_345', '三連単F_12-23-345'),
        ('sanrentan', 'form_1_234_2345', '三連単F_1-234-2345'),
        # 三連複
        ('sanrenpuku', 'box3', '三連複Box3'),
        ('sanrenpuku', 'box5', '三連複Box5'),
        # 馬単
        ('umatan', '1st_fix', '馬単1着固定'),
        ('umatan', '2nd_fix', '馬単2着固定'),
        ('umatan', 'box3', '馬単Box3'),
        # 馬連
        ('umaren', 'box3', '馬連Box3'),
        ('umaren', 'box5', '馬連Box5'),
        # 単勝
        ('tansho', 'top1', '単勝'),
    ]
    
    # スキップ条件定義
    skip_conditions = [
        ({'name': 'none'}, {}),
        ({'name': 'gap_002'}, {'min_score_gap': 0.02}),
        ({'name': 'gap_005'}, {'min_score_gap': 0.05}),
        ({'name': 'gap_010'}, {'min_score_gap': 0.10}),
        ({'name': 'odds_3_30'}, {'min_top1_odds': 3.0, 'max_top1_odds': 30.0}),
        ({'name': 'odds_5_50'}, {'min_top1_odds': 5.0, 'max_top1_odds': 50.0}),
        ({'name': 'gap_005_odds_5_50'}, {'min_score_gap': 0.05, 'min_top1_odds': 5.0, 'max_top1_odds': 50.0}),
    ]
    
    total_combos = len(bet_formations) * len(skip_conditions)
    logger.info(f"[{fold_name}] 評価中: {len(bet_formations)}買い方 x {len(skip_conditions)}条件 = {total_combos}パターン")
    
    for bet_type, formation, name in bet_formations:
        for skip_meta, skip_cond in skip_conditions:
            skip_name = skip_meta['name']
            strategy_name = f"{name}_{skip_name}"
            
            result = evaluate_strategy_with_skip(pred_df_jra, payout_map, bet_type, formation, skip_cond)
            result['strategy'] = strategy_name
            result['bet_type'] = bet_type
            result['formation'] = formation
            result['skip_condition'] = skip_name
            result['fold'] = fold_name
            results.append(result)
            
            if result['roi'] > 90:
                logger.info(f"  ★ {strategy_name}: ROI={result['roi']:.1f}%, Hit={result['hit_rate']:.2%}, Bet={result['races']}R")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='拡張戦略グリッドサーチ')
    parser.add_argument('--dataset_suffix', type=str, default='_v10_leakfix')
    parser.add_argument('--version', type=str, default='v23')
    parser.add_argument('--cv_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.cv_dir is None:
        args.cv_dir = os.path.join(project_root, 'experiments', f'{args.version}_regression_cv')
    
    logger.info("=" * 60)
    logger.info("拡張戦略グリッドサーチ")
    logger.info("- スキップ条件あり")
    logger.info("- フォーメーション対応")
    logger.info("=" * 60)
    
    # データロード
    parquet_path = os.path.join(project_root, f'data/processed/preprocessed_data{args.dataset_suffix}.parquet')
    logger.info(f"前処理済みデータをロード中: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
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
        lgbm_path = os.path.join(args.cv_dir, fold_name, f'lgbm_{args.version}.pkl')
        catboost_path = os.path.join(args.cv_dir, fold_name, f'catboost_{args.version}.pkl')
        meta_path = os.path.join(args.cv_dir, fold_name, f'meta_{args.version}.pkl')
        
        if not os.path.exists(meta_path):
            logger.warning(f"モデルが見つかりません: {meta_path}")
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
        
        # 予測
        X_valid = valid_df[feature_cols]
        lgbm_pred = lgbm_model.predict(X_valid)
        catboost_pred = catboost_model.predict(X_valid)
        meta_X = np.column_stack([lgbm_pred, catboost_pred])
        scores = meta_model.predict(meta_X)
        valid_df['score'] = scores
        
        # 払戻データ
        payout_df = load_payout_data([valid_year])
        if payout_df.empty:
            continue
        
        payout_map = build_payout_map(payout_df)
        
        # グリッドサーチ
        results = run_enhanced_grid_search(valid_df, payout_map, fold_name)
        all_results.extend(results)
    
    # 結果集計
    logger.info("\n" + "=" * 60)
    logger.info("全Fold結果サマリー")
    logger.info("=" * 60)
    
    results_df = pd.DataFrame(all_results)
    
    # 戦略ごとの平均
    summary = results_df.groupby('strategy').agg({
        'roi': ['mean', 'std', 'min', 'max'],
        'hit_rate': 'mean',
        'races': 'mean'
    }).round(2)
    
    # ROI平均でソート
    summary_sorted = summary.sort_values(('roi', 'mean'), ascending=False)
    
    print("\n戦略別平均ROI (上位30):")
    print(summary_sorted.head(30).to_string())
    
    # 一貫して良好な戦略 (全Fold ROI >= 80%)
    consistent_strategies = []
    for strategy in results_df['strategy'].unique():
        strat_df = results_df[results_df['strategy'] == strategy]
        min_roi = strat_df['roi'].min()
        avg_roi = strat_df['roi'].mean()
        if min_roi >= 80:
            consistent_strategies.append({
                'strategy': strategy,
                'avg_roi': avg_roi,
                'min_roi': min_roi,
                'max_roi': strat_df['roi'].max(),
                'avg_hit_rate': strat_df['hit_rate'].mean(),
                'avg_races': strat_df['races'].mean()
            })
    
    logger.info(f"\n=== 一貫して良好な戦略 (全Fold ROI >= 80%): {len(consistent_strategies)} 件 ===")
    for s in sorted(consistent_strategies, key=lambda x: -x['avg_roi'])[:20]:
        logger.info(f"  {s['strategy']}: 平均={s['avg_roi']:.1f}%, 最小={s['min_roi']:.1f}%, 的中率={s['avg_hit_rate']:.2%}, 年間{s['avg_races']:.0f}R")
    
    # 結果保存
    output_path = os.path.join(args.cv_dir, 'enhanced_grid_search.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': all_results,
            'consistent_strategies': consistent_strategies
        }, f, indent=2)
    
    logger.info(f"\n結果を保存: {output_path}")
    logger.info("拡張グリッドサーチ完了!")

if __name__ == "__main__":
    main()

"""
拡張グリッドサーチ - 多次元分析

新しい切り口:
1. スコア分布 (Top3集中度)
2. レースグレード
3. 複合オッズ (Top3平均)
4. 頭数
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
        payout_map[rid] = {'tansho': {}, 'umaren': {}, 'wide': {}, 'umatan': {}, 'sanrenpuku': {}, 'sanrentan': {}}
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
        # ワイド
        for i in range(1, 8):
            k_comb, k_pay = f'haraimodoshi_wide_{i}a', f'haraimodoshi_wide_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['wide'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        # 馬単
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


def precompute_race_data_extended(pred_df):
    """拡張レースデータをプリコンピュート（新次元含む）"""
    race_data = {}
    
    for race_id, group in pred_df.groupby('race_id'):
        sorted_group = group.nlargest(6, 'score')
        top_horses = sorted_group['horse_number'].astype(int).tolist()
        top_scores = sorted_group['score'].tolist()
        
        if len(top_horses) < 3:
            continue
        
        # 基本情報
        n_horses = len(group)
        top1_odds = sorted_group['odds'].iloc[0] if 'odds' in sorted_group.columns else None
        
        # Top3オッズ平均
        if 'odds' in sorted_group.columns:
            top3_odds = sorted_group['odds'].head(3).tolist()
            avg_top3_odds = np.mean([o for o in top3_odds if not pd.isna(o)]) if top3_odds else None
        else:
            avg_top3_odds = None
        
        # スコア分布: Top3集中度 (Top3スコア合計 / 全体スコア合計)
        all_scores = group['score'].tolist()
        top3_score_sum = sum(top_scores[:3])
        total_score_sum = sum(all_scores)
        score_concentration = top3_score_sum / total_score_sum if total_score_sum > 0 else 0
        
        # スコア差
        score_gap = top_scores[0] - top_scores[1] if len(top_scores) >= 2 else 0
        
        # グレード情報
        grade = group['grade'].iloc[0] if 'grade' in group.columns else None
        
        info = {
            'horses': top_horses,
            'scores': top_scores,
            'score_gap': score_gap,
            'top1_odds': top1_odds,
            'avg_top3_odds': avg_top3_odds,
            'score_concentration': score_concentration,
            'n_horses': n_horses,
            'grade': grade,
        }
        
        # フォーメーション
        info['formations'] = {
            'san_box3': list(permutations(top_horses[:3], 3)),
            'san_box5': list(permutations(top_horses[:5], 3)) if len(top_horses) >= 5 else [],
            'san_1st': [(top_horses[0], o1, o2) for o1, o2 in permutations(top_horses[1:5], 2)] if len(top_horses) >= 5 else [],
            'san_2nd': [(o1, top_horses[0], o2) for o1 in top_horses[1:4] for o2 in top_horses[1:5] if o1 != o2] if len(top_horses) >= 5 else [],
            'san_3rd': [(o1, o2, top_horses[0]) for o1, o2 in permutations(top_horses[1:5], 2)] if len(top_horses) >= 5 else [],
            # フォーメーション: 1着Top1-2, 2着Top1-4, 3着Top1-5
            'san_form_12_14_15': [(f, s, t) for f in top_horses[:2] for s in top_horses[:4] for t in top_horses[:5] if f != s and s != t and f != t] if len(top_horses) >= 5 else [],
            # 馬単
            'uma_1st': [(top_horses[0], o) for o in top_horses[1:5]] if len(top_horses) >= 5 else [],
            'uma_2nd': [(o, top_horses[0]) for o in top_horses[1:4]] if len(top_horses) >= 4 else [],
            'uma_box3': list(permutations(top_horses[:3], 2)),
        }
        
        info['combos'] = {
            'sanrenpuku_box3': list(combinations(top_horses[:3], 3)),
            'sanrenpuku_box5': list(combinations(top_horses[:5], 3)) if len(top_horses) >= 5 else [],
            # 三連複 軸流し: Top1と残りTop2-5から2頭
            'sanrenpuku_nagashi': [(top_horses[0], o1, o2) for o1, o2 in combinations(top_horses[1:5], 2)] if len(top_horses) >= 5 else [],
            'umaren_box3': list(combinations(top_horses[:3], 2)),
            'umaren_box5': list(combinations(top_horses[:5], 2)) if len(top_horses) >= 5 else [],
            # 馬連 軸流し: Top1とTop2-5
            'umaren_nagashi': [(top_horses[0], o) for o in top_horses[1:5]] if len(top_horses) >= 5 else [],
            'wide_box3': list(combinations(top_horses[:3], 2)),
            'wide_box5': list(combinations(top_horses[:5], 2)) if len(top_horses) >= 5 else [],
            # ワイド 軸流し: Top1とTop2-5
            'wide_nagashi': [(top_horses[0], o) for o in top_horses[1:5]] if len(top_horses) >= 5 else [],
            # フォーメーション: 馬連/ワイド 1,2着目信頼 -> 3,4,5着目へ流し
            'umaren_form_12_345': [(a, b) for a in top_horses[:2] for b in top_horses[2:5]] if len(top_horses) >= 5 else [],
            'wide_form_12_345': [(a, b) for a in top_horses[:2] for b in top_horses[2:5]] if len(top_horses) >= 5 else [],
            # 三連複 フォーメーション
            # 軸2頭（1,2） -> 相手3,4,5,6
            'sanrenpuku_form_12_3456': [(top_horses[0], top_horses[1], h) for h in top_horses[2:6]] if len(top_horses) >= 6 else [],
        }
        
        race_data[race_id] = info
    
    return race_data


def classify_race_extended(info):
    """拡張分類 - 複数の次元を返す"""
    gap = info['score_gap']
    top1_odds = info['top1_odds'] if info['top1_odds'] is not None and not pd.isna(info['top1_odds']) else 0
    avg_top3_odds = info['avg_top3_odds'] if info['avg_top3_odds'] is not None and not pd.isna(info['avg_top3_odds']) else 0
    concentration = info['score_concentration']
    n_horses = info['n_horses']
    grade = info['grade']
    
    # 次元1: スコア差
    if gap >= 0.05:
        gap_zone = 'high_gap'
    elif gap >= 0.02:
        gap_zone = 'mid_gap'
    else:
        gap_zone = 'low_gap'
    
    # 次元2: Top1オッズ
    if 1 <= top1_odds < 5:
        odds_zone = 'favorite'
    elif 5 <= top1_odds <= 30:
        odds_zone = 'mid_odds'
    elif top1_odds > 30:
        odds_zone = 'longshot'
    else:
        odds_zone = 'unknown_odds'
    
    # 次元3: スコア集中度
    if concentration >= 0.5:
        conc_zone = 'high_conc'  # Top3で50%以上のスコア
    elif concentration >= 0.35:
        conc_zone = 'mid_conc'
    else:
        conc_zone = 'low_conc'  # 拮抗
    
    # 次元4: 頭数
    if n_horses <= 10:
        field_zone = 'small_field'
    elif n_horses <= 14:
        field_zone = 'mid_field'
    else:
        field_zone = 'large_field'
    
    # 次元5: グレード
    if grade is not None:
        g = str(grade).upper()
        if 'G1' in g:
            grade_zone = 'g1'
        elif 'G2' in g:
            grade_zone = 'g2'
        elif 'G3' in g:
            grade_zone = 'g3'
        elif 'OP' in g or 'L' in g:
            grade_zone = 'open'
        else:
            grade_zone = 'class'
    else:
        grade_zone = 'class'
    
    # 次元6: Top3平均オッズ
    if avg_top3_odds and avg_top3_odds < 10:
        top3_odds_zone = 'low_avg_odds'
    elif avg_top3_odds and avg_top3_odds < 30:
        top3_odds_zone = 'mid_avg_odds'
    else:
        top3_odds_zone = 'high_avg_odds'
    
    return {
        'gap': gap_zone,
        'odds': odds_zone,
        'conc': conc_zone,
        'field': field_zone,
        'grade': grade_zone,
        'top3_odds': top3_odds_zone
    }


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
    elif bet_type == 'wide':
        combos = info['combos'].get(formation_key, [])
        bet_amount = len(combos) * 100
        for c in combos:
            c_sorted = sorted(c)
            key = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if key in payouts['wide']:
                win_amount += payouts['wide'][key]
    elif bet_type == 'umatan':
        formations = info['formations'].get(formation_key, [])
        bet_amount = len(formations) * 100
        for p in formations:
            key = f"{p[0]:02}{p[1]:02}"
            if key in payouts['umatan']:
                win_amount += payouts['umatan'][key]
    elif bet_type == 'tansho':
        bet_amount = 100
        key = f"{info['horses'][0]:02}"
        if key in payouts['tansho']:
            win_amount = payouts['tansho'][key]
    
    return bet_amount, win_amount


def run_extended_grid_search(all_race_data, all_payout_maps):
    """拡張グリッドサーチ - 各次元の各値に対する最適馬券を探索"""
    
    bet_options = [
        ('skip', None),
        ('tansho', 'top1'),
        # 馬連
        ('umaren', 'umaren_box3'),
        ('umaren', 'umaren_box5'),
        ('umaren', 'umaren_nagashi'),
        ('umaren', 'umaren_form_12_345'),
        # ワイド
        ('wide', 'wide_box3'),
        ('wide', 'wide_box5'),
        ('wide', 'wide_nagashi'),
        ('wide', 'wide_form_12_345'),
        # 馬単
        ('umatan', 'uma_1st'),
        ('umatan', 'uma_2nd'),
        ('umatan', 'uma_box3'),
        # 三連複
        ('sanrenpuku', 'sanrenpuku_box3'),
        ('sanrenpuku', 'sanrenpuku_box5'),
        ('sanrenpuku', 'sanrenpuku_nagashi'),
        ('sanrenpuku', 'sanrenpuku_form_12_3456'),
        # 三連単
        ('sanrentan', 'san_box3'),
        ('sanrentan', 'san_box5'),
        ('sanrentan', 'san_1st'),
        ('sanrentan', 'san_2nd'),
        ('sanrentan', 'san_3rd'),
        ('sanrentan', 'san_form_12_14_15'),
    ]
    
    dimensions = {
        'gap': ['high_gap', 'mid_gap', 'low_gap'],
        'odds': ['favorite', 'mid_odds', 'longshot'],
        'conc': ['high_conc', 'mid_conc', 'low_conc'],
        'field': ['small_field', 'mid_field', 'large_field'],
        'grade': ['g1', 'g2', 'g3', 'open', 'class'],
        'top3_odds': ['low_avg_odds', 'mid_avg_odds', 'high_avg_odds'],
    }
    
    results = {}
    
    for dim_name, dim_values in dimensions.items():
        logger.info(f"\n=== 次元: {dim_name} ===")
        results[dim_name] = {}
        
        for dim_value in dim_values:
            results[dim_name][dim_value] = {}
            
            for bet_type, formation in bet_options:
                fold_rois = []
                fold_counts = []
                
                for fold_name, race_data in all_race_data.items():
                    payout_map = all_payout_maps[fold_name]
                    
                    total_bet = 0
                    total_return = 0
                    count = 0
                    
                    for race_id, info in race_data.items():
                        if race_id not in payout_map:
                            continue
                        
                        zones = classify_race_extended(info)
                        if zones[dim_name] != dim_value:
                            continue
                        
                        if bet_type == 'skip':
                            continue
                        
                        count += 1
                        b, r = execute_bet(info, payout_map[race_id], bet_type, formation)
                        total_bet += b
                        total_return += r
                    
                    roi = total_return / total_bet * 100 if total_bet > 0 else 0
                    fold_rois.append(roi)
                    fold_counts.append(count)
                
                avg_roi = np.mean(fold_rois) if fold_rois else 0
                min_roi = min(fold_rois) if fold_rois else 0
                avg_count = np.mean(fold_counts) if fold_counts else 0
                
                bet_name = f"{bet_type}_{formation}" if formation else bet_type
                results[dim_name][dim_value][bet_name] = {
                    'avg_roi': avg_roi,
                    'min_roi': min_roi,
                    'fold_rois': fold_rois,
                    'avg_races': avg_count
                }
        
        # 各値での最適馬券を表示
        for dim_value in dim_values:
            best_bet = max(results[dim_name][dim_value].items(), 
                          key=lambda x: (x[1]['min_roi'], x[1]['avg_roi']))
            bet_name, stats = best_bet
            if bet_name != 'skip' and stats['avg_races'] > 50:
                logger.info(f"  {dim_value}: {bet_name} → 平均={stats['avg_roi']:.1f}%, 最小={stats['min_roi']:.1f}%, {stats['avg_races']:.0f}R/年")
    
    return results


def main():
    cv_dir = os.path.join(project_root, 'experiments', 'v23_regression_cv')
    
    logger.info("=" * 60)
    logger.info("拡張グリッドサーチ - 多次元分析")
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
        all_race_data[fold_name] = precompute_race_data_extended(valid_df)
        all_payout_maps[fold_name] = build_payout_map(payout_df)
        logger.info(f"{fold_name}: {len(all_race_data[fold_name])} races")
    
    # 拡張グリッドサーチ実行
    results = run_extended_grid_search(all_race_data, all_payout_maps)
    
    # 高ROI条件のサマリー
    logger.info("\n" + "=" * 60)
    logger.info("高ROI条件サマリー (min ROI >= 80%)")
    logger.info("=" * 60)
    
    high_roi_conditions = []
    for dim_name, dim_results in results.items():
        for dim_value, bet_results in dim_results.items():
            for bet_name, stats in bet_results.items():
                if stats['min_roi'] >= 80 and stats['avg_races'] > 50:
                    high_roi_conditions.append({
                        'dimension': dim_name,
                        'value': dim_value,
                        'bet': bet_name,
                        'avg_roi': stats['avg_roi'],
                        'min_roi': stats['min_roi'],
                        'races': stats['avg_races']
                    })
    
    high_roi_conditions.sort(key=lambda x: -x['avg_roi'])
    
    for cond in high_roi_conditions[:20]:
        logger.info(f"  {cond['dimension']}={cond['value']}: {cond['bet']} → 平均={cond['avg_roi']:.1f}%, 最小={cond['min_roi']:.1f}%, {cond['races']:.0f}R/年")
    
    # 保存
    output_path = os.path.join(cv_dir, 'extended_grid_search.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
            'high_roi_conditions': high_roi_conditions
        }, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n結果を保存: {output_path}")
    logger.info("完了!")

if __name__ == "__main__":
    main()

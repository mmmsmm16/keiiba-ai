"""
ポートフォリオ最適化スクリプト

目的:
多数の有望なベッティングルール候補から、トータルROIと安定性を最大化する
ルールの組み合わせ（ポートフォリオ）を自動探索する。

アプローチ:
1. 「候補ルール」を大量に生成する
   - 各次元（Gap, Odds, Conc, Field...）の各値について
   - 各券種（単勝, 馬連, ワイド, 三連複, 三連単...）について
   - ROI > 75% などの足切りラインを超えるものを候補とする
2. 貪欲法（Greedy Algorithm）による最適化
   - 現在のポートフォリオに、最もROI/シャープレシオを改善するルールを1つずつ追加していく
   - 過学習を防ぐため、Fold間の安定性も考慮する
"""
import os
import sys
import pickle
import json
import logging
import random
from datetime import datetime
from itertools import combinations, permutations, product

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tqdm import tqdm

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
        payout_map[rid] = {'tansho': {}, 'umaren': {}, 'umatan': {}, 'wide': {}, 'sanrenpuku': {}, 'sanrentan': {}}
        
        # 単勝
        for i in range(1, 4):
            k_num, k_pay = f'haraimodoshi_tansho_{i}a', f'haraimodoshi_tansho_{i}b'
            if k_num in row and row[k_num] and str(row[k_num]).strip():
                try: payout_map[rid]['tansho'][str(row[k_num]).zfill(2)] = int(row[k_pay])
                except: pass
        # 馬連
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
        # 三連複
        for i in range(1, 4):
            k_comb, k_pay = f'haraimodoshi_sanrenpuku_{i}a', f'haraimodoshi_sanrenpuku_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['sanrenpuku'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        # 三連単
        for i in range(1, 7):
            k_comb, k_pay = f'haraimodoshi_sanrentan_{i}a', f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['sanrentan'][str(row[k_comb])] = int(row[k_pay])
                except: pass
    return payout_map

def get_race_info(group):
    sorted_group = group.nlargest(6, 'score')
    top_horses = sorted_group['horse_number'].astype(int).tolist()
    top_scores = sorted_group['score'].tolist()
    
    if len(top_horses) < 3:
        return None
    
    n_horses = len(group)
    top1_odds = sorted_group['odds'].iloc[0] if 'odds' in sorted_group.columns else None
    
    top3_odds = sorted_group['odds'].head(3).tolist() if 'odds' in sorted_group.columns else []
    avg_top3_odds = np.mean([o for o in top3_odds if not pd.isna(o)]) if top3_odds else None
    
    all_scores = group['score'].tolist()
    top3_score_sum = sum(top_scores[:3])
    total_score_sum = sum(all_scores)
    score_concentration = top3_score_sum / total_score_sum if total_score_sum > 0 else 0
    score_gap = top_scores[0] - top_scores[1] if len(top_scores) >= 2 else 0
    
    grade = group['grade'].iloc[0] if 'grade' in group.columns else None # 現状データが怪しいが
    
    info = {
        'horses': top_horses,
        'gap': score_gap,
        'top1_odds': top1_odds,
        'avg_top3_odds': avg_top3_odds,
        'conc': score_concentration,
        'n_horses': n_horses,
        # 分類用タグ
        'tag_gap': 'high' if score_gap >= 0.05 else ('mid' if score_gap >= 0.02 else 'low'),
        'tag_odds': 'fav' if (top1_odds and top1_odds < 5) else ('mid' if (top1_odds and top1_odds <= 30) else 'long'),
        'tag_conc': 'high' if score_concentration >= 0.5 else ('mid' if score_concentration >= 0.35 else 'low'),
        'tag_field': 'small' if n_horses <= 10 else ('mid' if n_horses <= 14 else 'large'),
        'tag_avg_top3': 'low' if (avg_top3_odds and avg_top3_odds < 10) else ('mid' if (avg_top3_odds and avg_top3_odds <= 30) else 'high'),
    }
    
    # 予想生成 (キャッシュ)
    info['bets'] = generate_bet_candidates(top_horses, info)
    return info

def generate_bet_candidates(top_horses, info):
    bets = {}
    
    # 単勝
    bets['tansho_top1'] = ([f"{top_horses[0]:02}"], 100)
    
    # 馬連
    # Box
    bets['umaren_box3'] = ([f"{min(a,b):02}{max(a,b):02}" for a,b in combinations(top_horses[:3], 2)], 300)
    bets['umaren_box5'] = ([f"{min(a,b):02}{max(a,b):02}" for a,b in combinations(top_horses[:5], 2)], 1000)
    # 流し
    if len(top_horses) >= 5:
        bets['umaren_nagashi'] = ([f"{min(top_horses[0],b):02}{max(top_horses[0],b):02}" for b in top_horses[1:5]], 400)
    # フォーメーション
    if len(top_horses) >= 5:
        # 1,2 -> 3,4,5
        hits = []
        for a in top_horses[:2]:
            for b in top_horses[2:5]:
                hits.append(f"{min(a,b):02}{max(a,b):02}")
        bets['umaren_form'] = (hits, len(hits)*100)

    # ワイド
    if len(top_horses) >= 5:
        bets['wide_box5'] = ([f"{min(a,b):02}{max(a,b):02}" for a,b in combinations(top_horses[:5], 2)], 1000)
        bets['wide_nagashi'] = ([f"{min(top_horses[0],b):02}{max(top_horses[0],b):02}" for b in top_horses[1:5]], 400)
        # 1,2 -> 3,4,5
        hits = []
        for a in top_horses[:2]:
            for b in top_horses[2:5]:
                hits.append(f"{min(a,b):02}{max(a,b):02}")
        bets['wide_form'] = (hits, len(hits)*100)

    # 馬単
    if len(top_horses) >= 5:
        # 1st fix
        bets['umatan_1st'] = ([f"{top_horses[0]:02}{b:02}" for b in top_horses[1:5]], 400)
        # 2nd fix
        bets['umatan_2nd'] = ([f"{b:02}{top_horses[0]:02}" for b in top_horses[1:4]], 300)
    
    # 三連複
    if len(top_horses) >= 6:
        bets['sanrenpuku_box5'] = ([f"{''.join(sorted([f'{x:02}' for x in c]))}" for c in combinations(top_horses[:5], 3)], 1000)
        bets['sanrenpuku_nagashi'] = ([f"{''.join(sorted([f'{top_horses[0]:02}', f'{b:02}', f'{c:02}']))}" for b,c in combinations(top_horses[1:5], 2)], 600)
        # 1,2 -> 3,4,5,6
        hits = []
        for h in top_horses[2:6]:
            code = sorted([f"{top_horses[0]:02}", f"{top_horses[1]:02}", f"{h:02}"])
            hits.append("".join(code))
        bets['sanrenpuku_form'] = (hits, len(hits)*100)

    # 三連単
    if len(top_horses) >= 5:
        bets['sanrentan_1st'] = ([f"{top_horses[0]:02}{b:02}{c:02}" for b,c in permutations(top_horses[1:5], 2)], 1200)
        bets['sanrentan_box3'] = ([f"{a:02}{b:02}{c:02}" for a,b,c in permutations(top_horses[:3], 3)], 600)

    return bets

def calculate_rule_performance(rule_key, race_indices, all_race_data, all_payout_maps, folds):
    """特定のルールのパフォーマンスを計算"""
    # rule_key: (dimension, value, bet_type)
    dim, val, bet_type = rule_key
    
    total_bet = 0
    total_return = 0
    
    fold_stats = {f['name']: {'bet': 0, 'return': 0} for f in folds}
    
    for fold in folds:
        fname = fold['name']
        race_data = all_race_data[fname]
        payout_map = all_payout_maps[fname]
        
        # このFold内の対象レースを取得（高速化のためインデックス使用推奨だが、まずはシンプルに）
        for race_id, info in race_data.items():
            # 条件判定
            match = False
            if dim == 'gap' and info['tag_gap'] == val: match = True
            elif dim == 'odds' and info['tag_odds'] == val: match = True
            elif dim == 'conc' and info['tag_conc'] == val: match = True
            elif dim == 'field' and info['tag_field'] == val: match = True
            elif dim == 'avg_top3' and info['tag_avg_top3'] == val: match = True
            
            if not match: continue
            
            # ベット実行
            if bet_type not in info['bets']: continue
            codes, cost = info['bets'][bet_type]
            
            win = 0
            # 払い戻し取得
            if race_id not in payout_map: continue
            
            # ここでは簡易的に、bet_type名から払い戻しキーを推測
            ptype = bet_type.split('_')[0] # umaren, wide, etc
            if ptype not in payout_map[race_id]: continue
            
            pm = payout_map[race_id][ptype]
            for c in codes:
                if c in pm:
                    win += pm[c]
            
            fold_stats[fname]['bet'] += cost
            fold_stats[fname]['return'] += win
            total_bet += cost
            total_return += win
            
    return total_bet, total_return, fold_stats

def optimize_portfolio(candidate_rules, all_race_data, all_payout_maps):
    """貪欲法によるポートフォリオ最適化"""
    selected_rules = []
    
    # 全レースのIDリストを作成して、重複除外などを管理しやすくする
    # しかしここではシンプルに、「ルールを追加してROIが上がるなら採用」とする
    # ただし、同じレースに複数のルールが適用される場合の投資額増加も考慮する
    
    # 候補ルールの事前評価
    logger.info("候補ルールの事前評価中...")
    valid_candidates = []
    
    for rule in tqdm(candidate_rules):
        # 評価
        t_bet, t_ret, f_stats = calculate_rule_performance(rule, None, all_race_data, all_payout_maps, FOLDS)
        
        if t_bet == 0: continue
        roi = t_ret / t_bet * 100
        
        # 足切り: ROI < 85% は対象外（厳しめに）
        if roi < 85: continue
        
        # 安定性チェック: 全Foldで黒字、あるいは少なくとも大敗していないこと
        min_fold_roi = min([s['return']/s['bet']*100 if s['bet']>0 else 0 for s in f_stats.values()])
        if min_fold_roi < 70: continue # 70%割る年があるルールは除外
        
        valid_candidates.append({
            'rule': rule,
            'roi': roi,
            'bet': t_bet,
            'return': t_ret,
            'min_fold_roi': min_fold_roi,
            'fold_stats': f_stats
        })
    
    logger.info(f"有望な候補ルール数: {len(valid_candidates)}")
    valid_candidates.sort(key=lambda x: x['roi'], reverse=True)
    
    # 上位ルールの表示
    top_rules = valid_candidates[:15]
    for c in top_rules:
        logger.info(f"  {c['rule']}: ROI={c['roi']:.1f}%, Min={c['min_fold_roi']:.1f}%")

    # ポートフォリオシミュレーション
    # 上位のルールを順番に追加していき、トータル成績を見る
    # 重複投資も許容する（券種が違うならOK、同じなら投資額が増えるだけ）
    
    logger.info("\n=== ポートフォリオ構築 (Greedy Like) ===")
    current_portfolio = []
    
    # 既に採用した (dim, val) の組み合わせを記録し、似たようなルールの重複採用を防ぐ
    # 例: gap=low で三連単と馬単の両方を採用するのはありだが、同じ券種ばかりになっても...という判断
    # ここではシンプルに「ROI貢献度が高い順」に追加して評価する
    
    # 評価用関数
    def evaluate_portfolio(rules, all_race_data, all_payout_maps, folds):
        total_bet = 0
        total_return = 0
        for fold in folds:
            fname = fold['name']
            race_data = all_race_data[fname]
            payout_map = all_payout_maps[fname]
            
            for race_id, info in race_data.items():
                if race_id not in payout_map: continue
                
                # 適用される全てのルールの投資額と払戻額を合計
                race_bet = 0
                race_ret = 0
                
                for r in rules:
                    dim, val, bet_type = r['rule']
                    
                    # 条件判定
                    match = False
                    if dim == 'gap' and info['tag_gap'] == val: match = True
                    elif dim == 'odds' and info['tag_odds'] == val: match = True
                    elif dim == 'conc' and info['tag_conc'] == val: match = True
                    elif dim == 'field' and info['tag_field'] == val: match = True
                    elif dim == 'avg_top3' and info['tag_avg_top3'] == val: match = True
                    
                    if not match: continue
                    if bet_type not in info['bets']: continue
                    
                    codes, cost = info['bets'][bet_type]
                    
                    # 払い戻し
                    ptype = bet_type.split('_')[0]
                    if ptype not in payout_map[race_id]: continue
                    
                    win = 0
                    pm = payout_map[race_id][ptype]
                    for c in codes:
                        if c in pm:
                            win += pm[c]
                    
                    race_bet += cost
                    race_ret += win
                
                total_bet += race_bet
                total_return += race_ret
                
        return total_bet, total_return

    # 追加ループ
    best_portfolio_roi = 0
    final_portfolio = []
    
    for candidate in top_rules:
        # 試しに追加
        test_portfolio = final_portfolio + [candidate]
        
        # 評価
        tb, tr = evaluate_portfolio(test_portfolio, all_race_data, all_payout_maps, FOLDS)
        if tb == 0: continue
        roi = tr / tb * 100
        
        # 追加してROIが極端に下がらない、かつ利益額が増えるなら採用
        # ここでは「ROI 95%以上を維持しつつ利益最大化」or「ROIを更新」を狙う
        
        logger.info(f"Testing + {candidate['rule']}: Total ROI={roi:.1f}% (Bet={tb:,} Return={tr:,})")
        
        # シンプルなロジック: ROIが100%を超えているなら積極的に採用
        if roi >= 100:
            final_portfolio.append(candidate)
            logger.info("  -> Adopted (Based on ROI > 100%)")
        # ROIが多少下がっても、85%以上で利益が大きく増えるなら採用（ポートフォリオの分散効果）
        elif roi >= 85 and len(final_portfolio) < 5: # 最大5つまで
            final_portfolio.append(candidate)
            logger.info("  -> Adopted (Based on ROI > 85% & Diversification)")
            
    return final_portfolio

def main():
    cv_dir = os.path.join(project_root, 'experiments', 'v23_regression_cv')
    
    # データロード
    logger.info("データロード中...")
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
        if not os.path.exists(meta_path): continue
        
        with open(os.path.join(cv_dir, fold_name, 'lgbm_v23.pkl'), 'rb') as f:
            lgbm = pickle.load(f)
        with open(os.path.join(cv_dir, fold_name, 'catboost_v23.pkl'), 'rb') as f:
            catboost = pickle.load(f)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            
        valid_df = df[df['year'] == valid_year].copy()
        for col in feature_cols:
            if col not in valid_df.columns: valid_df[col] = 0
        if 'venue' in valid_df.columns:
            valid_df = valid_df[valid_df['venue'].isin([f"{i:02}" for i in range(1, 11)])]
            
        X = valid_df[feature_cols]
        valid_df['score'] = meta.predict(np.column_stack([lgbm.predict(X), catboost.predict(X)]))
        
        payout_df = load_payout_data([valid_year])
        
        race_data = {}
        for race_id, group in valid_df.groupby('race_id'):
            info = get_race_info(group)
            if info: race_data[race_id] = info
            
        all_race_data[fold_name] = race_data
        all_payout_maps[fold_name] = build_payout_map(payout_df)
        logger.info(f"{fold_name}: {len(race_data)} races")

    # 候補生成
    # 次元 x 値 x 券種
    dimensions = {
        'gap': ['high', 'mid', 'low'],
        'odds': ['fav', 'mid', 'long'],
        'conc': ['high', 'mid', 'low'],
        'field': ['small', 'mid', 'large'],
        'avg_top3': ['low', 'mid', 'high'],
    }
    
    bet_types = [
        'tansho_top1', 
        'umaren_box3', 'umaren_box5', 'umaren_nagashi', 'umaren_form',
        'wide_box5', 'wide_nagashi', 'wide_form',
        'umatan_1st', 'umatan_2nd',
        'sanrenpuku_box5', 'sanrenpuku_nagashi', 'sanrenpuku_form',
        'sanrentan_1st', 'sanrentan_box3'
    ]
    
    candidate_rules = []
    for dim, vals in dimensions.items():
        for v in vals:
            for bt in bet_types:
                candidate_rules.append((dim, v, bt))
                
    logger.info(f"生成された候補ルール数: {len(candidate_rules)}")
    
    # 最適化実行
    best_rules = optimize_portfolio(candidate_rules, all_race_data, all_payout_maps)
    
    # 結果保存
    output_path = os.path.join(cv_dir, 'portfolio_candidates.json')
    # JSON化できないタプルを文字列に変換
    serializable_rules = []
    for r in best_rules:
        item = r.copy()
        item['rule'] = list(item['rule'])
        serializable_rules.append(item)
        
    with open(output_path, 'w') as f:
        json.dump(serializable_rules, f, indent=2)
    
    logger.info("完了")

if __name__ == "__main__":
    main()

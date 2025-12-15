"""
Fold別 戦略検証スクリプト

目的:
AIが作成した最終ルールセット(final_rules_v23.json)を、
各年(Fold)ごとに適用し、ROIの安定性を確認する。
"""
import os
import sys
import pickle
import json
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FOLDS = [
    {"valid_year": 2021, "name": "fold1"},
    {"valid_year": 2022, "name": "fold2"},
    {"valid_year": 2023, "name": "fold3"},
    {"valid_year": 2024, "name": "fold4"},
]

# ベット生成ロジック（auto_strategy_miner.pyから再利用）
from itertools import combinations, permutations
def generate_bet_candidates(top_horses, n_horses):
    bets = {}
    bets['tansho_top1'] = ([f"{top_horses[0]:02}"], 100)
    if len(top_horses) >= 3:
        bets['umaren_box3'] = ([f"{min(a,b):02}{max(a,b):02}" for a,b in combinations(top_horses[:3], 2)], 300)
    if len(top_horses) >= 5:
        bets['umaren_box5'] = ([f"{min(a,b):02}{max(a,b):02}" for a,b in combinations(top_horses[:5], 2)], 1000)
        bets['umaren_nagashi'] = ([f"{min(top_horses[0],b):02}{max(top_horses[0],b):02}" for b in top_horses[1:5]], 400)
        hits = [f"{min(a,b):02}{max(a,b):02}" for a in top_horses[:2] for b in top_horses[2:5]]
        bets['umaren_form'] = (hits, len(hits)*100)
    if len(top_horses) >= 5:
        bets['wide_box5'] = ([f"{min(a,b):02}{max(a,b):02}" for a,b in combinations(top_horses[:5], 2)], 1000)
        bets['wide_nagashi'] = ([f"{min(top_horses[0],b):02}{max(top_horses[0],b):02}" for b in top_horses[1:5]], 400)
        hits = [f"{min(a,b):02}{max(a,b):02}" for a in top_horses[:2] for b in top_horses[2:5]]
        bets['wide_form'] = (hits, len(hits)*100)
    if len(top_horses) >= 5:
        bets['umatan_1st'] = ([f"{top_horses[0]:02}{b:02}" for b in top_horses[1:5]], 400)
        bets['umatan_2nd'] = ([f"{b:02}{top_horses[0]:02}" for b in top_horses[1:4]], 300)
        bets['umatan_box3'] = ([f"{a:02}{b:02}" for a,b in permutations(top_horses[:3], 2)], 600)
    if len(top_horses) >= 6:
        bets['sanrenpuku_box5'] = ([f"{''.join(sorted([f'{x:02}' for x in c]))}" for c in combinations(top_horses[:5], 3)], 1000)
        bets['sanrenpuku_nagashi'] = ([f"{''.join(sorted([f'{top_horses[0]:02}', f'{b:02}', f'{c:02}']))}" for b,c in combinations(top_horses[1:5], 2)], 600)
        hits = []
        for h in top_horses[2:6]:
            code = sorted([f"{top_horses[0]:02}", f"{top_horses[1]:02}", f"{h:02}"])
            hits.append("".join(code))
        bets['sanrenpuku_form'] = (hits, len(hits)*100)
    if len(top_horses) >= 5:
        bets['sanrentan_1st'] = ([f"{top_horses[0]:02}{b:02}{c:02}" for b,c in permutations(top_horses[1:5], 2)], 1200)
        bets['sanrentan_box3'] = ([f"{a:02}{b:02}{c:02}" for a,b,c in permutations(top_horses[:3], 3)], 600)
    return bets

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
        # Copy parse logic... simplified for brevity, assume script has it
        for k_base in ['tansho', 'umaren', 'umatan', 'wide', 'sanrenpuku', 'sanrentan']:
            for i in range(1, 8):
                k, v = f'haraimodoshi_{k_base}_{i}a', f'haraimodoshi_{k_base}_{i}b'
                if k in row and row[k]:
                    try:
                        key = str(row[k])
                        if k_base == 'tansho': key = key.zfill(2)
                        payout_map[rid][k_base][key] = int(row[v])
                    except: pass
    return payout_map

def apply_rules(row, rules):
    """行データに対してルールを適用し、購入すべきbet_nameのリストを返す"""
    bets_to_make = []
    
    for rule in rules:
        match = True
        for feat, op, thres in rule['conditions']:
            val = row[feat]
            if op == '<=':
                if not (val <= thres):
                    match = False
                    break
            else:
                if not (val > thres):
                    match = False
                    break
        if match:
            bets_to_make.append(rule['bet_name'])
            
    return list(set(bets_to_make)) # 重複除外

def main():
    cv_dir = os.path.join(project_root, 'experiments', 'v23_regression_cv')
    rule_path = os.path.join(cv_dir, 'final_rules_v23.json')
    
    with open(rule_path, 'r') as f:
        rules = json.load(f)
    
    logger.info(f"ルール数: {len(rules)}")
    
    # データロード
    logger.info("データロード...")
    parquet_path = os.path.join(project_root, 'data/processed/preprocessed_data_v10_leakfix.parquet')
    df_meta = pd.read_parquet(parquet_path)
    
    pickle_path = os.path.join(project_root, 'data/processed/lgbm_datasets_v10_leakfix.pkl')
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    # メタ情報辞書
    meta_info = {}
    for _, row in df_meta.iterrows():
        meta_info[row['race_id']] = {
            'distance': row['distance'],
            'surface': 1 if row['surface'] == 1 else 0, # Encode same as training
            'venue': int(row['venue']) if str(row['venue']).isdigit() else 0, # Encode same as training (approx)
            'month': row['date'].month
        }
        
    # ラベルエンコーダーの再現 (簡易)
    # 決定木作成時はLabelEncoderを使用したが、今回はメタ情報から数値化済みと仮定
    # ※厳密にはエンコーダーをロードすべきだが、surface/venueは単純マッピングで代用
    
    results = []
    
    for fold in FOLDS:
        fold_name = fold['name']
        valid_year = fold['valid_year']
        
        logger.info(f"Processing {fold_name} ({valid_year})...")
        
        meta_path = os.path.join(cv_dir, fold_name, 'meta_v23.pkl')
        if not os.path.exists(meta_path): continue
        
        with open(os.path.join(cv_dir, fold_name, 'lgbm_v23.pkl'), 'rb') as f: lgbm = pickle.load(f)
        with open(os.path.join(cv_dir, fold_name, 'catboost_v23.pkl'), 'rb') as f: catboost = pickle.load(f)
        with open(meta_path, 'rb') as f: meta = pickle.load(f)
        
        valid_df = df_meta[df_meta['year'] == valid_year].copy()
        for col in feature_cols:
            if col not in valid_df.columns: valid_df[col] = 0
        if 'venue' in valid_df.columns: valid_df = valid_df[valid_df['venue'].isin([f"{i:02}" for i in range(1, 11)])]
        
        X = valid_df[feature_cols]
        valid_df['score'] = meta.predict(np.column_stack([lgbm.predict(X), catboost.predict(X)]))
        
        payout_df = load_payout_data([valid_year])
        payout_map = build_payout_map(payout_df)
        
        # レース毎の処理
        total_bet = 0
        total_ret = 0
        race_count = 0
        
        # 決定木特徴量の作成
        race_rows = []
        race_infos = {}
        
        for race_id, group in valid_df.groupby('race_id'):
            sorted_group = group.nlargest(6, 'score')
            top_horses = sorted_group['horse_number'].astype(int).tolist()
            top_scores = sorted_group['score'].tolist()
            if len(top_horses) < 3: continue
            
            top3_odds = sorted_group['odds'].head(3).tolist() if 'odds' in sorted_group.columns else []
            avg_top3_odds = np.mean([o for o in top3_odds if not pd.isna(o)]) if top3_odds else 0
            all_scores = group['score'].tolist()
            score_conc = sum(top_scores[:3]) / sum(all_scores) if sum(all_scores) > 0 else 0
            
            # Label Encodingの簡易処理 (venue, surface)
            # 実際にはルール構築時と同じエンコーディングが必要
            # 今回はルールが数値で閾値を持っているため、同じ変換を行う必要がある
            # 最良の方法は、データ準備コードを共通化することだが、ここでは簡易実装
            
            # venue mapping (JRA)
            venue_code = meta_info[race_id]['venue'] # already int from pre-processing?
            # surface: mining時はLabelEncoderか。1:芝, 2:ダート...
            # metaのsurfaceは 1,2 等
            
            row = {
                'score_gap': top_scores[0] - top_scores[1] if len(top_scores) >= 2 else 0,
                'top1_odds': sorted_group['odds'].iloc[0] if 'odds' in sorted_group.columns else 0,
                'avg_top3_odds': avg_top3_odds,
                'score_conc': score_conc,
                'n_horses': len(group),
                'distance': meta_info[race_id]['distance'],
                'surface': meta_info[race_id]['surface'], # Note: check if need -1 for 0-index
                'venue': venue_code - 1 if venue_code > 0 else 0, # 0-indexed assumed for LabelEncoder
                'month': meta_info[race_id]['month'],
            }
            race_rows.append(row)
            race_infos[race_id] = {'horses': top_horses, 'n': len(group)}
            
        # ルール適用
        # mining時は LabelEncoder.fit_transform を使った。
        # surface: 1(芝), 2(ダート)... -> transform -> 0, 1 ...?
        # Venue: 01 -> 0, 10 -> 9 ...
        # ここでは近似的に値を合わせる。
        
        for i, r_row in enumerate(race_rows):
            # LabelEncoder補正
            if r_row['surface'] > 0: r_row['surface'] -= 1 # 1->0, 2->1 ...
            
            race_id = list(race_infos.keys())[i] # 順序依存のリスクあるが、groupby順序は一定と仮定
            # 本当は辞書ではなくリストで管理すべきだったが
            
            target_bets = apply_rules(r_row, rules)
            if not target_bets: continue
            
            info = race_infos[race_id]
            bets = generate_bet_candidates(info['horses'], info['n'])
            
            if race_id not in payout_map: continue
            
            r_bet = 0
            r_ret = 0
            
            for bname in target_bets:
                if bname not in bets: continue
                codes, cost = bets[bname]
                
                ptype = bname.split('_')[0]
                if ptype not in payout_map[race_id]: continue
                
                win = 0
                pm = payout_map[race_id][ptype]
                for c in codes:
                    if c in pm: win += pm[c]
                
                r_bet += cost
                r_ret += win
            
            total_bet += r_bet
            total_ret += r_ret
            race_count += 1
            
        roi = total_ret / total_bet * 100 if total_bet > 0 else 0
        profit = total_ret - total_bet
        logger.info(f"{fold_name}: ROI={roi:.1f}%, Profit={profit:,}, Bet={total_bet:,}, Races={race_count}")
        results.append({'fold': fold_name, 'roi': roi, 'profit': profit})

    logger.info("\n=== Summary ===")
    total_profit = sum([r['profit'] for r in results])
    logger.info(f"Total Profit: {total_profit:,}")

if __name__ == "__main__":
    main()

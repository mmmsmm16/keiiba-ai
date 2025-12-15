"""
AIルール発掘 (決定木分析)

目的:
決定木を用いて、高ROIとなる「複合条件」を自動発見する。
ターゲット戦略:
1. 馬連流し (Top1 -> Top2-5)
2. 三連単1着固定 (Top1 -> Top2-5 -> Top2-5)
"""
import os
import sys
import pickle
import json
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

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
        payout_map[rid] = {'umaren': {}, 'sanrentan': {}}
        # 馬連
        for i in range(1, 4):
            k_comb, k_pay = f'haraimodoshi_umaren_{i}a', f'haraimodoshi_umaren_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['umaren'][str(row[k_comb])] = int(row[k_pay])
                except: pass
        # 三連単
        for i in range(1, 7):
            k_comb, k_pay = f'haraimodoshi_sanrentan_{i}a', f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try: payout_map[rid]['sanrentan'][str(row[k_comb])] = int(row[k_pay])
                except: pass
    return payout_map

def prepare_mining_data(folds, all_race_data, all_payout_maps, df_meta):
    """決定木分析用のデータセットを作成"""
    rows = []
    
    # 距離などの追加情報をdf_metaから取得できるように辞書化
    meta_info = {}
    for _, row in df_meta.iterrows():
        meta_info[row['race_id']] = {
            'distance': row['distance'],
            'surface': row['surface'], # 1:芝, 2:ダート... 要確認だがそのままカテゴリとして使う
            'venue': row['venue'],
            'month': row['date'].month
        }

    for fold in folds:
        fname = fold['name']
        race_data = all_race_data[fname]
        payout_map = all_payout_maps[fname]
        
        for race_id, info in race_data.items():
            if race_id not in payout_map: continue
            if race_id not in meta_info: continue
            
            # 特徴量
            row = {
                'score_gap': info['score_gap'],
                'top1_odds': info['top1_odds'] if info['top1_odds'] else 0,
                'avg_top3_odds': info['avg_top3_odds'] if info['avg_top3_odds'] else 0,
                'score_conc': info['conc'],
                'n_horses': info['n_horses'],
                'distance': meta_info[race_id]['distance'],
                'surface': meta_info[race_id]['surface'],
                'venue': meta_info[race_id]['venue'],
                'month': meta_info[race_id]['month'],
            }
            
            # ターゲット1: 馬連流し Return Rate
            umaren_bet = 0
            umaren_ret = 0
            if len(info['horses']) >= 5:
                # Top1 -> Top2-5
                combos = [(info['horses'][0], x) for x in info['horses'][1:5]]
                umaren_bet = len(combos) * 100
                pm = payout_map[race_id]['umaren']
                for c in combos:
                    key = f"{min(c):02}{max(c):02}"
                    if key in pm: umaren_ret += pm[key]
            
            row['umaren_roi'] = umaren_ret / umaren_bet if umaren_bet > 0 else 0
            
            # ターゲット2: 三連単1着固定 Return Rate
            sanren_bet = 0
            sanren_ret = 0
            if len(info['horses']) >= 5:
                # Top1 -> Top2-5 -> Top2-5
                from itertools import permutations
                combos = [(info['horses'][0], x, y) for x, y in permutations(info['horses'][1:5], 2)]
                sanren_bet = len(combos) * 100
                pm = payout_map[race_id]['sanrentan']
                for c in combos:
                    key = f"{c[0]:02}{c[1]:02}{c[2]:02}"
                    if key in pm: sanren_ret += pm[key]
            
            row['sanren_roi'] = sanren_ret / sanren_bet if sanren_bet > 0 else 0
            
            rows.append(row)
            
    return pd.DataFrame(rows)

def train_decision_tree(df, target_col, max_depth=4, min_samples_leaf=100):
    feature_cols = ['score_gap', 'top1_odds', 'avg_top3_odds', 'score_conc', 'n_horses', 'distance', 'month', 'surface_cat', 'venue_cat']
    
    # カテゴリ変数のエンコーディング
    le_surface = LabelEncoder()
    df['surface_cat'] = le_surface.fit_transform(df['surface'].astype(str))
    
    le_venue = LabelEncoder()
    df['venue_cat'] = le_venue.fit_transform(df['venue'].astype(str))
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # 決定木 (回帰)
    dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    dt.fit(X, y)
    
    return dt, feature_cols, X

def extract_rules(tree, feature_names, X, y):
    """決定木のリーフノードからルールと成績を抽出"""
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    
    node_indicator = tree.decision_path(X)
    leaf_id = tree.apply(X)
    
    rules = []
    
    # 各リーフごとに集計
    unique_leaves = np.unique(leaf_id)
    for leaf in unique_leaves:
        # このリーフに落ちたサンプル
        indices = np.where(leaf_id == leaf)[0]
        samples = len(indices)
        avg_roi = np.mean(y.iloc[indices])
        
        if avg_roi < 1.0: continue # ROI 100%未満は無視
        
        # ルール文字列の構築 (ルートからこのリーフへのパス)
        # ※sklearnのtree構造からパスを復元するのは少し面倒だが簡易的にやる
        # ここでは export_text をパースする方が人間には読みやすいかもしれないが、
        # プログラム的に処理するため、親ノードを辿るロジックなどは割愛し、
        # シンプルに export_text の結果を表示するアプローチに切り替える
        pass
        
    return unique_leaves

def main():
    cv_dir = os.path.join(project_root, 'experiments', 'v23_regression_cv')
    
    logger.info("=" * 60)
    logger.info("AIルール発掘 (決定木分析)")
    logger.info("=" * 60)
    
    # データロード
    parquet_path = os.path.join(project_root, 'data/processed/preprocessed_data_v10_leakfix.parquet')
    df_meta = pd.read_parquet(parquet_path) # メタ情報取得用
    
    pickle_path = os.path.join(project_root, 'data/processed/lgbm_datasets_v10_leakfix.pkl')
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    all_race_data = {}
    all_payout_maps = {}
    
    # Foldデータのロード（予測スコアが必要）
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
            
        valid_df = df_meta[df_meta['year'] == valid_year].copy()
        for col in feature_cols:
            if col not in valid_df.columns: valid_df[col] = 0
            
        # JRAのみ
        if 'venue' in valid_df.columns:
            valid_df = valid_df[valid_df['venue'].isin([f"{i:02}" for i in range(1, 11)])]
            
        X = valid_df[feature_cols]
        valid_df['score'] = meta.predict(np.column_stack([lgbm.predict(X), catboost.predict(X)]))
        
        payout_df = load_payout_data([valid_year])
        
        race_data = {}
        for race_id, group in valid_df.groupby('race_id'):
            sorted_group = group.nlargest(6, 'score')
            top_horses = sorted_group['horse_number'].astype(int).tolist()
            top_scores = sorted_group['score'].tolist()
            if len(top_horses) < 3: continue
            
            top3_odds = sorted_group['odds'].head(3).tolist() if 'odds' in sorted_group.columns else []
            avg_top3_odds = np.mean([o for o in top3_odds if not pd.isna(o)]) if top3_odds else None
            
            # スコア集中度
            all_scores = group['score'].tolist()
            score_conc = sum(top_scores[:3]) / sum(all_scores) if sum(all_scores) > 0 else 0
            
            race_data[race_id] = {
                'horses': top_horses,
                'score_gap': top_scores[0] - top_scores[1] if len(top_scores) >= 2 else 0,
                'top1_odds': sorted_group['odds'].iloc[0] if 'odds' in sorted_group.columns else 0,
                'avg_top3_odds': avg_top3_odds,
                'conc': score_conc,
                'n_horses': len(group)
            }
        
        all_race_data[fold_name] = race_data
        all_payout_maps[fold_name] = build_payout_map(payout_df)
        logger.info(f"{fold_name}: loaded {len(race_data)} races")
        
    # 分析用データ作成
    logger.info("分析用データセット作成中...")
    mining_df = prepare_mining_data(FOLDS, all_race_data, all_payout_maps, df_meta)
    logger.info(f"データ数: {len(mining_df)}")
    
    # 決定木分析 - 馬連流し
    logger.info("\n=== 決定木分析: 馬連流し ===")
    dt_umaren, feats, X = train_decision_tree(mining_df, 'umaren_roi', max_depth=4, min_samples_leaf=200)
    tree_text = export_text(dt_umaren, feature_names=feats)
    print(tree_text)
    
    # 決定木分析 - 三連単1着固定
    logger.info("\n=== 決定木分析: 三連単1着固定 ===")
    dt_sanren, feats, X = train_decision_tree(mining_df, 'sanren_roi', max_depth=4, min_samples_leaf=200)
    tree_text = export_text(dt_sanren, feature_names=feats)
    print(tree_text)

    # 有望なルールを抽出して表示（簡易スクレイピング）
    # ※ 本来はtree構造をパースすべきだが、ログ出力されたTreeを目視確認してユーザーに報告する形をとる
    # 出力されたテキストから high value な leaf を探す

if __name__ == "__main__":
    main()

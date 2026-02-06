import argparse
import sys
import os
import yaml
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import ndcg_score
import logging

# プロジェクトルートへのパス追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.dataset import DatasetSplitter

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def update_leaderboard(exp_name: str, ndcg_val: float):
    leaderboard_path = "reports/experiment_leaderboard.md"
    if not os.path.exists(leaderboard_path):
        logger.warning(f"Leaderboard file not found: {leaderboard_path}")
        return

    with open(leaderboard_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    updated = False
    for line in lines:
        if f"| **{exp_name}** |" in line or f"| {exp_name} |" in line:
            # カラム位置を特定して置換するのは正規表現だと複雑なので、
            # シンプルに既存行をパースして再構築するか、文字列置換を行う
            # フォーマット: | Exp ID | ... | NDCG@5 | ...
            # NDCG@5 は 7番目のカラム (インデックス6)
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 8:
                # parts[0] is empty string (before first |)
                # parts[1] is Exp ID
                # ...
                # parts[7] is NDCG@5
                parts[7] = f"**{ndcg_val:.4f}**"
                new_line = " | ".join(parts) + "\n"
                new_lines.append(new_line)
                updated = True
                continue
        new_lines.append(line)

    if updated:
        with open(leaderboard_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        logger.info(f"Leaderboard updated for {exp_name}: NDCG@5 = {ndcg_val:.4f}")
    else:
        logger.warning(f"Experiment {exp_name} not found in leaderboard.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config.get('experiment_name')
    feature_blocks = config.get('features', [])
    dataset_cfg = config.get('dataset', {})
    
    logger.info(f"Evaluating NDCG for {exp_name}")
    
    # 1. Load Data
    loader = JraVanDataLoader()
    start_date = dataset_cfg.get('train_start_date', '2020-01-01')
    end_date = dataset_cfg.get('test_end_date', '2024-12-31')
    jra_only = dataset_cfg.get('jra_only', True)
    
    logger.info(f"Loading data ({start_date} ~ {end_date})...")
    raw_df = loader.load(history_start_date=start_date, end_date=end_date, jra_only=jra_only)
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    # 2. Features
    pipeline = FeaturePipeline(cache_dir="data/features")
    df = pipeline.load_features(clean_df, feature_blocks)
    
    # マージ漏れのメタデータを補完
    key_cols = ['race_id', 'date', 'horse_id', 'rank']
    for k in key_cols:
        if k not in df.columns and k in clean_df.columns:
            df = pd.merge(df, clean_df[['race_id', 'horse_number', k]], on=['race_id', 'horse_number'], how='left', suffixes=('', '_dup'))
            if f"{k}_dup" in df.columns:
                df = df.drop(columns=[f"{k}_dup"])

    # year
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year

    # 3. Create Dataset (Validation Only)
    # target_type="ranking" を指定することで、yにRelevance Score (0,1,2,3) をセットしてもらう
    splitter = DatasetSplitter()
    valid_year = dataset_cfg.get('valid_year', 2024)
    
    logger.info(f"Creating validation dataset (Year: {valid_year}, Target: ranking)...")
    # rankingターゲット (1着=3, 2着=2, 3着=1, 他=0) を自動生成
    datasets = splitter.split_and_create_dataset(df, valid_year=valid_year, target_type="ranking")
    valid_set = datasets['valid']
    
    X = valid_set['X']
    y_true = valid_set['y'] # Relevance Scores
    group = valid_set['group']
    
    logger.info(f"Valid Data: {len(X)} rows, {len(group)} races")
    
    # 4. Load Model
    model_path = f"models/experiments/{exp_name}/model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    logger.info("Predicting...")
    # Binaryモデルの予測値 (確率)
    y_score = model.predict(X)
    
    # 5. Calculate NDCG@5
    logger.info("Calculating NDCG@5...")
    
    # group情報を使ってレースごとに分割して計算
    # groupは各クエリ(レース)のサンプル数リスト
    
    ndcg_list = []
    current_idx = 0
    
    for size in group:
        # スライス
        y_t = y_true.iloc[current_idx : current_idx + size].to_numpy()
        y_s = y_score[current_idx : current_idx + size]
        
        # サンプルが1つしかない、あるいは全てターゲット0の場合はNDCG定義不可(または0/1)だが
        # scikit-learnのndcg_scoreは y_true が全て0だと 0.0 を返す(警告出るかも)
        # y_trueのshapeは (n_samples,) -> ndcg_scoreは (n_samples, n_labels) ではなく (n_queries, n_docs)
        # ここでは1クエリ分なので calculate ndcg_score([y_t], [y_s])
        
        if np.sum(y_t) > 0:
            # ndcg_score expects (n_samples, n_features) layout? No.
            # ndcg_score(y_true, y_score, k=...)
            # y_true : array-like of shape (n_samples, n_labels)
            # True scores for each item in the ranking set.
            # Waittt. ndcg_score takes Multi-label?
            # No, standard use: shape (n_queries, n_docs).
            # So we pass [[y_t...]], [[y_s...]]
            
            score = ndcg_score([y_t], [y_s], k=5)
            ndcg_list.append(score)
        else:
            # 正解がいないレース（欠損等）はスキップあるいは0
            # 普通は複勝圏内がいるはず
            pass
            
        current_idx += size
        
    avg_ndcg = np.mean(ndcg_list)
    logger.info(f"=== Result ===")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Average NDCG@5: {avg_ndcg:.4f}")
    
    # 6. Update Leaderboard
    update_leaderboard(exp_name, avg_ndcg)

if __name__ == "__main__":
    main()

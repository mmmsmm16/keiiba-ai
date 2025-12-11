import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import pickle
import logging
import argparse
from datetime import datetime

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_v8():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_suffix', type=str, default='_v8', help='データセットファイルのサフィックス')
    parser.add_argument('--output_model', type=str, default='model_v8.pkl', help='保存するモデルファイル名')
    args = parser.parse_args()

    # 1. データロード
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
    dataset_path = os.path.join(data_dir, f'lgbm_datasets{args.data_suffix}.pkl')
    
    logger.info(f"データセットをロード中: {dataset_path}")
    if not os.path.exists(dataset_path):
        logger.error("データセットが見つかりません。run_preprocessing.pyを実行してください。")
        return

    # pickleからロード (datasets dict)
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    
    # datasets keys: X_train, y_train, q_train, ... (created by DatasetSplitter)
    # しかしDatasetSplitterは標準でClassifier/Regressor用のsplitをするかもしれない。
    # ここでは日付で分割し直すか、既存の分割を利用するか。
    # 通常DatasetSplitterは全データを保持してから分割する機能がある。
    # ここではdatasetsの中身を確認しつつ、必要ならDataFrameから再構成するが、
    # dataset.py の実装次第。通常は辞書形式でdfそのものが入っていることが多い。
    
    # DatasetSplitterの実装は確認していないが、pickled objectが辞書で、
    # 'train', 'valid', 'test' キーがあるか、X_trainなどが直接あるか。
    # ここでは安全のため、datasets全体がDataFrameであるかチェックするか...
    # いや、run_preprocessing.pyを見ると:
    # datasets = splitter.split_and_create_dataset(df)
    # pd.to_pickle(datasets, dataset_path)
    # とある。datasetsの中身を想定して書く。
    # もしdatasetsが辞書なら:
    
    if isinstance(datasets, dict):
        if 'train' in datasets:
            # {'train': df, 'valid': df, ...} 形式
            train_df = datasets['train']
            valid_df = datasets['valid']
            test_df = datasets['test']
        else:
            # {'X_train': ..., 'y_train': ...} 形式
            # これは使いにくいので、元のdfから分割し直したいが...
            # DatasetSplitterがdfを返していることを期待する。
            pass
    elif isinstance(datasets, pd.DataFrame):
        # 全データの場合
        df = datasets
        # 時系列分割
        # Train: <= 2023
        # Valid: 2024
        df['date'] = pd.to_datetime(df['date'])
        train_df = df[df['date'].dt.year <= 2023].copy()
        valid_df = df[df['date'].dt.year == 2024].copy()
        test_df = df[df['date'].dt.year == 2025].copy()
    else:
        # 想定外だが、とりあえずX_train, y_train系が入っていると仮定して進めるのは危険。
        # 今回は run_preprocessing.py の直前で df を保存している output_parquet_name を使う手もある。
        parquet_path = os.path.join(data_dir, f'preprocessed_data{args.data_suffix}.parquet')
        logger.info(f"Parquetファイルからロードを試みます: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        df['date'] = pd.to_datetime(df['date'])
        train_df = df[df['date'].dt.year <= 2023].copy()
        valid_df = df[df['date'].dt.year == 2024].copy()
        test_df = df[df['date'].dt.year == 2025].copy()

    logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}")

    # 2. LambdaRank用データ作成
    # 必要な列: 特徴量(X), ターゲット(y: Relevance), クエリ(group: race_id)
    
    # 特徴量カラムの選定 (数値型のみ、メタデータ除外)
    exclude_cols = [
        'rank', 'rank_str', 'date', 'venue', 'race_number', 'horse_number', 'horse_id', 
        'jockey_id', 'trainer_id', 'sire_id', 'mare_id', 'owner_id', 
        'race_id', 'title', 'weather', 'surface', 'state', 'horse_name', 'sex',
        'raw_time', 'time', 'odds', 'popularity', 'last_3f', 
        'passed', 'passing_rank', 'pass_1', 'pass_2', 'pass_3', 'pass_4',
        'weight_diff_sign', 'row_id', 'target'
    ]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    # 数値型のみ残す
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    
    logger.info(f"使用特徴量数: {len(feature_cols)}")
    
    def create_lgb_dataset(d, features):
        d = d.sort_values('race_id') # group作成のためソート必須
        X = d[features]
        
        # Relevance Label作成
        # 1着:3, 2着:2, 3着:1, 他:0
        def get_relevance(rank):
            if pd.isna(rank): return 0
            if rank == 1: return 3
            elif rank == 2: return 2
            elif rank == 3: return 1
            else: return 0
            
        y = d['rank'].apply(get_relevance).values
        
        # Query Group作成 (同じrace_idの行数リスト)
        group = d.groupby('race_id').size().values
        
        return lgb.Dataset(X, y, group=group)

    train_data = create_lgb_dataset(train_df, feature_cols)
    valid_data = create_lgb_dataset(valid_df, feature_cols)
    
    # 3. 学習パラメータ (LambdaRank)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbosity': -1,
        'seed': 42
    }
    
    logger.info("学習開始 (LightGBM LambdaRank)...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=2000,
        callbacks=callbacks
    )
    
    # 4. モデル保存
    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, args.output_model)
    
    logger.info(f"モデルを保存します: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    # 特徴量重要度の保存
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(model_dir, f'importance_{args.output_model}.csv')
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"特徴量重要度を保存しました: {importance_path}")
    
    # 上位特徴量の表示
    print("\n=== Feature Importance (Top 20) ===")
    print(importance_df.head(20))

if __name__ == "__main__":
    train_v8()

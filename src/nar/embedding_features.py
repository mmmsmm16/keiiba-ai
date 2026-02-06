
import pandas as pd
import numpy as np
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class EmbeddingFeatureEngineer:
    """
    Entity Embedding特徴量をデータフレームに追加するクラス。
    学習済みのEmbedding Map (ID -> Vector) をロードし、各IDをベクトルに変換します。
    """
    def __init__(self, embedding_dir='models/embeddings'):
        # 絶対パスか相対パスかを判定。相対パスの場合はプロジェクトルート基準と仮定
        if not os.path.isabs(embedding_dir):
            base_dir = os.path.join(os.path.dirname(__file__), '../../')
            embedding_dir = os.path.join(base_dir, embedding_dir)
            
        self.embedding_dir = embedding_dir
        self.map_path = os.path.join(embedding_dir, 'embedding_maps.pkl')
        self.maps = None
        self.means = {}
        
        if os.path.exists(self.map_path):
            logger.info(f"Embedding Mapをロード中: {self.map_path}")
            with open(self.map_path, 'rb') as f:
                self.maps = pickle.load(f)
                
            # 未知ID用に平均ベクトルを事前計算
            for col, emp_map in self.maps.items():
                vectors = np.array(list(emp_map.values()))
                self.means[col] = np.mean(vectors, axis=0).tolist()
        else:
            logger.warning(f"Embedding Mapが見つかりません: {self.map_path}。特徴量は追加されません。")

    def add_features(self, df):
        """
        DataFrameにEmbedding特徴量を追加します。
        対象カラム: horse_id, jockey_id, trainer_id, sire_id
        """
        if self.maps is None:
            return df
            
        logger.info("Embedding特徴量を追加中...")
        
        feature_dfs = []
        
        for col, emb_map in self.maps.items():
            if col not in df.columns:
                continue
                
            # 文字列型に変換してKeyとして使用
            keys = df[col].astype(str).values
            default_vec = self.means[col]
            
            # マッピング実行
            # リスト内包表記で約100万行程度なら十分高速
            vectors = [emb_map.get(k, default_vec) for k in keys]
            
            # DataFrame化
            dim = len(default_vec)
            cols = [f'{col}_emb_{i}' for i in range(dim)]
            vec_df = pd.DataFrame(vectors, columns=cols, index=df.index)
            
            # 型最適化 (float32)
            vec_df = vec_df.astype('float32')
            
            feature_dfs.append(vec_df)
            
        if feature_dfs:
            # 元のDataFrameに結合
            df = pd.concat([df] + feature_dfs, axis=1)
            
        return df

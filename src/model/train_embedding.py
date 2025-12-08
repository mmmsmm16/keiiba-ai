
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
import logging
import argparse

# 設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RacingDataset(Dataset):
    """
    PyTorch用データセットクラス
    """
    def __init__(self, X, y):
        self.X = {k: torch.tensor(v, dtype=torch.long) for k, v in X.items()}
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.X.items()}, self.y[idx]

class EmbeddingModel(nn.Module):
    """
    Entity Embedding学習用モデル
    各IDをEmbedding層に通した後、結合してMLPで着順スコアを予測する
    """
    def __init__(self, cardinalities, embedding_dims, hidden_units=[64, 32]):
        super(EmbeddingModel, self).__init__()
        self.embeddings = nn.ModuleDict()
        input_dim = 0
        
        # 各カテゴリ変数のEmbedding層を定義
        for col, card in cardinalities.items():
            dim = embedding_dims[col]
            self.embeddings[col] = nn.Embedding(card, dim)
            input_dim += dim
            
        layers = []
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = units
            
        layers.append(nn.Linear(input_dim, 1)) # Rank Score (0-1) を予測
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_dict):
        emb_list = []
        for col, emb_layer in self.embeddings.items():
            emb_list.append(emb_layer(x_dict[col]))
        
        # 全てのEmbeddingベクトルを結合
        x = torch.cat(emb_list, dim=1)
        return self.mlp(x)

def get_embedding_dim(cardinality):
    """
    カーディナリティに基づいてEmbedding次元数を決定する
    """
    return min(50, (cardinality + 1) // 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data.parquet')
    parser.add_argument('--output_dir', type=str, default='models/embeddings')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. データロード
    logger.info(f"{args.input} からデータをロード中...")
    df = pd.read_parquet(args.input)
    
    # 着順が存在する行のみ抽出（学習用）
    df = df[df['rank'].notna()].copy()
    
    # ターゲット変数の作成: 正規化された着順 (0=最下位, 1=1着)
    # Regressionタスクとして解くため、出走頭数で正規化する
    df['n_horses'] = df.groupby('race_id')['horse_number'].transform('count')
    df['target'] = (df['n_horses'] - df['rank']) / (df['n_horses'] - 1 + 1e-6)
    df['target'] = df['target'].clip(0, 1) # 念のため0-1範囲にクリップ

    # 2. IDのエンコーディング
    id_cols = ['horse_id', 'jockey_id', 'trainer_id', 'sire_id']
    id_cols = [c for c in id_cols if c in df.columns]
    
    encoders = {}
    X_data = {}
    cardinalities = {}
    embedding_dims = {}
    
    logger.info("カテゴリ変数をエンコーディング中...")
    for col in id_cols:
        df[col] = df[col].astype(str).fillna('UNKNOWN')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
        X_data[col] = df[col].values
        cardinalities[col] = len(le.classes_)
        # 次元数は最大8に制限（GBDTでの効率のため）
        embedding_dims[col] = min(8, get_embedding_dim(cardinalities[col])) 
        
        logger.info(f"{col}: Cardinality={cardinalities[col]}, Dim={embedding_dims[col]}")
        
        # エンコーダの保存
        with open(os.path.join(args.output_dir, f'{col}_encoder.pkl'), 'wb') as f:
            pickle.dump(le, f)

    y_data = df['target'].values
    
    # データ分割
    X_train_np = {k: v for k, v in X_data.items()}
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = {k: v[train_idx] for k, v in X_data.items()}
    y_train = y_data[train_idx]
    
    X_val = {k: v[val_idx] for k, v in X_data.items()}
    y_val = y_data[val_idx]
    
    # Dataset & DataLoader作成
    train_ds = RacingDataset(X_train, y_train)
    val_ds = RacingDataset(X_val, y_val)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 3. モデル構築
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    
    model = EmbeddingModel(cardinalities, embedding_dims).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 学習ループ
    logger.info("学習を開始します...")
    best_loss = float('inf')
    patience = 3
    no_improve = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_dl:
            batch_X = {k: v.to(device) for k, v in batch_X.items()}
            batch_y = batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 検証
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_dl:
                batch_X = {k: v.to(device) for k, v in batch_X.items()}
                batch_y = batch_y.to(device).unsqueeze(1)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dl)
        avg_val_loss = val_loss / len(val_dl)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve = 0
            # モデル保存
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'embedding_model.pth'))
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping (早期終了)")
                break
                
    # 5. Embeddingの抽出と保存
    logger.info("Embeddingを抽出・保存中...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'embedding_model.pth')))
    model.eval()
    
    embedding_maps = {}
    
    for col, emb_layer in model.embeddings.items():
        # 重み取得: (Cardinality, Dim)
        weights = emb_layer.weight.detach().cpu().numpy()
        encoder = encoders[col]
        
        # Raw ID (str) -> Vector (list) のマップを作成
        emb_map = {}
        classes = encoder.classes_
        for i, raw_id in enumerate(classes):
            emb_map[raw_id] = weights[i].tolist()
            
        embedding_maps[col] = emb_map
        logger.info(f"{col} のEmbeddingを保存: {len(emb_map)} 件")
        
    # 保存
    with open(os.path.join(args.output_dir, 'embedding_maps.pkl'), 'wb') as f:
        pickle.dump(embedding_maps, f)
        
    logger.info("完了。")

if __name__ == "__main__":
    main()

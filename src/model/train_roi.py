"""
ROI最適化モデル学習スクリプト
"""
import os
import sys
import argparse
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from model.roi_model import ROIModel
from model.roi_loss import EVMaxLoss, OddsWeightedBCE, ROIProxyLoss, CombinedROILoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RaceDataset(Dataset):
    """
    レース単位のデータセット
    
    各サンプルは1レースの全馬データ
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list, max_horses: int = 18):
        """
        Args:
            df: 全データ（race_id, rank, odds, feature_cols を含む）
            feature_cols: 使用する特徴量カラム
            max_horses: 最大馬数（パディング用）
        """
        self.feature_cols = feature_cols
        self.max_horses = max_horses
        
        # レースごとにグループ化
        self.races = []
        
        for race_id, grp in df.groupby('race_id'):
            if len(grp) < 3:  # 出走馬が少なすぎるレースはスキップ
                continue
            
            # ソート（予測ランク or horse_number）
            grp = grp.sort_values('horse_number')
            
            # 特徴量
            X = grp[feature_cols].values.astype(np.float32)
            
            # ターゲット: 勝者フラグ
            ranks = grp['rank'].values
            is_winner = (ranks == 1).astype(np.float32)
            
            # オッズ
            odds = grp['odds'].fillna(1.0).values.astype(np.float32)
            
            self.races.append({
                'race_id': race_id,
                'X': X,
                'is_winner': is_winner,
                'odds': odds,
                'n_horses': len(grp)
            })
        
        logger.info(f"Created dataset with {len(self.races)} races")
    
    def __len__(self):
        return len(self.races)
    
    def __getitem__(self, idx):
        race = self.races[idx]
        n = race['n_horses']
        
        # パディング
        X_padded = np.zeros((self.max_horses, len(self.feature_cols)), dtype=np.float32)
        is_winner_padded = np.zeros(self.max_horses, dtype=np.float32)
        odds_padded = np.ones(self.max_horses, dtype=np.float32)
        mask = np.zeros(self.max_horses, dtype=np.float32)
        
        X_padded[:n] = race['X']
        is_winner_padded[:n] = race['is_winner']
        odds_padded[:n] = race['odds']
        mask[:n] = 1.0
        
        return {
            'X': torch.tensor(X_padded),
            'is_winner': torch.tensor(is_winner_padded),
            'odds': torch.tensor(odds_padded),
            'mask': torch.tensor(mask)
        }


def load_data(data_path: str, feature_path: str = None):
    """データ読み込み"""
    logger.info(f"Loading data from {data_path}")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.pkl'):
        with open(data_path, 'rb') as f:
            datasets = pickle.load(f)
        # datasets形式の場合
        train_X = datasets['train']['X']
        train_y = datasets['train']['y']
        valid_X = datasets['valid']['X']
        valid_y = datasets['valid']['y']
        
        # DataFrameに戻す（race_id等を別途取得する必要あり）
        logger.warning("pkl format assumes preprocessed_data.parquet exists")
        df = pd.read_parquet('data/processed/preprocessed_data.parquet')
    else:
        raise ValueError(f"Unknown file format: {data_path}")
    
    return df


def train_epoch(model, dataloader, optimizer, criterion, device):
    """1エポック学習"""
    model.model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        X = batch['X'].to(device)
        is_winner = batch['is_winner'].to(device)
        odds = batch['odds'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        probs = model.model(X, mask)
        loss = criterion(probs, is_winner, odds, mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    """評価"""
    model.model.eval()
    total_loss = 0
    n_batches = 0
    
    # ROI計算用
    total_cost = 0
    total_return = 0
    n_hits = 0
    n_races = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X = batch['X'].to(device)
            is_winner = batch['is_winner'].to(device)
            odds = batch['odds'].to(device)
            mask = batch['mask'].to(device)
            
            probs = model.model(X, mask)
            loss = criterion(probs, is_winner, odds, mask)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Top1予測のROI計算
            batch_size = X.shape[0]
            for i in range(batch_size):
                valid_mask = mask[i].bool()
                valid_probs = probs[i][valid_mask]
                valid_winner = is_winner[i][valid_mask]
                valid_odds = odds[i][valid_mask]
                
                if len(valid_probs) == 0:
                    continue
                
                # Top1予測
                top1_idx = valid_probs.argmax()
                is_hit = valid_winner[top1_idx].item() == 1
                
                total_cost += 100
                if is_hit:
                    total_return += valid_odds[top1_idx].item() * 100
                    n_hits += 1
                n_races += 1
    
    avg_loss = total_loss / n_batches
    roi = (total_return / total_cost * 100) if total_cost > 0 else 0
    accuracy = (n_hits / n_races * 100) if n_races > 0 else 0
    
    return {
        'loss': avg_loss,
        'roi': roi,
        'accuracy': accuracy,
        'n_races': n_races
    }


def main():
    parser = argparse.ArgumentParser(description='ROI Model Training')
    parser.add_argument('--data', type=str, default='data/processed/preprocessed_data.parquet')
    parser.add_argument('--features', type=str, default='experiments/v12_tabnet_revival/data/lgbm_datasets.pkl')
    parser.add_argument('--output', type=str, default='experiments/v14_roi')
    parser.add_argument('--model-type', type=str, default='simple', choices=['simple', 'attention'])
    parser.add_argument('--loss', type=str, default='combined', 
                       choices=['evmax', 'odds_bce', 'roi_proxy', 'combined'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--valid-year', type=int, default=2025)
    args = parser.parse_args()
    
    # 出力ディレクトリ
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'models'), exist_ok=True)
    
    # データ読み込み
    df = load_data(args.data)
    
    # JRAのみ
    jra_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    df['venue_code'] = df['race_id'].astype(str).str[4:6]
    df = df[df['venue_code'].isin(jra_codes)].copy()
    
    # 特徴量カラム取得
    with open(args.features, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    # 欠損カラム埋め
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    
    # 数値化
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(1.0)
    df = df.dropna(subset=['rank'])
    
    # Train/Valid分割
    train_df = df[df['year'] < args.valid_year].copy()
    valid_df = df[df['year'] == args.valid_year].copy()
    
    logger.info(f"Train: {len(train_df)} rows, Valid: {len(valid_df)} rows")
    
    # Dataset作成
    train_dataset = RaceDataset(train_df, feature_cols)
    valid_dataset = RaceDataset(valid_df, feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    
    # モデル
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = ROIModel(model_type=args.model_type, hidden_dim=args.hidden_dim, device=device)
    model.build_model(len(feature_cols))
    
    # 損失関数
    if args.loss == 'evmax':
        criterion = EVMaxLoss()
    elif args.loss == 'odds_bce':
        criterion = OddsWeightedBCE()
    elif args.loss == 'roi_proxy':
        criterion = ROIProxyLoss()
    else:
        criterion = CombinedROILoss()
    
    # オプティマイザ
    optimizer = AdamW(model.model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 学習ループ
    best_roi = 0
    best_epoch = 0
    
    logger.info("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_metrics = evaluate(model, valid_loader, criterion, device)
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_metrics['loss']:.4f} | "
            f"ROI: {valid_metrics['roi']:.1f}% | "
            f"Acc: {valid_metrics['accuracy']:.1f}%"
        )
        
        # ベストモデル保存
        if valid_metrics['roi'] > best_roi:
            best_roi = valid_metrics['roi']
            best_epoch = epoch
            model.save(os.path.join(args.output, 'models', 'roi_model_best.pt'))
    
    # 最終結果
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed!")
    logger.info(f"Best ROI: {best_roi:.1f}% (epoch {best_epoch})")
    logger.info(f"Model saved to {args.output}/models/roi_model_best.pt")


if __name__ == '__main__':
    main()

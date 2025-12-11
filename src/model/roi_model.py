"""
ROI最適化用PyTorchモデル
- RaceWinPredictor: レース単位で各馬の勝率を予測
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RaceWinPredictor(nn.Module):
    """
    レース単位勝率予測モデル
    
    各馬の特徴量を入力し、レース内での勝率（softmax）を出力
    
    Args:
        input_dim: 特徴量次元
        hidden_dim: 隠れ層次元
        num_layers: 隠れ層数
        dropout: ドロップアウト率
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        
        # 馬ごとの特徴量エンコーダー
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # スコア出力層 (1馬 → 1スコア)
        self.score_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, max_horses, features]
            mask: [batch, max_horses] - 有効な馬は1, パディングは0
        
        Returns:
            probs: [batch, max_horses] - レース内での勝率（0-1, 合計1）
        """
        batch_size, max_horses, _ = x.shape
        
        # [batch * max_horses, features]
        x_flat = x.view(-1, self.input_dim)
        
        # エンコード
        h = self.encoder(x_flat)
        
        # スコア [batch * max_horses, 1]
        scores = self.score_head(h)
        
        # [batch, max_horses]
        scores = scores.view(batch_size, max_horses)
        
        # マスク適用 (パディング馬は-inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax でレース内確率
        probs = F.softmax(scores, dim=-1)
        
        return probs
    
    def predict_scores(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        生スコアを取得（ソート用）
        """
        batch_size, max_horses, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        h = self.encoder(x_flat)
        scores = self.score_head(h)
        scores = scores.view(batch_size, max_horses)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        return scores


class AttentionRacePredictor(nn.Module):
    """
    Attention付きレース予測モデル
    
    馬間の相互作用をモデル化
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 特徴量埋め込み
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Self-Attention (馬間相互作用)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 出力層
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, max_horses, features]
            mask: [batch, max_horses]
        
        Returns:
            probs: [batch, max_horses]
        """
        # 埋め込み
        h = self.input_proj(x)
        
        # Attention用マスク
        if mask is not None:
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None
        
        # Self-Attention
        h, _ = self.attention(h, h, h, key_padding_mask=key_padding_mask)
        
        # スコア
        scores = self.output(h).squeeze(-1)
        
        # マスク適用
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        probs = F.softmax(scores, dim=-1)
        
        return probs


class ROIModel:
    """
    ROI最適化モデルのラッパークラス
    
    学習・推論・保存・ロードを管理
    """
    def __init__(self, model_type: str = 'simple', input_dim: int = None,
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        self.model = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
    def build_model(self, input_dim: int):
        """モデル構築"""
        self.input_dim = input_dim
        
        if self.model_type == 'simple':
            self.model = RaceWinPredictor(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
        elif self.model_type == 'attention':
            self.model = AttentionRacePredictor(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Built {self.model_type} model: input_dim={input_dim}, hidden={self.hidden_dim}, layers={self.num_layers}, params={total_params:,}")
        return self.model
    
    def predict(self, X: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        推論
        
        Args:
            X: [n_races, max_horses, features]
            mask: [n_races, max_horses]
        
        Returns:
            scores: [n_races, max_horses]
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).to(self.device)
            if mask is not None:
                m = torch.tensor(mask, dtype=torch.float32).to(self.device)
            else:
                m = None
            
            if hasattr(self.model, 'predict_scores'):
                scores = self.model.predict_scores(x, m)
            else:
                scores = self.model(x, m)
            
            return scores.cpu().numpy()
    
    def save(self, path: str):
        """モデル保存"""
        torch.save({
            'model_state': self.model.state_dict(),
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """モデルロード"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model_type = checkpoint['model_type']
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint.get('num_layers', 2)
        self.dropout = checkpoint.get('dropout', 0.3)
        
        self.build_model(self.input_dim)
        self.model.load_state_dict(checkpoint['model_state'])
        logger.info(f"Model loaded from {path}")

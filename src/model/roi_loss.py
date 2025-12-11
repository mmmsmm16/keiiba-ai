"""
ROI最適化用カスタム損失関数
- EVMaxLoss: 期待値最大化（勝者のEVを最大化）
- OddsWeightedBCE: オッズ加重クロスエントロピー
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EVMaxLoss(nn.Module):
    """
    期待値最大化損失
    
    勝率予測が正確 AND 高オッズ馬を優先 することを同時に最適化
    
    Args:
        alpha: 敗者ペナルティの重み (default: 0.1)
        eps: 数値安定性のための小さな値
    """
    def __init__(self, alpha: float = 0.1, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
    
    def forward(self, pred_probs: torch.Tensor, is_winner: torch.Tensor, 
                odds: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_probs: モデルの勝率予測 [batch, max_horses] (0-1)
            is_winner: 実際に勝ったか [batch, max_horses] (0 or 1)
            odds: オッズ [batch, max_horses]
            mask: 有効な馬のマスク [batch, max_horses] (padded horses = 0)
        
        Returns:
            loss: スカラー
        """
        if mask is None:
            mask = torch.ones_like(pred_probs)
        
        # Expected Value = prob * odds
        ev = pred_probs * odds * mask
        
        # 勝者のEVを最大化したい → -EVが損失
        winner_ev = (ev * is_winner).sum()
        
        # 敗者への高確率予測にペナルティ
        non_winner = (1 - is_winner) * mask
        loser_penalty = (pred_probs * non_winner).mean()
        
        # 総損失
        loss = -winner_ev + self.alpha * loser_penalty
        
        return loss


class OddsWeightedBCE(nn.Module):
    """
    オッズ加重クロスエントロピー損失
    
    高オッズ馬の勝利をより重視するBCE
    
    Args:
        odds_weight_scale: オッズ重みのスケール (default: 0.1)
    """
    def __init__(self, odds_weight_scale: float = 0.1):
        super().__init__()
        self.scale = odds_weight_scale
    
    def forward(self, pred_probs: torch.Tensor, is_winner: torch.Tensor,
                odds: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_probs: モデルの勝率予測 [batch, max_horses]
            is_winner: 実際に勝ったか [batch, max_horses]
            odds: オッズ [batch, max_horses]
            mask: 有効な馬のマスク [batch, max_horses]
        """
        if mask is None:
            mask = torch.ones_like(pred_probs)
        
        # オッズベースの重み (高オッズ勝者をより重視)
        # log1p(odds) で 1.5倍→0.4, 10倍→2.4, 100倍→4.6
        weights = 1.0 + self.scale * torch.log1p(odds)
        weights = weights * is_winner + (1.0 - is_winner)  # 敗者は重み1
        weights = weights * mask
        
        # Binary Cross Entropy
        bce = F.binary_cross_entropy(
            pred_probs.clamp(1e-7, 1 - 1e-7),
            is_winner,
            weight=weights,
            reduction='sum'
        )
        
        # マスク内の有効サンプル数で正規化
        n_valid = mask.sum()
        loss = bce / (n_valid + 1e-7)
        
        return loss


class ROIProxyLoss(nn.Module):
    """
    ROI近似損失（微分可能なROIプロキシ）
    
    ROI = Σ(payout) / Σ(cost) を近似
    - 勝つと予測した馬が実際に勝てば payout = odds * 100
    - そうでなければ payout = 0
    
    Args:
        threshold: ベット判定閾値 (soft_threshold使用)
        temperature: softmax温度パラメータ
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, pred_probs: torch.Tensor, is_winner: torch.Tensor,
                odds: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        レース単位でTop1予測馬のROIを近似
        
        Args:
            pred_probs: [batch, max_horses]
            is_winner: [batch, max_horses]
            odds: [batch, max_horses]
            mask: [batch, max_horses]
        """
        if mask is None:
            mask = torch.ones_like(pred_probs)
        
        # Softmax でTop1への「賭け確率」を計算
        # temperature低い → シャープ（Top1に集中）
        masked_probs = pred_probs * mask + (1 - mask) * (-1e9)
        bet_weights = F.softmax(masked_probs / self.temperature, dim=-1)
        
        # 期待ペイアウト = Σ(bet_weight * is_winner * odds)
        payout = (bet_weights * is_winner * odds * 100).sum(dim=-1)
        
        # コスト = 100 (1点買い固定)
        cost = 100.0
        
        # ROI = payout / cost
        # 最大化したいので -ROI が損失
        roi = payout.mean() / cost
        loss = -roi + 1.0  # ROI 100% → loss 0, ROI 0% → loss 1
        
        return loss


class CombinedROILoss(nn.Module):
    """
    複合損失: 予測精度 + ROI最適化
    
    Args:
        accuracy_weight: 精度損失の重み
        roi_weight: ROI損失の重み
    """
    def __init__(self, accuracy_weight: float = 0.5, roi_weight: float = 0.5):
        super().__init__()
        self.accuracy_weight = accuracy_weight
        self.roi_weight = roi_weight
        self.bce = OddsWeightedBCE()
        self.roi = ROIProxyLoss()
    
    def forward(self, pred_probs: torch.Tensor, is_winner: torch.Tensor,
                odds: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        
        loss_acc = self.bce(pred_probs, is_winner, odds, mask)
        loss_roi = self.roi(pred_probs, is_winner, odds, mask)
        
        return self.accuracy_weight * loss_acc + self.roi_weight * loss_roi


class AccuracyROILoss(nn.Module):
    """
    的中率 + ROI 組み合わせ損失
    
    1. Top1予測が勝者の場合にボーナス（的中率最適化）
    2. 勝者のEVを最大化（ROI最適化）
    
    Args:
        accuracy_weight: 的中率損失の重み (default: 0.7)
        roi_weight: ROI損失の重み (default: 0.3)
    """
    def __init__(self, accuracy_weight: float = 0.7, roi_weight: float = 0.3):
        super().__init__()
        self.accuracy_weight = accuracy_weight
        self.roi_weight = roi_weight
    
    def forward(self, pred_probs: torch.Tensor, is_winner: torch.Tensor,
                odds: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred_probs)
        
        batch_size = pred_probs.shape[0]
        
        # === 1. 的中率損失: Top1予測が勝者かどうか ===
        # 勝者の予測確率を最大化
        winner_probs = (pred_probs * is_winner * mask).sum(dim=-1)  # [batch]
        
        # クロスエントロピー的なアプローチ
        # 勝者の確率が高いほど損失が小さい
        acc_loss = -torch.log(winner_probs.clamp(1e-7) + 1e-7).mean()
        
        # === 2. ROI損失: 勝者のEVを最大化 ===
        # Top1予測馬の期待値
        # Softmaxで「賭ける確率」を計算
        masked_probs = pred_probs * mask + (1 - mask) * (-1e9)
        bet_weights = F.softmax(masked_probs, dim=-1)
        
        # 期待ペイアウト
        payout = (bet_weights * is_winner * odds).sum(dim=-1)  # [batch]
        roi_loss = -payout.mean()  # 最大化したいので負号
        
        # === 組み合わせ ===
        total_loss = self.accuracy_weight * acc_loss + self.roi_weight * roi_loss
        
        return total_loss


class RankingLoss(nn.Module):
    """
    ランキング損失（ListNet風）
    
    勝者が最も高いスコアを持つように最適化
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, pred_probs: torch.Tensor, is_winner: torch.Tensor,
                odds: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred_probs)
        
        # モデル出力 (pred_probs) は既にSoftmax済みの確率分布
        # そのまま分布として使用する
        pred_dist = pred_probs
        
        # 正解分布: 勝者のみ1 (勝者がいない場合はゼロ除算回避)
        winner_count = is_winner.sum(dim=-1, keepdim=True).clamp(min=1)
        target_dist = is_winner / winner_count
        
        # KLダイバージェンス
        # pred_distが0の場合の対策としてclamp
        loss = F.kl_div(
            pred_dist.clamp(min=1e-7).log(),
            target_dist,
            reduction='batchmean'
        )
        
        return loss


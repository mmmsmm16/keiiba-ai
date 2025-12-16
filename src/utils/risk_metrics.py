"""
Risk Metrics Utilities
资金曲线ベースのMax Drawdown計算

定義:
- equity_t = bankroll + cumulative_profit_t
- peak_t = max(equity_0..equity_t)
- drawdown_t = (peak_t - equity_t) / peak_t  (0 <= dd <= 1)
- max_dd = max(drawdown_t)

Usage:
    from utils.risk_metrics import compute_equity_curve, compute_max_drawdown
    equity = compute_equity_curve(profits, bankroll=100000)
    max_dd = compute_max_drawdown(equity)  # 0..1
    print(f"Max DD: {max_dd * 100:.2f}%")
"""

import numpy as np
import pandas as pd
from typing import List, Union, Tuple


def compute_equity_curve(
    transactions: Union[List[float], np.ndarray, pd.Series],
    initial_bankroll: float = 100000.0,
    stop_if_bankrupt: bool = True
) -> pd.Series:
    """
    取引履歴から資金曲線を構築
    
    Args:
        transactions: 各期間の損益（profit/loss）のリスト
            正: 利益、負: 損失
        initial_bankroll: 初期資金（必ず正の値）
        stop_if_bankrupt: True=破産時(equity<=0)に停止、以降bet=0で継続
    
    Returns:
        資金曲線（equity curve）のSeries。index=0から開始
    
    Raises:
        ValueError: initial_bankroll <= 0 の場合
    """
    if initial_bankroll <= 0:
        raise ValueError(f"initial_bankroll must be positive, got {initial_bankroll}")
    
    if isinstance(transactions, pd.Series):
        profits = transactions.values
    else:
        profits = np.array(transactions, dtype=float)
    
    if len(profits) == 0:
        return pd.Series([initial_bankroll], dtype=float)
    
    # 資金曲線を構築
    equity = np.zeros(len(profits) + 1, dtype=float)
    equity[0] = initial_bankroll
    
    bankrupt = False
    for i, profit in enumerate(profits):
        if bankrupt:
            # 破産後は資金0のまま
            equity[i + 1] = 0.0
        else:
            new_equity = equity[i] + profit
            
            if stop_if_bankrupt and new_equity <= 0:
                # 破産: 資金を0にクリップし、以降はbet=0
                equity[i + 1] = 0.0
                bankrupt = True
            else:
                equity[i + 1] = new_equity
    
    return pd.Series(equity, dtype=float)


def compute_max_drawdown(equity: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    資金曲線から最大ドローダウンを計算
    
    定義:
        drawdown_t = (peak_t - equity_t) / peak_t
        max_dd = max(drawdown_t)
    
    Args:
        equity: 資金曲線（compute_equity_curve の出力）
    
    Returns:
        最大ドローダウン（0.0 ~ 1.0の範囲）
        - 0.0: ドローダウンなし（単調増加）
        - 1.0: 完全な破産（資金が0になった）
    
    Note:
        原理的に0〜1を超えない設計
    """
    if isinstance(equity, (list, np.ndarray)):
        equity = pd.Series(equity, dtype=float)
    
    if len(equity) == 0:
        return 0.0
    
    # ピーク（累積最大値）を計算
    peak = equity.cummax()
    
    # ドローダウン計算（ゼロ割り防止）
    # peak > 0 の場合のみ計算、それ以外は0
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(
            peak > 0,
            (peak - equity) / peak,
            0.0
        )
    
    # NaN/Infを0に置換
    drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 最大値を取得し、0〜1にクリップ
    max_dd = float(np.max(drawdown))
    max_dd = max(0.0, min(1.0, max_dd))
    
    return max_dd


def compute_max_drawdown_from_transactions(
    transactions: Union[List[float], np.ndarray, pd.Series],
    initial_bankroll: float = 100000.0,
    stop_if_bankrupt: bool = True
) -> Tuple[float, pd.Series]:
    """
    取引履歴から直接Max DDを計算する便利関数
    
    Args:
        transactions: 損益リスト
        initial_bankroll: 初期資金
        stop_if_bankrupt: 破産時停止フラグ
    
    Returns:
        (max_dd, equity_curve)
        max_dd: 0.0〜1.0
        equity_curve: 資金曲線Series
    """
    equity = compute_equity_curve(transactions, initial_bankroll, stop_if_bankrupt)
    max_dd = compute_max_drawdown(equity)
    return max_dd, equity


def format_max_dd_percent(max_dd: float) -> str:
    """
    Max DDを表示用にフォーマット
    
    Args:
        max_dd: 0.0〜1.0のMax DD値
    
    Returns:
        "XX.XX%" 形式の文字列
    """
    return f"{max_dd * 100:.2f}%"


# Validation helper
def validate_max_dd(max_dd: float, threshold: float = 1.0) -> bool:
    """
    Max DDがvalidation閾値を満たすかチェック
    
    Args:
        max_dd: 計算されたMax DD（0〜1）
        threshold: 許容閾値（デフォルト1.0 = 100%）
    
    Returns:
        True if max_dd <= threshold
    """
    return 0.0 <= max_dd <= threshold

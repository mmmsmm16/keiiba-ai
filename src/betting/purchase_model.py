import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PurchaseModel:
    """
    購入戦略モデル
    - 市場確率(P_market)の計算と正規化
    - 期待値(EV)の計算
    - 資金配分(Kelly Criterion等)
    """

    def __init__(self, strategies=None):
        self.strategies = strategies or {}

    def calculate_market_probability(self, df: pd.DataFrame, odds_col='odds', race_id_col='race_id') -> pd.DataFrame:
        """
        オッズから市場確率(P_market)を計算し、Overroundを補正する。
        P_market = (1 / odds) / Sigma(1 / odds)
        """
        df = df.copy()
        
        # 0やNaNのオッズを除外または補正
        df[odds_col] = df[odds_col].replace(0, np.nan)
        
        # Raw Probability (単勝支持率相当だが、控除率込み)
        df['raw_prob'] = 1.0 / df[odds_col]
        
        # レースごとのOverround (Total Probability) を計算
        overrounds = df.groupby(race_id_col)['raw_prob'].transform('sum')
        df['overround'] = overrounds
        
        # Normalized Market Probability (合計が1になるように正規化)
        # これにより、「もし控除率がなかったら市場が予測している確率」に近似
        df['p_market'] = df['raw_prob'] / overrounds
        
        return df

    def calculate_expected_value(self, df: pd.DataFrame, prob_col='prob', odds_col='odds') -> pd.DataFrame:
        """
        期待値を計算する。
        EV = (P_model * Odds) - 1
        """
        # 単純な期待値 (回収率期待値 - 1)
        # return > 0 ならプラス期待値
        df['expected_return'] = df[prob_col] * df[odds_col]
        df['ev'] = df['expected_return'] - 1.0
        return df

    def apply_betting_strategy(self, df: pd.DataFrame, strategy_name='kelly', **kwargs) -> pd.DataFrame:
        """
        資金配分を計算する。
        """
        if strategy_name == 'kelly':
            return self._strategy_kelly(df, **kwargs)
        elif strategy_name == 'flat':
            return self._strategy_flat(df, **kwargs)
        else:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return df

    def _strategy_kelly(self, df: pd.DataFrame, bankroll=10000, fraction=0.1, max_bet_rate=0.05, prob_col='prob', odds_col='odds') -> pd.DataFrame:
        """
        Kelly Criterionによる資金配分
        f* = (bp - q) / b = (p(o-1) - (1-p)) / (o-1) = p - (1-p)/(o-1)
           = (p*o - 1) / (o - 1)
        
        bet_amount = bankroll * f* * fraction
        """
        df = df.copy()
        p = df[prob_col]
        o = df[odds_col]
        
        # Kelly Fraction
        # b = odds - 1
        # f = (b*p - (1-p)) / b = (p(b+1) - 1) / b = (p*o - 1) / (o-1)
        
        # オッズ1.0倍の場合は計算不可（無限大）になるので除外
        valid_mask = (o > 1.0)
        
        kelly_f = np.zeros(len(df))
        
        # ベクトル計算
        # p*o - 1 is 'ev'
        numerator = (p * o) - 1
        denominator = o - 1
        
        k_val = numerator / denominator
        
        # マイナス期待値はベットしない (0にする)
        k_val = np.maximum(k_val, 0)
        
        kelly_f[valid_mask] = k_val[valid_mask]
        
        # Fraction適用 (ハーフケリーなど)
        bet_rate = kelly_f * fraction
        
        # 上限設定 (総資金のX%以上は賭けない)
        bet_rate = np.minimum(bet_rate, max_bet_rate)
        
        # 金額計算 (100円単位に丸める)
        bet_amount = np.floor((bankroll * bet_rate) / 100) * 100
        
        df['bet_rate'] = bet_rate
        df['bet_amount'] = bet_amount
        
        return df

    def _strategy_flat(self, df: pd.DataFrame, bet_amount=100, threshold=0.0) -> pd.DataFrame:
        """
        単純な定額ベット
        EV > threshold の場合に bet_amount を賭ける
        """
        df = df.copy()
        
        # EVが計算されていない場合は計算
        if 'ev' not in df.columns:
            if 'prob' in df.columns and 'odds' in df.columns:
                df = self.calculate_expected_value(df)
            else:
                logger.warning("Cannot calc EV. Missing prob or odds.")
                df['bet_amount'] = 0
                return df
                
        # 条件判定
        do_bet = df['ev'] > threshold
        df['bet_amount'] = np.where(do_bet, bet_amount, 0)
        
        return df

if __name__ == "__main__":
    # Test execution
    print("Testing PurchaseModel...")
    
    # Sample Data
    data = {
        'race_id': ['R1', 'R1', 'R1', 'R2', 'R2', 'R2'],
        'horse_id': [1, 2, 3, 4, 5, 6],
        'prob': [0.5, 0.3, 0.2, 0.4, 0.4, 0.2], # Model Probability
        'odds': [1.8, 4.0, 10.0, 2.0, 3.0, 15.0]  # Market Odds
    }
    df = pd.DataFrame(data)
    
    pm = PurchaseModel()
    
    # 1. Market Prob
    print("\n--- Market Probability ---")
    df = pm.calculate_market_probability(df)
    print(df[['race_id', 'odds', 'raw_prob', 'overround', 'p_market']])
    
    # 2. EV
    print("\n--- Expected Value ---")
    df = pm.calculate_expected_value(df)
    print(df[['race_id', 'prob', 'odds', 'ev']])
    
    # 3. Kelly
    print("\n--- Kelly Bet (Bankroll=10,000, Fraction=0.5) ---")
    df = pm.apply_betting_strategy(df, strategy_name='kelly', bankroll=10000, fraction=0.5)
    print(df[['race_id', 'horse_id', 'prob', 'odds', 'ev', 'bet_rate', 'bet_amount']])

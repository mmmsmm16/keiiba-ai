import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KellySimulator:
    def __init__(self, df: pd.DataFrame, initial_capital: float = 1000000):
        self.df = df.sort_values(['date', 'race_id']).copy()
        self.initial_capital = initial_capital

    def run_simulation(self, 
                       t1: float, t2: float, e1: float, en1: float, s1: int,
                       bet_type: str, 
                       kelly_fraction: float = 0.0, 
                       unit_bet: float = 1000,
                       min_bet: float = 100, 
                       max_bet: float = 10000,
                       stop_loss_dd: float = 0.3): # 30% drawdown stop loss
        """
        kelly_fraction == 0: 固定額 (unit_bet)
        kelly_fraction > 0: 分数Kelly
        """
        capital = self.initial_capital
        history = []
        
        # 意思決定用フィルタ
        payout_col = 'payout_win' if bet_type == 'win' else 'payout_place'
        
        # Kelly計算用: Placeオッズの近似値
        def get_approx_odds(row):
            if bet_type == 'win':
                return row['odds_win_pre']
            else:
                # 複勝オッズは約 (単勝オッズ-1)*0.3 + 1.1 と仮定
                return (row['odds_win_pre'] - 1) * 0.3 + 1.1

        current_max_capital = capital
        is_stopped = False
        
        for idx, row in self.df.iterrows():
            # 1位のみ
            if row['rank_pred'] != 1: continue
            
            # 条件チェック
            is_bet = (row['p1'] >= t1) and \
                     (row['margin'] >= t2) and \
                     ((row['p_cal'] * row['odds_win_pre']) >= e1) and \
                     (row['entropy'] <= en1) and \
                     (row['field_size'] <= s1)
            
            if not is_bet:
                continue
            
            # 停止ルール判定
            drawdown = (current_max_capital - capital) / current_max_capital if current_max_capital > 0 else 0
            if drawdown > stop_loss_dd:
                is_stopped = True
            
            # ベット額の決定
            if is_stopped:
                # 停止中は最小額にするか、0にするか
                # ユーザー案: 「一時停止（もしくは固定額に戻す）」
                amount = min_bet 
            elif kelly_fraction == 0:
                amount = unit_bet
            else:
                # Kelly
                odds = get_approx_odds(row)
                odds = max(odds, 1.01)
                p = row['p_cal']
                k = (p * odds - 1) / (odds - 1)
                amount = capital * k * kelly_fraction
                
                # 制約
                amount = max(min_bet, min(max_bet, amount))
                if amount > capital: amount = capital # 破産防止
            
            # 払戻
            payout = (row[payout_col] / 100.0) * amount
            net = payout - amount
            capital += net
            
            if capital > current_max_capital:
                current_max_capital = capital
                
            history.append({
                'race_id': row['race_id'],
                'date': row['date'],
                'bet_amount': amount,
                'return_amount': payout,
                'capital': capital,
                'drawdown': drawdown,
                'is_stopped': is_stopped
            })
            
        h_df = pd.DataFrame(history)
        if h_df.empty: return None
        
        # 指標計算
        total_bet = h_df['bet_amount'].sum()
        total_ret = h_df['return_amount'].sum()
        roi = total_ret / total_bet
        final_capital = capital
        
        return {
            'roi': roi,
            'final_capital': final_capital,
            'num_bets': len(h_df),
            'max_dd': h_df['drawdown'].max(),
            'history': h_df
        }

def compare_sizing_strategies(input_path: str):
    df = pd.read_parquet(input_path)
    sim = KellySimulator(df)
    
    # 1. 102% ROI Strategy (Balanced)
    params_102 = {'t1': 0.6, 't2': 0.1, 'e1': 2.6, 'en1': 99.0, 's1': 18, 'bet_type': 'win'}
    # 2. 116% ROI Strategy (Aggressive)
    params_116 = {'t1': 0.6, 't2': 0.1, 'e1': 3.0, 'en1': 99.0, 's1': 18, 'bet_type': 'win'}
    
    # Simulation
    res_fixed = sim.run_simulation(**params_102, kelly_fraction=0, unit_bet=1000)
    res_kelly = sim.run_simulation(**params_102, kelly_fraction=0.25)
    res_116 = sim.run_simulation(**params_116, kelly_fraction=0.25)
    
    print("\n--- Strategy Performance: Baseline vs Kelly ---")
    if res_fixed:
        print(f"102% Win Fixed:  ROI={res_fixed['roi']:.4f}, Final={res_fixed['final_capital']:.0f}, Bets={res_fixed['num_bets']}, MaxDD={res_fixed['max_dd']:.4f}")
    if res_kelly:
        print(f"102% Win Kelly:  ROI={res_kelly['roi']:.4f}, Final={res_kelly['final_capital']:.0f}, Bets={res_kelly['num_bets']}, MaxDD={res_kelly['max_dd']:.4f}")
    if res_116:
        print(f"116% Win Kelly:  ROI={res_116['roi']:.4f}, Final={res_116['final_capital']:.0f}, Bets={res_116['num_bets']}, MaxDD={res_116['max_dd']:.4f}")

if __name__ == "__main__":
    compare_sizing_strategies("reports/simulations/v13_e1_enriched_2022_2024.parquet")

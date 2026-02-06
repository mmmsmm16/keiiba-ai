import pandas as pd
import numpy as np
import logging
import argparse
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ROISimulator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Ensure date is datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
    def simulate_flat_bet(self, top_k: int = 1, bet_type: str = 'win', unit_bet: int = 100):
        """
        ベタ買いシミュレーション (上位K頭)
        """
        logger.info(f"Simulating Flat Bet: Top {top_k} {bet_type}")
        
        # 上位K頭を抽出 (rank_pred <= top_k)
        # res は各馬1行
        candidates = self.df[self.df['rank_pred'] <= top_k].copy()
        
        payout_col = 'payout_win' if bet_type == 'win' else 'payout_place'
        
        candidates['bet_amount'] = unit_bet
        candidates['return_amount'] = (candidates[payout_col] / 100.0) * unit_bet
        candidates['is_hit'] = (candidates[payout_col] > 0).astype(int)
        
        return self._aggregate_results(candidates, f"FlatTop{top_k}_{bet_type}")

    def simulate_threshold_bet(self, min_ev: float = 1.0, min_p: float = 0.0, bet_type: str = 'win', unit_bet: int = 100):
        """
        閾値（期待値 or 確率）ベースのシミュレーション
        """
        logger.info(f"Simulating Threshold Bet: EV > {min_ev}, P > {min_p} ({bet_type})")
        
        df = self.df.copy()
        # EV計算: p_cal * odds_win_pre
        # FIXME: Placeなどの場合は odds_win_pre から概算する必要がある（ユーザーの指摘）
        # 現時点では odds_win_pre をそのまま使う（Win想定）
        df['ev'] = df['p_cal'] * df['odds_win_pre']
        
        candidates = df[(df['ev'] >= min_ev) & (df['p_cal'] >= min_p)].copy()
        
        payout_col = 'payout_win' if bet_type == 'win' else 'payout_place'
        candidates['bet_amount'] = unit_bet
        candidates['return_amount'] = (candidates[payout_col] / 100.0) * unit_bet
        candidates['is_hit'] = (candidates[payout_col] > 0).astype(int)
        
        return self._aggregate_results(candidates, f"Threshold_EV{min_ev}_P{min_p}_{bet_type}")

    def simulate_kelly_bet(self, kelly_fraction: float = 1.0, bet_type: str = 'win', total_budget: float = 1000000):
        """
        ケリー基準に基づく資金配分シミュレーション
        b: オッズ - 1
        p: 的中確率
        k = p - (1-p)/b = (p * (b+1) - 1) / b = (p * odds - 1) / (odds - 1)
        """
        logger.info(f"Simulating Kelly Bet: Fraction {kelly_fraction} ({bet_type})")
        
        df = self.df.copy()
        # odds_win_pre が 1 以下の場合は計算できないのでクリップ
        df['odds_val'] = df['odds_win_pre'].clip(lower=1.01)
        
        # ケリー指数の計算
        df['kelly_k'] = (df['p_cal'] * df['odds_val'] - 1) / (df['odds_val'] - 1)
        
        # 0以下の場合は購入しない
        candidates = df[df['kelly_k'] > 0].copy()
        
        payout_col = 'payout_win' if bet_type == 'win' else 'payout_place'
        
        # NOTE: 資金推移をシミュレーションするためにレース順に処理する必要がある
        candidates = candidates.sort_values(['date', 'race_id'])
        
        # 簡易的な実装: 各レースでの投入額 = 現在の予算 * kelly_k * kelly_fraction
        # レース内での複数頭購入は等分または重み付けが必要だが、一旦「各行独立」で計算を簡略化
        # 本来はレースごとに集計して賭けるべきだが、ここでは「行ごとの期待利益最大化」とする
        current_capital = total_budget
        bet_history = []
        
        for idx, row in candidates.iterrows():
            bet_size = current_capital * row['kelly_k'] * kelly_fraction
            # 最小賭け金 (100円)
            if bet_size < 100: bet_size = 0
            
            payout = (row[payout_col] / 100.0) * bet_size
            net_profit = payout - bet_size
            
            current_capital += net_profit
            
            bet_history.append({
                'race_id': row['race_id'],
                'horse_id': row['horse_id'],
                'date': row['date'],
                'bet_amount': bet_size,
                'return_amount': payout,
                'is_hit': 1 if row[payout_col] > 0 else 0,
                'capital': current_capital
            })
            
        history_df = pd.DataFrame(bet_history)
        if history_df.empty: return None
        
        return self._aggregate_results(history_df, f"Kelly_f{kelly_fraction}_{bet_type}")

    def _aggregate_results(self, bet_df: pd.DataFrame, strategy_name: str):
        if bet_df.empty:
            logger.warning(f"Strategy {strategy_name}: No bets placed.")
            return None
            
        # 全体統計
        total_bet = bet_df['bet_amount'].sum()
        total_ret = bet_df['return_amount'].sum()
        total_hit = bet_df['is_hit'].sum()
        hit_rate = total_hit / len(bet_df) if len(bet_df) > 0 else 0
        roi = total_ret / total_bet if total_bet > 0 else 0
        
        # 日別・月別統計
        bet_df['month'] = bet_df['date'].dt.to_period('M')
        monthly = bet_df.groupby('month').agg({
            'bet_amount': 'sum',
            'return_amount': 'sum',
            'is_hit': 'sum',
            'race_id': 'nunique'
        })
        monthly['roi'] = monthly['return_amount'] / monthly['bet_amount']
        
        # 資金推移とドローダウン
        # レース順にソートして累積を計算
        bet_df = bet_df.sort_values(['date', 'race_id'])
        bet_df['cum_profit'] = (bet_df['return_amount'] - bet_df['bet_amount']).cumsum()
        bet_df['cum_max'] = bet_df['cum_profit'].cummax()
        bet_df['drawdown'] = bet_df['cum_max'] - bet_df['cum_profit']
        max_drawdown = bet_df['drawdown'].max()
        
        summary = {
            'strategy': strategy_name,
            'total_bet': total_bet,
            'total_return': total_ret,
            'roi': roi,
            'hit_rate': hit_rate,
            'max_drawdown': max_drawdown,
            'num_bets': len(bet_df),
            'monthly_roi_std': monthly['roi'].std() if len(monthly) > 1 else 0
        }
        
        return {
            'summary': summary,
            'monthly': monthly,
            'details': bet_df
        }

def run_simulation_suite(input_path: str):
    df = pd.read_parquet(input_path)
    sim = ROISimulator(df)
    
    results = []
    
    # 1. Flat Bets (Win & Place)
    for k in [1, 3]:
        for bt in ['win', 'place']:
            res = sim.simulate_flat_bet(top_k=k, bet_type=bt)
            if res: results.append(res)
        
    # 2. Threshold Bets (Win & Place, EV & P)
    for ev in [1.2, 1.5]:
        for bt in ['win', 'place']:
            res = sim.simulate_threshold_bet(min_ev=ev, bet_type=bt)
            if res: results.append(res)
            
    # 高確率のみ (p_cal > 0.4)
    for p_min in [0.3, 0.4]:
        res = sim.simulate_threshold_bet(min_ev=1.1, min_p=p_min, bet_type='win')
        if res: results.append(res)
        
    # 3. Kelly Bets
    for f in [0.1]:
        res = sim.simulate_kelly_bet(kelly_fraction=f, bet_type='win')
        if res: results.append(res)
        
    # 結果の表示・保存
    summary_list = [r['summary'] for r in results]
    summary_df = pd.DataFrame(summary_list).sort_values('roi', ascending=False)
    print("\n--- Simulation Summary (Sorted by ROI) ---")
    print(summary_df[['strategy', 'num_bets', 'roi', 'hit_rate', 'max_drawdown']].to_string(index=False))
    
    # CSVで詳細保存
    os.makedirs("reports/simulations/results", exist_ok=True)
    summary_df.to_csv("reports/simulations/results/v13_e1_summary.csv", index=False)
    
    # 年別でも見たい
    print("\n--- Yearly ROI ---")
    for year in [2022, 2023, 2024]:
        year_sim = ROISimulator(df[df['year_valid'] == year])
        # Win EV 1.2 を代表として表示
        y_res = year_sim.simulate_threshold_bet(min_ev=1.2, bet_type='win')
        if y_res:
             print(f"Year {year} (EV>1.2 Win): ROI = {y_res['summary']['roi']:.4f}, Bets = {y_res['summary']['num_bets']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/simulations/v13_e1_predictions_2022_2024.parquet")
    args = parser.parse_args()
    
    run_simulation_suite(args.input)

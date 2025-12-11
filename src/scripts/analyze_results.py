"""
実験結果詳細分析スクリプト
- 予測ランクごとの着順分布
- 勝率・複勝率・回収率
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from tabulate import tabulate

# パス設定
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(message)s')  # シンプルなログ
logger = logging.getLogger(__name__)

def load_predictions(exp_name):
    """予測データの読み込み"""
    pred_path = os.path.join(project_root, 'experiments', exp_name, 'reports', 'predictions.parquet')
    if not os.path.exists(pred_path):
        logger.error(f"Predictions file not found: {pred_path}")
        return None
    
    logger.info(f"Loading predictions from {pred_path}...")
    df = pd.read_parquet(pred_path)
    return df

def analyze_rank_performance(df):
    """予測ランクごとの成績分析"""
    # レースごとに予測スコアでランク付け
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    # ターゲット(0-3)を実際の着順(1-18)に変換
    # rankカラムがあればそれを使う、なければtargetから復元
    if 'rank' not in df.columns:
        if 'target' in df.columns:
            # target: 3->1, 2->2, 1->3, 0->other
            df['rank'] = df['target'].map({3: 1, 2: 2, 1: 3}).fillna(99)
        else:
            logger.error("No rank or target column found")
            return
            
    # ランク別集計 (Top 10)
    stats = []
    
    for r in range(1, 11):
        rank_df = df[df['pred_rank'] == r]
        if len(rank_df) == 0: continue
        
        n_races = len(rank_df)
        wins = len(rank_df[rank_df['rank'] == 1])
        place_2 = len(rank_df[rank_df['rank'] == 2])
        place_3 = len(rank_df[rank_df['rank'] == 3])
        place_all = wins + place_2 + place_3
        
        # 回収率（単勝・複勝）
        win_return = rank_df[rank_df['rank'] == 1]['odds'].sum() * 100
        # 複勝配当データがない場合の簡易計算（今回は単勝のみ）
        # win_roi = win_return / (n_races * 100) * 100
        
        avg_rank = rank_df[rank_df['rank'] <= 18]['rank'].mean()
        
        stats.append({
            'Rank': int(r),
            'Win%': wins / n_races * 100,
            'Hit2%': (wins + place_2) / n_races * 100,
            'Place%': place_all / n_races * 100,
            'AvgActualRank': avg_rank,
            'Count': n_races,
            'WinROI': win_return / (n_races * 100) * 100
        })
        
    print("\n=== 予測ランク別成績 ===")
    print(tabulate(stats, headers='keys', tablefmt='github', floatfmt=".1f"))
    
    # 着順分布行列 (Confusion Matrix的な)
    print("\n=== Top 5 予測馬の着順分布 (%) ===")
    dist_data = []
    for r in range(1, 6):
        rank_df = df[df['pred_rank'] == r]
        total = len(rank_df)
        dist = {
            'PredRank': r,
            '1st': len(rank_df[rank_df['rank'] == 1]) / total * 100,
            '2nd': len(rank_df[rank_df['rank'] == 2]) / total * 100,
            '3rd': len(rank_df[rank_df['rank'] == 3]) / total * 100,
            '4-5th': len(rank_df[rank_df['rank'].isin([4, 5])]) / total * 100,
            '6-10th': len(rank_df[rank_df['rank'].between(6, 10)]) / total * 100,
            '11th+': len(rank_df[rank_df['rank'] > 10]) / total * 100
        }
        dist_data.append(dist)
    
    print(tabulate(dist_data, headers='keys', tablefmt='github', floatfmt=".1f"))

    # 分析サマリ
    top1 = stats[0]
    print(f"\n[Summary]")
    print(f"Top1予測の複勝率: {top1['Place%']:.1f}% (平均着順: {top1['AvgActualRank']:.1f})")
    print(f"Top1予測の単勝回収率: {top1['WinROI']:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment directory")
    args = parser.parse_args()
    
    df = load_predictions(args.experiment_name)
    if df is not None:
        analyze_rank_performance(df)

if __name__ == "__main__":
    main()

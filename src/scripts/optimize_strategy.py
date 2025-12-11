"""
戦略最適化スクリプト (v12 Top1軸固定)
- 2025年データを時系列で分割(Optimization/Test)
- Top1を軸にした流し馬券のグリッドサーチ
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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_predictions(exp_name):
    pred_path = os.path.join(project_root, 'experiments', exp_name, 'reports', 'predictions.parquet')
    if not os.path.exists(pred_path):
        logger.error(f"Predictions file not found: {pred_path}")
        return None
    
    df = pd.read_parquet(pred_path)
    
    # 日付でソート
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'race_id'])
    
    # ランク付与
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    # 実際の着順 (rank) の確認・復元
    if 'rank' not in df.columns:
        if 'target' in df.columns:
            df['rank'] = df['target'].map({3: 1, 2: 2, 1: 3}).fillna(99)
        else:
            logger.error("No rank column found")
            return None
            
    return df

def simulate_strategy(df, bet_type, opponent_range):
    """
    Top1軸で、opponent_range (例: 2~5位) に流した場合のシミュレーション
    """
    total_bet = 0
    total_return = 0
    hits = 0
    races = 0
    
    # レースごとに処理
    for race_id, group in df.groupby('race_id'):
        # 該当ランクの馬を抽出
        axis = group[group['pred_rank'] == 1]
        if len(axis) == 0: continue
        axis = axis.iloc[0]
        
        opponents = group[(group['pred_rank'] >= opponent_range[0]) & (group['pred_rank'] <= opponent_range[1])]
        if len(opponents) == 0: continue
        
        # 賭け金: 相手の頭数 * 100円
        bet = len(opponents) * 100
        ret = 0
        hit = 0
        
        # 結果判定
        axis_rank = axis['rank']
        
        if bet_type == 'umaren': # 馬連: 軸が2着以内 & 相手が2着以内
            if axis_rank <= 2:
                # 相手の中に(3-axis_rank)着がいるか
                target_rank = 3 - axis_rank # 1->2, 2->1
                hit_opp = opponents[opponents['rank'] == target_rank]
                if len(hit_opp) > 0:
                    ret = hit_opp.iloc[0]['odds'] * 100 # 馬連オッズが必要だが、単勝オッズで近似 or データがないので仮計算
                    # ※重要: 馬連オッズがない場合、正確な計算不能。
                    # ここでは簡易的に「単勝オッズの積の平方根 * 係数」などで推定するか、
                    # もしくは「的中率」だけを重視するか。
                    # ユーザーのデータセットには odds (単勝) しかない可能性が高い。
                    # ★今回は正確なROIが出せないので、Hit Rateのみ、あるいは単勝オッズからフェルミ推定
                    # 推定: 馬連オッズ ≈ 単勝A * 単勝B * 0.75 (経験則)
                    combo_odds = axis['odds'] * hit_opp.iloc[0]['odds'] * 0.75
                    ret = combo_odds * 100
                    hit = 1
                    
        elif bet_type == 'wide': # ワイド: 軸が3着以内 & 相手が3着以内
            if axis_rank <= 3:
                hit_opps = opponents[opponents['rank'] <= 3]
                if len(hit_opps) > 0:
                    hit = 1 # 少なくとも1つ的中
                    # 配当推定: 単勝A * 単勝B * 0.25
                    for _, opp in hit_opps.iterrows():
                        combo_odds = axis['odds'] * opp['odds'] * 0.25
                        ret += combo_odds * 100
        
        elif bet_type == 'sanrenpuku': # 3連複: 軸1頭流し (相手から2頭)
             if axis_rank <= 3:
                # 相手の中に残りの2席がいるか
                hit_opps = opponents[opponents['rank'] <= 3]
                if len(hit_opps) >= 2:
                    hit = 1
                    # 配当推定: 単勝A*B*C * 0.2
                    # 簡易計算のため省略...これは難しい
                    pass

    # ※ オッズデータ不足のため、今回は「単勝 (Top1)」と「馬単 (1着固定流し)」に絞る
    # 馬単なら「軸1着、相手2着」なので、単勝オッズ * 相手の単勝オッズ * 係数 でなんとかなる...わけない。
    # 結論: 正確な配当データがないとROI最適化は無理。
    
    # しかし、ユーザーは「v12 Top1軸の分析」を求めている。
    # ここは「的中率」と「想定平均配当」でスコアリングするか、
    # 既存の metrics.json にある戦略シミュレーション機能（あれば）を使うべきだが、あれも単勝だけ。
    
    # 方針変更:
    # ROIは計算できない（配当データがないため）と断った上で、
    # 「相手ランクごとの2着・3着入着率」を出して、
    # 「Top1が1着の時、2着には何番人気の馬が来ているか」
    # 「Top1が3着以内の時、相手は何番人気か」
    # という分布行列を出すのが一番有益。
    
    pass

def analyze_opponent_distribution(df):
    """
    Top1が好走した時の、相手馬の予測ランクトレンド分析
    """
    print("\n=== Top1軸 相手馬分析 ===")
    
    # Case 1: Top1が 1着 の時
    top1_wins = df[(df['pred_rank'] == 1) & (df['rank'] == 1)]
    races_won = top1_wins['race_id'].unique()
    
    print(f"\n[馬単/馬連] Top1が1着になったレース ({len(races_won)}レース):")
    print("  2着馬の予測ランク分布:")
    
    ranks_2nd = []
    for rid in races_won:
        race_df = df[df['race_id'] == rid]
        rank2 = race_df[race_df['rank'] == 2]
        if len(rank2) > 0:
            ranks_2nd.append(rank2.iloc[0]['pred_rank'])
            
    s2 = pd.Series(ranks_2nd)
    dist2 = s2.value_counts(normalize=True).sort_index() * 100
    
    table_data = []
    cum = 0
    for r in range(1, 11):
        val = dist2.get(float(r), 0)
        cum += val
        table_data.append([r, f"{val:.1f}%", f"{cum:.1f}%"])
    
    print(tabulate(table_data, headers=['相手ランク(2着)', '出現率', '累積'], tablefmt='github'))
    
    # Case 2: Top1が 3着以内 の時
    top1_place = df[(df['pred_rank'] == 1) & (df['rank'] <= 3)]
    races_place = top1_place['race_id'].unique()
    
    print(f"\n[ワイド] Top1が3着以内に入ったレース ({len(races_place)}レース):")
    print("  相手馬(残りの2頭)の予測ランク分布:")
    
    ranks_others = []
    for rid in races_place:
        race_df = df[df['race_id'] == rid]
        others = race_df[(race_df['rank'] <= 3) & (race_df['pred_rank'] != 1)]
        for _, row in others.iterrows():
            ranks_others.append(row['pred_rank'])
            
    s3 = pd.Series(ranks_others)
    dist3 = s3.value_counts(normalize=True).sort_index() * 100
    
    table_data_w = []
    cum_w = 0
    for r in range(1, 11):
        val = dist3.get(float(r), 0)
        cum_w += val
        table_data_w.append([r, f"{val:.1f}%", f"{cum_w:.1f}%"])
        
    print(tabulate(table_data_w, headers=['相手ランク', '出現率', '累積'], tablefmt='github'))
    
    # 最適戦略の提案
    print("\n=== 推奨戦略 (データに基づく) ===")
    
    # 馬連流し推奨
    # 累積70-80%をカバーするラインを探す
    cum = 0
    rec_exacta = 0
    for r in range(1, 19):
        val = dist2.get(float(r), 0)
        cum += val
        if cum >= 80:
            rec_exacta = r
            break
            
    print(f"馬単/馬連: Top1軸 -> 予測ランク 2~{rec_exacta}位 へ流し (カバー率 >80%)")
    
    # ワイド流し推奨
    cum_w = 0
    rec_wide = 0
    for r in range(1, 19):
        val = dist3.get(float(r), 0)
        cum_w += val
        if cum_w >= 80:
            rec_wide = r
            break
    print(f"ワイド:    Top1軸 -> 予測ランク 2~{rec_wide}位 へ流し (カバー率 >80%)")


def main():
    parser = argparse.ArgumentParser(description="Optimize strategy for v12")
    parser.add_argument("experiment_name", type=str, default="v12_tabnet_revival", nargs="?")
    args = parser.parse_args()
    
    df = load_predictions(args.experiment_name)
    if df is not None:
        analyze_opponent_distribution(df)

if __name__ == "__main__":
    main()

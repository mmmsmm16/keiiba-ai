import pandas as pd
import numpy as np
import os
import sys
from itertools import permutations, combinations

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def analyze_multiaxis_strategy():
    # 1. データ読み込み
    base_dir = os.path.join(os.path.dirname(__file__), '../../../experiments')
    pred_path = os.path.join(base_dir, 'v7_ensemble_full/reports/predictions.parquet')
    payout_path = os.path.join(base_dir, 'payouts_2024_2025.parquet')
    
    print(f"Loading predictions from {pred_path}...")
    df = pd.read_parquet(pred_path)
    
    # 2025年のデータのみ
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year == 2025].copy()
    
    print(f"Loading payouts from {payout_path}...")
    payout_df = pd.read_parquet(payout_path)
    
    # 払い戻しデータを辞書化
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = str(row.get('race_id', ''))
        if not rid: continue
            
        if rid not in payout_map:
            payout_map[rid] = {'sanrentan': {}}
            
        # Sanrentan (三連単)
        for k in range(1, 7):
            comb = row.get(f'haraimodoshi_sanrentan_{k}a')
            pay = row.get(f'haraimodoshi_sanrentan_{k}b')
            if comb and pay and pd.notna(comb) and pd.notna(pay):
                try:
                    # 数値として読み込まれた場合の前ゼロ落ち対策
                    key = str(int(float(comb))).zfill(6)
                    payout_map[rid]['sanrentan'][key] = int(float(pay))
                except:
                    pass

    # 2. レースごとに処理
    q1_results = []
    q2_results = []
    
    race_ids = df['race_id'].unique()
    print(f"Analyzing {len(race_ids)} races...")
    
    for race_id in race_ids:
        race_df = df[df['race_id'] == race_id].copy()
        
        # ソート: Score降順
        race_df = race_df.sort_values('score', ascending=False)
        
        if len(race_df) < 5: continue
        
        # 予測1位の情報
        top1 = race_df.iloc[0]
        top1_pop = top1['popularity'] if pd.notna(top1['popularity']) else 1
        
        # --- 戦略条件: 予測1位が4番人気以上 ---
        if top1_pop < 4:
            continue
            
        # 軸: 予測1位
        axis = int(top1['horse_number'])
        
        # 相手: 予測2-5位 (4頭)
        opps = [int(race_df.iloc[i]['horse_number']) for i in range(1, 5) if i < len(race_df)]
        if len(opps) < 1: continue
        
        # 買い目生成: 三連単1頭軸マルチ (相手4頭 = 36点)
        # 相手から2頭選び(combinations)、軸と合わせた3頭の順列(permutations)を生成
        tickets = []
        for pair in combinations(opps, 2):
            # pair = (o1, o2)
            # 3頭セット {axis, o1, o2} の全順列
            three_horses = [axis, pair[0], pair[1]]
            for p in permutations(three_horses, 3):
                tickets.append(f"{p[0]:02}{p[1]:02}{p[2]:02}")
            
        cost = len(tickets) * 100 # 3600円
        return_amount = 0
        hit = False
        
        rid_str = str(race_id)
        if rid_str in payout_map and 'sanrentan' in payout_map[rid_str]:
            race_payouts = payout_map[rid_str]['sanrentan']
            for t in tickets:
                if t in race_payouts:
                    return_amount += race_payouts[t]
                    hit = True

        res_data = {
            'race_id': race_id,
            'date': top1['date'],
            'cost': cost,
            'return': return_amount,
            'hit': hit
        }
        
        # Q1 (1-3月) か Q2+ (4-12月) か
        month = top1['date'].month
        if 1 <= month <= 3:
            q1_results.append(res_data)
        else:
            q2_results.append(res_data)

    # 3. 集計と表示
    
    def print_stats(label, results):
        if not results:
            print(f"\n--- {label} (データなし) ---")
            return
            
        df_res = pd.DataFrame(results)
        total_races = len(df_res)
        total_hits = df_res['hit'].sum()
        hit_rate = total_hits / total_races * 100
        
        total_cost = df_res['cost'].sum()
        total_return = df_res['return'].sum()
        roi = total_return / total_cost * 100
        profit = total_return - total_cost
        
        print(f"\n--- {label} ---")
        print(f"  レース数    : {total_races}")
        print(f"  的中数      : {total_hits} ({hit_rate:.1f}%)")
        print(f"  投資        : ¥{total_cost:,}")
        print(f"  回収        : ¥{total_return:,}")
        print(f"  収支        : {'+' if profit >= 0 else ''}¥{profit:,}")
        print(f"  ROI         : {roi:.1f}%")

    print("\n" + "="*50)
    print("📊 三連単1頭軸マルチ (相手4頭) 戦略分析")
    print("="*50)
    print(f"条件: 予測1位の人気 >= 4")
    print(f"買い目: 軸1頭 - 相手4頭 (36点)")
    
    print_stats("Q1 (1月-3月) [学習/調整期間]", q1_results)
    print_stats("Q2+ (4月-12月) [検証期間]", q2_results)
    
    # 全体
    all_results = q1_results + q2_results
    print_stats("Global (2025年全体)", all_results)
    print("="*50)

if __name__ == "__main__":
    analyze_multiaxis_strategy()

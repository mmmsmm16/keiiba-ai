import pandas as pd
import numpy as np
import os
import sys
from itertools import permutations

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def analyze_strategy_stats():
    # 1. データ読み込み
    base_dir = os.path.join(os.path.dirname(__file__), '../../../experiments')
    pred_path = os.path.join(base_dir, 'v7_ensemble_full/reports/predictions.parquet')
    payout_path = os.path.join(base_dir, 'payouts_2024_2025.parquet')
    
    print(f"Loading predictions from {pred_path}...")
    df = pd.read_parquet(pred_path)
    
    # 2025年のデータのみに絞る
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year == 2025].copy()
    
    print(f"Loading payouts from {payout_path}...")
    payout_df = pd.read_parquet(payout_path)
    
    # 払い戻しデータを辞書化 for 高速アクセス
    # race_id -> type -> key -> payout
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = str(row.get('race_id', ''))
        if not rid: continue
            
        if rid not in payout_map:
            payout_map[rid] = {'tansho': {}, 'sanrentan': {}, 'sanrenpuku': {}}
            
        # Tansho (単勝)
        for k in range(1, 4):
            comb = row.get(f'haraimodoshi_tansho_{k}a')
            pay = row.get(f'haraimodoshi_tansho_{k}b')
            if comb and pay and pd.notna(comb) and pd.notna(pay):
                try:
                    payout_map[rid]['tansho'][str(int(float(comb))).zfill(2)] = int(float(pay))
                except:
                    pass
        
        # Sanrentan (三連単)
        for k in range(1, 7):
            comb = row.get(f'haraimodoshi_sanrentan_{k}a')
            pay = row.get(f'haraimodoshi_sanrentan_{k}b')
            if comb and pay and pd.notna(comb) and pd.notna(pay):
                try:
                    # 数値として読み込まれた場合の前ゼロ落ち対策
                    # 例: 10203 -> 010203
                    key = str(int(float(comb))).zfill(6)
                    payout_map[rid]['sanrentan'][key] = int(float(pay))
                except:
                    pass

    # 2. レースごとに処理
    results = []
    
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
        
        # --- 戦略条件: 予測1位が4番人気以上 (人気薄) ---
        if top1_pop < 4:
            continue
            
        # 購入対象レース
        axis = int(top1['horse_number'])
        
        # 相手: 予測2-5位 (4頭)
        opps = [int(race_df.iloc[i]['horse_number']) for i in range(1, 5) if i < len(race_df)]
        if len(opps) < 1: continue
        
        # 買い目生成: 三連単1頭軸流し (相手4頭 = 12点)
        tickets = []
        # Axis -> Opp -> Opp
        for o1, o2 in permutations(opps, 2):
            tickets.append(f"{axis:02}{o1:02}{o2:02}")
            
        cost = len(tickets) * 100
        return_amount = 0
        hit = False
        hit_payout = 0
        
        rid_str = str(race_id)
        if rid_str in payout_map and 'sanrentan' in payout_map[rid_str]:
            race_payouts = payout_map[rid_str]['sanrentan']
            for t in tickets:
                if t in race_payouts:
                    return_amount += race_payouts[t]
                    hit = True
                    hit_payout = race_payouts[t] # 1点あたりと仮定(重複的中はほぼない賭け式)

        results.append({
            'race_id': race_id,
            'date': top1['date'],
            'venue': top1['venue'],
            'cost': cost,
            'return': return_amount,
            'hit': hit,
            'hit_payout': hit_payout
        })

    # 3. 集計
    res_df = pd.DataFrame(results)
    
    if res_df.empty:
        print("No target races found.")
        return

    # 日付順にソート（連敗計算のため）
    res_df = res_df.sort_values('date')

    total_races = len(res_df)
    total_hits = res_df['hit'].sum()
    hit_rate = total_hits / total_races * 100
    
    total_cost = res_df['cost'].sum()
    total_return = res_df['return'].sum()
    roi = total_return / total_cost * 100
    
    # 払い戻し統計 (的中したレースのみ)
    hit_df = res_df[res_df['hit']]
    min_payout = hit_df['hit_payout'].min() if not hit_df.empty else 0
    max_payout = hit_df['hit_payout'].max() if not hit_df.empty else 0
    avg_payout = hit_df['hit_payout'].mean() if not hit_df.empty else 0
    median_payout = hit_df['hit_payout'].median() if not hit_df.empty else 0
    
    # 連敗計算
    # True/Falseの配列から、連続するFalseの最大長を求める
    hits = res_df['hit'].values
    max_losing_streak = 0
    current_streak = 0
    for h in hits:
        if not h:
            current_streak += 1
        else:
            max_losing_streak = max(max_losing_streak, current_streak)
            current_streak = 0
    max_losing_streak = max(max_losing_streak, current_streak) # 最後まで負けてた場合
    
    print("\n" + "="*50)
    print("📊 Option C (穴狙い) 戦略分析レポート (2025年)")
    print("="*50)
    print(f"条件: 予測1位の人気 >= 4")
    print(f"賭式: 三連単 1頭軸相手4頭流し (12点)")
    print("-" * 30)
    print(f"対象レース数  : {total_races} レース")
    print(f"的中レース数  : {total_hits} レース")
    print(f"的中率        : {hit_rate:.2f}%")
    print(f"総投資額      : ¥{total_cost:,}")
    print(f"総払戻額      : ¥{total_return:,}")
    print(f"回収率 (ROI)  : {roi:.1f}%")
    print("-" * 30)
    print(f"最小払戻額    : ¥{min_payout:,.0f}")
    print(f"最大払戻額    : ¥{max_payout:,.0f}")
    print(f"平均払戻額    : ¥{avg_payout:,.0f}")
    print(f"中央値払戻額  : ¥{median_payout:,.0f}")
    print("-" * 30)
    print(f"最大連敗数    : {max_losing_streak} レース")
    print("="*50)
    
    # 月別の成績も出してみる
    res_df['month'] = res_df['date'].dt.month
    monthly = res_df.groupby('month').agg({
        'race_id': 'count',
        'hit': 'sum',
        'cost': 'sum',
        'return': 'sum'
    })
    monthly['roi'] = monthly['return'] / monthly['cost'] * 100
    monthly['hit_rate'] = monthly['hit'] / monthly['race_id'] * 100
    
    print("\n📅 月別成績")
    print(monthly[['race_id', 'hit_rate', 'roi']])

if __name__ == "__main__":
    analyze_strategy_stats()

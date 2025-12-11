import pandas as pd
import numpy as np
import os
import sys
from itertools import permutations, combinations

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def analyze_formation_strategy():
    # 1. データ読み込み
    base_dir = os.path.join(os.path.dirname(__file__), '../../../experiments')
    pred_path = os.path.join(base_dir, 'v7_ensemble_full/reports/predictions.parquet')
    payout_path = os.path.join(base_dir, 'payouts_2024_2025.parquet')
    
    print(f"Loading predictions from {pred_path}...")
    df = pd.read_parquet(pred_path)
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
            payout_map[rid] = {'sanrentan': {}, 'sanrenpuku': {}, 'wide': {}}
            
        # Sanrentan
        for k in range(1, 7):
            comb = row.get(f'haraimodoshi_sanrentan_{k}a')
            pay = row.get(f'haraimodoshi_sanrentan_{k}b')
            if comb and pay and pd.notna(comb) and pd.notna(pay):
                try:
                    key = str(int(float(comb))).zfill(6)
                    payout_map[rid]['sanrentan'][key] = int(float(pay))
                except: pass

        # Sanrenpuku
        for k in range(1, 4):
            comb = row.get(f'haraimodoshi_sanrenpuku_{k}a')
            pay = row.get(f'haraimodoshi_sanrenpuku_{k}b')
            if comb and pay and pd.notna(comb) and pd.notna(pay):
                try:
                    key = str(int(float(comb))).zfill(6)
                    payout_map[rid]['sanrenpuku'][key] = int(float(pay))
                except: pass
                
        # Wide (必要なら)
        # for k in range(1, 8): ... 省略

    # 2. 戦略定義
    strategies = {
        "Current (1-Axis Flow)": {"cost": 0, "return": 0, "hits": 0, "races": 0},
        "Multi 3 (Axis+Top3 Opps)": {"cost": 0, "return": 0, "hits": 0, "races": 0},
        "Zurashi 24 (Axis+2,3 Cover)": {"cost": 0, "return": 0, "hits": 0, "races": 0},
        "Sanrenpuku Box 5": {"cost": 0, "return": 0, "hits": 0, "races": 0},
    }
    
    race_ids = df['race_id'].unique()
    print(f"Analyzing {len(race_ids)} races...")
    
    for race_id in race_ids:
        race_df = df[df['race_id'] == race_id].copy()
        race_df = race_df.sort_values('score', ascending=False)
        
        if len(race_df) < 5: continue
        
        top1 = race_df.iloc[0]
        if pd.notna(top1['popularity']) and top1['popularity'] < 4:
            continue # 人気馬スキップ
            
        h_nums = race_df['horse_number'].astype(int).tolist()
        # Top 5 horses
        top5 = h_nums[:5]
        axis = h_nums[0]
        opps4 = h_nums[1:5] # 2,3,4,5位
        opps3 = h_nums[1:4] # 2,3,4位
        
        rid_str = str(race_id)
        if rid_str not in payout_map: continue
        
        # --- 1. Current (1-Axis Flow) ---
        # 1 -> 2,3,4,5 -> 2,3,4,5 (12点)
        tickets = []
        for p in permutations(opps4, 2):
            tickets.append(f"{axis:02}{p[0]:02}{p[1]:02}")
        
        st = strategies["Current (1-Axis Flow)"]
        st["races"] += 1
        st["cost"] += len(tickets) * 100
        for t in tickets:
            if t in payout_map[rid_str]['sanrentan']:
                st["return"] += payout_map[rid_str]['sanrentan'][t]
                st["hits"] += 1
                break # 1レース1的中前提（単系）
                
        # --- 2. Multi 3 (Axis + Top3 Opps) ---
        # Axis + opps3 の4頭から、Axisを含む3頭の順列 (相手3頭から2頭選ぶ)
        # 18点
        tickets = []
        for pair in combinations(opps3, 2):
            # {axis, pair0, pair1} の順列
            for p in permutations([axis, pair[0], pair[1]], 3):
                tickets.append(f"{p[0]:02}{p[1]:02}{p[2]:02}")
                
        st = strategies["Multi 3 (Axis+Top3 Opps)"]
        st["races"] += 1
        st["cost"] += len(tickets) * 100
        for t in tickets:
            if t in payout_map[rid_str]['sanrentan']:
                st["return"] += payout_map[rid_str]['sanrentan'][t]
                st["hits"] += 1
                break

        # --- 3. Zurashi 24 ---
        # A: 1 -> 2-5 -> 2-5 (12点) -> Currentと同じ
        # B: 2,3 -> 1 -> 2-5 (6点)
        # C: 2,3 -> 2-5 -> 1 (6点)
        tickets = []
        # A
        for p in permutations(opps4, 2):
            tickets.append(f"{axis:02}{p[0]:02}{p[1]:02}")
        
        cov_axis = h_nums[1:3] # 2,3位
        cov_opps = h_nums[1:5] # 2,3,4,5位
        
        for ca in cov_axis:
            # B: ca -> axis -> (opps - ca)
            rem_opps = [o for o in cov_opps if o != ca]
            for ro in rem_opps:
                tickets.append(f"{ca:02}{axis:02}{ro:02}")
            # C: ca -> (opps - ca) -> axis
            for ro in rem_opps:
                tickets.append(f"{ca:02}{ro:02}{axis:02}")
                
        # 重複削除 (念のため、理論上はないはずだが)
        tickets = list(set(tickets))
        
        st = strategies["Zurashi 24 (Axis+2,3 Cover)"]
        st["races"] += 1
        st["cost"] += len(tickets) * 100
        hit = False
        amt = 0
        for t in tickets:
            if t in payout_map[rid_str]['sanrentan']:
                amt += payout_map[rid_str]['sanrentan'][t]
                hit = True
        if hit:
            st["hits"] += 1
            st["return"] += amt

        # --- 4. Sanrenpuku Box 5 ---
        # 1-5位 BOX (10点)
        tickets = []
        # 三連複のキーは昇順ソートして結合
        for c in combinations(top5, 3):
            s = sorted(c)
            tickets.append(f"{s[0]:02}{s[1]:02}{s[2]:02}")
            
        st = strategies["Sanrenpuku Box 5"]
        st["races"] += 1
        st["cost"] += len(tickets) * 100
        hit = False
        amt = 0
        for t in tickets:
            if t in payout_map[rid_str]['sanrenpuku']:
                amt += payout_map[rid_str]['sanrenpuku'][t]
                hit = True
        if hit:
            st["hits"] += 1
            st["return"] += amt
            
    # 結果表示
    print("\n" + "="*60)
    print("🧪 フォーメーション戦略比較 (2025 Prediction Pop>=4)")
    print("="*60)
    
    for name, s in strategies.items():
        roi = s["return"] / s["cost"] * 100 if s["cost"] > 0 else 0
        hr = s["hits"] / s["races"] * 100 if s["races"] > 0 else 0
        profit = s["return"] - s["cost"]
        
        print(f"🔹 {name}")
        print(f"   Hit Rate: {hr:5.2f}% | ROI: {roi:5.1f}% | Profit: ¥{profit:,}")
        print("-" * 30)
        
    print("="*60)

if __name__ == "__main__":
    analyze_formation_strategy()

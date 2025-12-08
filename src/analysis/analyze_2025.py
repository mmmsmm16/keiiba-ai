
import pandas as pd
import os

EXPERIMENTS_DIR = "experiments"
PRED_FILE = os.path.join(EXPERIMENTS_DIR, "predictions_catboost_v7.parquet")
PAYOUT_FILE = os.path.join(EXPERIMENTS_DIR, "payouts_2024_2025.parquet")

def analyze():
    print(f"Loading {PRED_FILE}...")
    if not os.path.exists(PRED_FILE) or not os.path.exists(PAYOUT_FILE):
        print("Files not found.")
        return

    df = pd.read_parquet(PRED_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for 2025
    df_2025 = df[df['date'].dt.year == 2025]
    print(f"2025 Rows: {len(df_2025)}")
    if df_2025.empty:
        print("No data for 2025 in predictions file.")
        return

    pay = pd.read_parquet(PAYOUT_FILE)
    
    # Simple Strategy Simulation
    # Perfect Portfolio:
    # 1. Solid: Odds < 3.0, EV >= 1.0 -> Form 1->2-5->2-8 (24 pts)
    # 2. Long: Odds >= 10.0, EV >= 1.3 -> Nagashi 1->2-7 (30 pts, 3T)
    
    # Payout Map
    pm = {}
    for _, row in pay.iterrows():
        rid = row['race_id']
        pm[rid] = {'sanrentan': {}, 'wide': {}}
        for i in range(1, 9): # 1-8
            k_comb = f'haraimodoshi_sanrentan_{i}a'
            k_pay = f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb]:
                try:
                    pm[rid]['sanrentan'][str(row[k_comb])] = float(row[k_pay])
                except: pass

    total_cost = 0
    total_return = 0
    bets_count = 0
    hit_count = 0

    rids = df_2025['race_id'].unique()
    print(f"2025 Races: {len(rids)}")
    
    for rid in rids:
        race_df = df_2025[df_2025['race_id'] == rid].sort_values('score', ascending=False)
        top = race_df.iloc[0]
        
        # Recalc EV if needed (but v7 has it)
        if 'expected_value' in top:
            ev = top['expected_value']
        else:
            ev = top['prob'] * top['odds']
            
        odds = top['odds'] if not pd.isna(top['odds']) else 0
        h1 = int(top['horse_number'])
        
        cost = 0
        payout = 0
        is_hit = False
        
        # Solid
        if odds < 3.0 and ev >= 1.0:
            cost = 2400
            # Check results
            if rid in pm:
                # 2nd list: 2-5 (indices 1-4)
                # 3rd list: 2-8 (indices 1-7)
                opps = race_df.iloc[1:8] # up to index 7 (8th horse)
                rank_list = [int(x) for x in opps['horse_number']]
                if len(rank_list) >= 7:
                    r2_cands = rank_list[0:4] # 2,3,4,5
                    r3_cands = rank_list[0:7] # 2..8
                    
                    # Sanrentan check
                    for k, p in pm[rid]['sanrentan'].items():
                         # k usually '010203'
                         try:
                             w1 = int(k[0:2])
                             w2 = int(k[2:4])
                             w3 = int(k[4:6])
                             if w1 == h1 and w2 in r2_cands and w3 in r3_cands:
                                 payout += p
                         except: pass

        # Longshot
        elif odds >= 10.0 and ev >= 1.3:
            cost = 3000
            if rid in pm:
                opps = race_df.iloc[1:7] # 2-7
                opp_nums = [int(x) for x in opps['horse_number']]
                for k, p in pm[rid]['sanrentan'].items():
                     try:
                         w1 = int(k[0:2])
                         w2 = int(k[2:4])
                         w3 = int(k[4:6])
                         if w1 == h1 and w2 in opp_nums and w3 in opp_nums:
                             payout += p
                     except: pass
        
        if cost > 0:
            total_cost += cost
            total_return += payout
            bets_count += 1
            if payout > 0: hit_count += 1
            
    print(f"Bets: {bets_count}")
    print(f"Cost: {total_cost}")
    print(f"Return: {total_return}")
    if total_cost > 0:
        print(f"ROI: {total_return / total_cost * 100:.2f}%")
        print(f"Hit Rate: {hit_count / bets_count * 100:.2f}%")

if __name__ == "__main__":
    analyze()

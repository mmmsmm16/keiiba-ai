import pandas as pd
import numpy as np
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from inference.loader import InferenceDataLoader

import glob
import itertools

def analyze(file_pattern):
    print(f"Loading files matching: {file_pattern}")
    files = glob.glob(file_pattern)
    if not files:
        print("No files found.")
        return

    df_list = []
    for f in files:
        print(f"Reading {f}...")
        try:
            tmp = pd.read_csv(f)
            tmp['race_id'] = tmp['race_id'].astype(str)
            df_list.append(tmp)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not df_list:
        return
        
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total Rows Loaded: {len(df)}")
    
    # Rank fetching logic ...
    # (Kept as is)
    
    # 0. Fetch Payout Data (jvd_hr)
    print("Fetching Payout Data...")
    try:
        loader = InferenceDataLoader()
        rids = df['race_id'].unique().tolist()
        
        chunk_size = 1000
        payout_list = []
        
        for i in range(0, len(rids), chunk_size):
            chunk = rids[i:i+chunk_size]
            ids_str = ",".join([f"'{rid}'" for rid in chunk])
            
            # Fetch Umaren, Wide, Sanrenpuku, Sanrentan
            query = f"""
            SELECT
                CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
                haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
                haraimodoshi_umaren_2a, haraimodoshi_umaren_2b,
                haraimodoshi_umaren_3a, haraimodoshi_umaren_3b,
                haraimodoshi_wide_1a, haraimodoshi_wide_1b,
                haraimodoshi_wide_2a, haraimodoshi_wide_2b,
                haraimodoshi_wide_3a, haraimodoshi_wide_3b,
                haraimodoshi_wide_4a, haraimodoshi_wide_4b,
                haraimodoshi_wide_5a, haraimodoshi_wide_5b,
                haraimodoshi_wide_6a, haraimodoshi_wide_6b,
                haraimodoshi_wide_7a, haraimodoshi_wide_7b,
                haraimodoshi_sanrenpuku_1a, haraimodoshi_sanrenpuku_1b,
                haraimodoshi_sanrenpuku_2a, haraimodoshi_sanrenpuku_2b,
                haraimodoshi_sanrenpuku_3a, haraimodoshi_sanrenpuku_3b,
                haraimodoshi_sanrentan_1a, haraimodoshi_sanrentan_1b,
                haraimodoshi_sanrentan_2a, haraimodoshi_sanrentan_2b,
                haraimodoshi_sanrentan_3a, haraimodoshi_sanrentan_3b,
                haraimodoshi_sanrentan_4a, haraimodoshi_sanrentan_4b,
                haraimodoshi_sanrentan_5a, haraimodoshi_sanrentan_5b,
                haraimodoshi_sanrentan_6a, haraimodoshi_sanrentan_6b
            FROM jvd_hr
            WHERE CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) IN ({ids_str})
            """
            tmp = pd.read_sql(query, loader.engine)
            payout_list.append(tmp)
            
        payout_map = {}
        if payout_list:
            payout_df = pd.concat(payout_list)
            # Build fast lookup map
            for _, row in payout_df.iterrows():
                rid = row['race_id']
                if rid not in payout_map: 
                    payout_map[rid] = {
                        'umaren': {}, 'wide': {}, 
                        'sanrenpuku': {}, 'sanrentan': {}
                    }
                
                # Umaren (3 entries max)
                for k in range(1, 4):
                    comb = row.get(f'haraimodoshi_umaren_{k}a')
                    pay = row.get(f'haraimodoshi_umaren_{k}b')
                    if comb and pay and str(pay).strip():
                        try:
                            payout_map[rid]['umaren'][str(comb).strip()] = int(float(str(pay).strip()))
                        except: pass
                
                # Wide (7 entries max)
                for k in range(1, 8):
                    comb = row.get(f'haraimodoshi_wide_{k}a')
                    pay = row.get(f'haraimodoshi_wide_{k}b')
                    if comb and pay and str(pay).strip():
                        try:
                            payout_map[rid]['wide'][str(comb).strip()] = int(float(str(pay).strip()))
                        except: pass

                # Sanrenpuku (3 entries max)
                for k in range(1, 4):
                    comb = row.get(f'haraimodoshi_sanrenpuku_{k}a')
                    pay = row.get(f'haraimodoshi_sanrenpuku_{k}b')
                    if comb and pay and str(pay).strip():
                        try:
                            payout_map[rid]['sanrenpuku'][str(comb).strip()] = int(float(str(pay).strip()))
                        except: pass

                # Sanrentan (6 entries max)
                for k in range(1, 7):
                    comb = row.get(f'haraimodoshi_sanrentan_{k}a')
                    pay = row.get(f'haraimodoshi_sanrentan_{k}b')
                    if comb and pay and str(pay).strip():
                        try:
                            payout_map[rid]['sanrentan'][str(comb).strip()] = int(float(str(pay).strip()))
                        except: pass
                        
    except Exception as e:
        print(f"Payout Fetch Error: {e}")
        payout_map = {}

    # If rank is missing, fetch from DB (existing logic)
    if 'rank' not in df.columns or df['rank'].isnull().all():
        print("Rank missing or empty. Fetching from DB...")
        try:
            loader = InferenceDataLoader()
            rids = df['race_id'].unique().tolist()
            print(f"Fetching results for {len(rids)} races...")
            
            chunk_size = 1000
            results_list = []
            
            for i in range(0, len(rids), chunk_size):
                chunk = rids[i:i+chunk_size]
                ids_str = ",".join([f"'{rid}'" for rid in chunk])
                
                query = f"""
                SELECT
                    CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
                    umaban::integer AS horse_number,
                    kakutei_chakujun::integer AS rank
                FROM jvd_se
                WHERE CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) IN ({ids_str})
                """
                
                tmp = pd.read_sql(query, loader.engine)
                results_list.append(tmp)
            
            if results_list:
                res_df = pd.concat(results_list)
                print(f"Fetched {len(res_df)} results.")
                if 'race_id' in res_df.columns:
                     res_df['race_id'] = res_df['race_id'].astype(str)
                
                df = pd.merge(df, res_df, on=['race_id', 'horse_number'], how='left')
                
                if 'rank_x' in df.columns:
                     df['rank'] = df['rank_y'].fillna(df['rank_x'])
                     df = df.drop(columns=['rank_x', 'rank_y'])
            
        except Exception as e:
            print(f"DB Fetch Error: {e}")
            return
    
    if 'rank' not in df.columns:
        print("Error: Could not retrieve rank data.")
        return

    # Filter out races where rank is all null (future races)
    valid_races = df.groupby('race_id')['rank'].apply(lambda x: x.notnull().any())
    valid_race_ids = valid_races[valid_races].index
    
    df = df[df['race_id'].isin(valid_race_ids)].copy()
    
    if df.empty:
        print("No valid race results found (all ranks are null).")
        return

    print(f"Analyzed Races: {df['race_id'].nunique()}")
    print(f"Total Rows: {len(df)}")

    # Assign Predicted Rank based on Score
    df['pred_rank'] = df.groupby('race_id')['score'].rank(ascending=False, method='first')
    
    # 1. Top-N Accuracy (Prediction Rank 1 Horse)
    top1_df = df[df['pred_rank'] == 1]
    
    win_rate = (top1_df['rank'] == 1).mean()
    ren_rate = (top1_df['rank'] <= 2).mean()
    fuku_rate = (top1_df['rank'] <= 3).mean()
    
    print("\n=== ✨ Top-1 Predicted Horse Performance ===")
    print(f"Win Rate (単勝率): {win_rate:.1%}")
    print(f"Ren Rate (連対率): {ren_rate:.1%}")
    print(f"Fuku Rate (複勝率): {fuku_rate:.1%}")
    
    # 2. ROI Simulation (Flat Bet on Top-1)
    # Using 'odds' column (Win Odds)
    if 'odds' in df.columns:
        hits = top1_df[top1_df['rank'] == 1]
        # odds are typically already numeric (e.g. 3.4). If null, treat as 1.0 (refund) or 0 (loss)
        # Assuming odds are correct. Note: `odds` in prediction might be pre-race odds.
        # Ideally we verify with refund data, but this is an approximation.
        return_amount = hits['odds'].fillna(0).sum()
        cost = len(top1_df)
        roi = return_amount / cost * 100 if cost > 0 else 0
        print(f"\n=== 💰 Single Bet Simulation (Top-1) ===")
        print(f"Total Bets: {cost}")
        print(f"Total Return: {return_amount:.1f}")
        print(f"ROI: {roi:.1f}%")
    
    # 3. Correlation (Spearman)
    # Filter out non-finishers? rank is numeric.
    corr_df = df.dropna(subset=['rank', 'score'])
    spearman = corr_df.groupby('race_id').apply(lambda x: x['score'].corr(x['rank'], method='spearman')).mean()
    # Rank is 1=Best, Score is High=Best. Correlation should be NEGATIVE close to -1.
    print(f"\n=== 📊 Correlation (Score vs Rank) ===")
    print(f"Spearman Corr (Avg): {spearman:.3f} (Ideal: -1.0)")
    
    # 4. Confusion Matrix (Pred Rank vs Actual Rank)
    print(f"\n=== 🎯 Pred Rank Performance (Win/Place) ===")
    print(f"{'PredRank':<8} | {'Win Rate':<8} | {'Ren Rate':<8} | {'Place Rate':<10} | {'Avg Actual Rank':<15}")
    print(f"{'-'*8}-|-{'-'*8}-|-{'-'*8}-|-{'-'*10}-|-{'-'*15}")
    
    total_races = df['race_id'].nunique() # Define total_races for the new calculation logic
    
    for r in range(1, 6):
        sub = df[df['pred_rank'] == r]
        count = len(sub)
        if count == 0: continue
        
        win_r = len(sub[sub['rank'] == 1]) / total_races * 100
        ren_r = len(sub[sub['rank'] <= 2]) / total_races * 100
        place_r = len(sub[sub['rank'] <= 3]) / total_races * 100
        avg_rank = sub['rank'].mean()
        
        print(f"{r:<8} | {win_r:7.1f}% | {ren_r:7.1f}% | {place_r:9.1f}% | {avg_rank:.1f}")

    # 4.5 Box Hit Rate Simulation
    print("\n=== 📦 Box Statistics (Does the model capture winners?) ===")
    
    # Group by race
    races = df.groupby('race_id')
    n_races = len(races)
    
    top2_box_win = 0 # Top2 contains 1st place
    top3_box_win = 0 # Top3 contains 1st place
    top5_box_win = 0 # Top5 contains 1st place
    
    top2_box_ren = 0 # Top2 contains 1st AND 2nd (Umaren Hit)
    top3_box_ren = 0 # Top3 contains 1st AND 2nd
    top5_box_ren = 0 # Top5 contains 1st AND 2nd
    
    top3_box_3ren = 0 # Top3 contains 1st, 2nd, 3rd (Sanrenpuku Hit)
    top5_box_3ren = 0 # Top5 contains 1st, 2nd, 3rd
    
    for rid, group in races:
        # Sort by score
        sorted_g = group.sort_values('score', ascending=False)
        
        actual_1st = group[group['rank'] == 1]['horse_number'].values
        actual_2nd = group[group['rank'] == 2]['horse_number'].values
        actual_3rd = group[group['rank'] == 3]['horse_number'].values
        
        if len(actual_1st) == 0: continue # Data missing
        h1 = actual_1st[0]
        h2 = actual_2nd[0] if len(actual_2nd) > 0 else -1
        h3 = actual_3rd[0] if len(actual_3rd) > 0 else -1
        
        # Predicted sets
        top2 = set(sorted_g.head(2)['horse_number'].values)
        top3 = set(sorted_g.head(3)['horse_number'].values)
        top5 = set(sorted_g.head(5)['horse_number'].values)
        
        # Check Winner
        if h1 in top2: top2_box_win += 1
        if h1 in top3: top3_box_win += 1
        if h1 in top5: top5_box_win += 1
        
        # Check Umaren (1st & 2nd)
        if {h1, h2}.issubset(top2): top2_box_ren += 1
        if {h1, h2}.issubset(top3): top3_box_ren += 1
        if {h1, h2}.issubset(top5): top5_box_ren += 1
        
        # Check Sanrenpuku (1st, 2nd, 3rd)
        if {h1, h2, h3}.issubset(top3): top3_box_3ren += 1
        if {h1, h2, h3}.issubset(top5): top5_box_3ren += 1

    print(f"Total Races: {n_races}")
    print(f"Top 2 Box | Winner Included: {top2_box_win/n_races:.1%} | Umaren Hit: {top2_box_ren/n_races:.1%}")
    print(f"Top 3 Box | Winner Included: {top3_box_win/n_races:.1%} | Umaren Hit: {top3_box_ren/n_races:.1%} | 3Ren Hit: {top3_box_3ren/n_races:.1%}")
    print(f"Top 5 Box | Winner Included: {top5_box_win/n_races:.1%} | Umaren Hit: {top5_box_ren/n_races:.1%} | 3Ren Hit: {top5_box_3ren/n_races:.1%}")

    # 4.6 Score Difference Analysis
    print("\n=== ⚖️ Score Difference Analysis (Rank 1 vs Rank 2) ===")
    
    diff_data = []
    
    for rid, group in races:
        sorted_g = group.sort_values('score', ascending=False)
        if len(sorted_g) < 2: continue
        
        r1 = sorted_g.iloc[0]
        r2 = sorted_g.iloc[1]
        
        score_diff = r1['score'] - r2['score']
        
        winner_rank = 0 # 1=Rank1 won, 2=Rank2 won, 0=Other
        if r1['rank'] == 1: winner_rank = 1
        elif r2['rank'] == 1: winner_rank = 2
        
        diff_data.append({
            'race_id': rid,
            'diff': score_diff,
            'winner': winner_rank
        })
        
    if diff_data:
        ddf = pd.DataFrame(diff_data)
        
        # Create bins hardcoded for consistency or quartiles
        # Scores are likely uncalibrated logits or raw scores. 
        # Let's try Quartiles first.
        try:
            ddf['diff_group'] = pd.qcut(ddf['diff'], q=4)
            
            print("\nWin Rate by Score Difference (Quartiles):")
            print(f"{'Diff Range':<20} | {'Count':<5} | {'R1 Win':<8} | {'R2 Win':<8} | {'Other':<8}")
            print("-" * 60)
            
            grouped = ddf.groupby('diff_group', observed=False)
            for interval, g in grouped:
                total = len(g)
                w1 = (g['winner'] == 1).sum()
                w2 = (g['winner'] == 2).sum()
                wo = (g['winner'] == 0).sum()
                
                print(f"{str(interval):<20} | {total:<5} | {w1/total:7.1%} | {w2/total:7.1%} | {wo/total:7.1%}")
                
        except Exception as e:
            print(f"Stats Error: {e}")

    # 4.7 Adaptive Strategy Simulation
    print("\n=== 🧠 Adaptive Strategy Simulation ===")
    print("Strategy: If (ScoreR1 - ScoreR2) < Threshold -> Bet Rank 2, Else -> Bet Rank 1")
    print("-" * 70)
    print(f"{'Threshold':<10} | {'Target':<10} | {'Win Rate':<8} | {'Return':<8} | {'ROI':<8}")
    
    thresholds = [0.03, 0.05, 0.08, 0.10, 0.15]
    
    if diff_data:
        ddf = pd.DataFrame(diff_data)
        # We need odds for calculation.
        # Merge odds back to ddf
        # ddf has race_id. We need to look up odds for Rank 1 and Rank 2 in original df.
        
        # Helper to get odds
        def get_odds(rid, rank_order):
            # rank_order: 1 for Rank1 horse, 2 for Rank2 horse
            rows = df[(df['race_id'] == rid) & (df['pred_rank'] == rank_order)]
            if rows.empty: return 0.0
            return rows.iloc[0]['odds'] if not pd.isna(rows.iloc[0]['odds']) else 0.0

        ddf['odds_r1'] = ddf['race_id'].apply(lambda x: get_odds(x, 1))
        ddf['odds_r2'] = ddf['race_id'].apply(lambda x: get_odds(x, 2))
        
        for th in thresholds:
            total_cost = 0
            total_return = 0
            wins = 0
            
            for _, row in ddf.iterrows():
                diff = row['diff']
                winner = row['winner'] # 1=R1, 2=R2
                
                # Rule
                if diff < th:
                    bet_target = 2 # Bet Rank 2 (Ana/Rival)
                    bet_odds = row['odds_r2']
                else:
                    bet_target = 1 # Bet Rank 1 (Honmei)
                    bet_odds = row['odds_r1']
                    
                total_cost += 1 # Unit bet
                
                if winner == bet_target:
                    wins += 1
                    total_return += bet_odds
            
            roi = total_return / total_cost * 100 if total_cost > 0 else 0
            win_rate = wins / total_cost if total_cost > 0 else 0
            
            print(f"{th:<10.2f} | {'Mixed':<10} | {win_rate:8.1%} | {total_return:<8.1f} | {roi:8.1f}%")

    # 4.8 Adaptive Strategy (Umaren/Wide)
    if payout_map:
        print("\n=== 🧩 Adaptive Strategy (Umaren & Wide) ===")
        print("Strategy: If Score Diff < Threshold -> Buy Rank 1-2 (One Point)")
        print("-" * 80)
        print(f"{'Threshold':<10} | {'Ticket':<10} | {'Hit Rate':<8} | {'Return':<8} | {'ROI':<8}")
        
        for th in thresholds:
            # Umaren
            cost_u = 0; ret_u = 0; hit_u = 0
            # Wide
            cost_w = 0; ret_w = 0; hit_w = 0
            
            for _, row in ddf.iterrows():
                # Only bet if adaptive condition met? 
                # Or always bet R1-R2 but segment by threshold?
                # User asked for "Adaptive".
                # Strategy: 
                # If diff < th: BUY (Rank 1 - Rank 2)
                # Else: SKIP (or buy something else? No, usually 'Ken' or buy Tan-sho R1)
                
                # Let's verify R1-R2 buying ONLY when diff is small (Box logic)
                if row['diff'] < th: 
                    rid = row['race_id']
                    
                    # Need horse numbers for R1 and R2
                    # We need to fetch from original df
                    r1_h = df[(df['race_id'] == rid) & (df['pred_rank'] == 1)]['horse_number'].iloc[0]
                    r2_h = df[(df['race_id'] == rid) & (df['pred_rank'] == 2)]['horse_number'].iloc[0]
                    
                    comb = sorted([int(r1_h), int(r2_h)])
                    comb_str = f"{comb[0]:02}{comb[1]:02}"
                    
                    # Umaren
                    cost_u += 1
                    if rid in payout_map and comb_str in payout_map[rid]['umaren']:
                        ret_u += payout_map[rid]['umaren'][comb_str] / 100 # Adjust per unit? payout is per 100yen. Cost is 1 unit.
                        hit_u += 1

                    # Wide
                    cost_w += 1
                    if rid in payout_map and comb_str in payout_map[rid]['wide']:
                        ret_w += payout_map[rid]['wide'][comb_str] / 100
                        hit_w += 1
                        
            roi_u = ret_u / cost_u * 100 if cost_u > 0 else 0
            hit_rate_u = hit_u / cost_u if cost_u > 0 else 0
            
            roi_w = ret_w / cost_w * 100 if cost_w > 0 else 0
            hit_rate_w = hit_w / cost_w if cost_w > 0 else 0
            
            print(f"{th:<10.2f} | {'Umaren':<10} | {hit_rate_u:8.1%} | {ret_u:<8.1f} | {roi_u:8.1f}%")
            print(f"{th:<10.2f} | {'Wide':<10}   | {hit_rate_w:8.1%} | {ret_w:<8.1f} | {roi_w:8.1f}%")

    # 4.9 Advanced 3-Ren Formations
    if payout_map:
        print("\n=== 🎲 Advanced 3-Ren Formations Search ===")
        print("Testing various formation strategies for 3-Renpuku & 3-Rentan")
        print("-" * 100)
        print(f"{'Type':<12} | {'Strategy':<35} | {'Cost/R':<6} | {'Hit Rate':<8} | {'Return':<8} | {'ROI':<8}")

        strategies = [
            # 3-Renpuku
            ('SanRenPuku', 'Box 5', 10, lambda r1, r2, top: list(itertools.combinations(top[:5], 3))),
            ('SanRenPuku', 'Axis 1 -> Flow 6 (1-6)', 15, lambda r1, r2, top: [(r1, x, y) for x, y in itertools.combinations(top[1:7], 2)]), # Axis 1, choose 2 from 6 opps (Ranking 2-7). Count: 6C2=15.
            ('SanRenPuku', 'Axis 1-2 -> Flow 6 (2pts)', 5, lambda r1, r2, top: [(r1, r2, x) for x in top[2:7]]), # Axis 1-2 fixed, choose 1 from 5 opps (Ranking 3-7). Wait, Flow 6 means 6 opponents? Let's say opps=5 (Rank 3-7). top[2:7] is 5 horses. Count: 5.
            
            # 3-Rentan
            ('SanRenTan', 'Box 3 (Top 3)', 6, lambda r1, r2, top: list(itertools.permutations(top[:3], 3))), 
            ('SanRenTan', '1st Fixed > 2nd/3rd (Top 5)', 20, lambda r1, r2, top: [(r1, x, y) for x, y in itertools.permutations(top[1:6], 2)]), # 1 > 2,3,4,5,6 > 2,3,4,5,6. Count: 1 * 5 * 4 = 20.
            ('SanRenTan', '1st/2nd Fold > 3rd (Top 6)', 8, lambda r1, r2, top: [(x, y, z) for x, y in itertools.permutations([r1, r2], 2) for z in top[2:6]]), # 1,2 > 1,2 > 3,4,5,6. (1>2>3..6) + (2>1>3..6). Count: 2 * 4 = 8.
            # 1st/2nd Multi (Fold) > 3rd (Top 5) => 1,2 > 1,2 > 3,4,5 (2*3=6 points)
             ('SanRenTan', '1st/2nd Fold > 3rd (Top 5)', 6, lambda r1, r2, top: [(x, y, z) for x, y in itertools.permutations([r1, r2], 2) for z in top[2:5]]),
            
             # User Request: Rank 3-6 in 2nd Row (Betting against Rank 2 for 2nd place)
             # 1st (R1) > 2nd (R3-6) > 3rd (R2-6)
             # Top[0] is R1. Top[1] is R2. Top[2:6] is R3-R6 (4 horses).
             # 3rd row candidates: R2 (Top[1]) + R3-R6 (Top[2:6]) = Top[1:6]
             ('SanRenTan', 'Gap: 1 > 3-6 > 2-6', 16, lambda r1, r2, top: [(r1, x, y) for x in top[2:6] for y in top[1:6] if y != x]),
             
             # Sandwich: 1 > 3-6 > 2
             ('SanRenTan', 'Sandwich: 1 > 3-6 > 2', 4, lambda r1, r2, top: [(r1, x, r2) for x in top[2:6]]),
             
             # User Request: "1,2 - 3,4,5,6 - 1,2" (Double Sandwich)
             # 1st: R1, R2
             # 2nd: R3-R6
             # 3rd: R1, R2 (Must be different from 1st)
             # essentially: (1 > 3-6 > 2) + (2 > 3-6 > 1)
             ('SanRenTan', 'Double Sandwich (1,2 > 3-6 > 1,2)', 8, lambda r1, r2, top: [(r1, x, r2) for x in top[2:6]] + [(r2, x, r1) for x in top[2:6]]),
        ]
    
    # helper
    def print_segment_report(name, group_col):
        if group_col not in df.columns: return
        print(f"\n=== 🌍 Segmentation Analysis: {name} ===")
        print(f"{'Segment':<12} | {'Count':<5} | {'R1 Win':<6} | {'R2 Win':<6} | {'Umaren(1-2)':<11} | {'ROI(Adapt-Fold)':<14}")
        print("-" * 75)
        
        # Determine Adaptive Fold ROI logic per segment
        # Using simplified average for ROI estimation
        
        groups = df.groupby(group_col)
        for g_name, g_df in groups:
            count = g_df['race_id'].nunique()
            if count < 5: continue
            
            r1_win = g_df[g_df['pred_rank'] == 1]['rank'].eq(1).mean() * 100
            r2_win = g_df[g_df['pred_rank'] == 2]['rank'].eq(1).mean() * 100
            
            # Umaren 1-2 Hit Rate
            # Need to aggregate by race
            # Check if Rank 1 and Rank 2 horses in this race finished 1st and 2nd
            # g_df has one row per horse.
            # Pivot or iterate? Iterate group names?
            # Re-select from race_groups might be faster.
            
            hits_12 = 0
            hits_adapt = 0
            cost_adapt = 0
            return_adapt = 0
            
            # Filter main race loop for this segment
            # We need to map rid -> segment value
            segment_rids = g_df['race_id'].unique()
            
            for rid in segment_rids:
                if rid not in payout_map: continue
                # Re-use logic? Hard to call function.
                # Simplified check for Umaren 1-2
                
                # Get ranks
                # We need explicit horse numbers for Payout Map
                sub = df[df['race_id'] == rid]
                r1 = sub[sub['pred_rank'] == 1]
                r2 = sub[sub['pred_rank'] == 2]
                if r1.empty or r2.empty: continue
                
                r1_n = r1['horse_number'].iloc[0]
                r2_n = r2['horse_number'].iloc[0]
                r1_s = r1['score'].iloc[0]
                r2_s = r2['score'].iloc[0]
                
                # Umaren 1-2
                comb = sorted([int(r1_n), int(r2_n)])
                k = f"{comb[0]:02}{comb[1]:02}"
                if k in payout_map[rid]['umaren']:
                    hits_12 += 1
                    
                # Adaptive Fold (Diff 0.15)
                # If diff < 0.15: Multi (8pts). Else: 1st Fixed (20pts) -> simplified logic for simulation
                # Reuse the logic from Strategy blocks?
                diff = r1_s - r2_s
                
                # Let's use the exact Adaptive Fold logic
                top_nums = sub.sort_values('pred_rank')['horse_number'].astype(int).tolist()
                combos = []
                if diff < 0.15:
                    opps = top_nums[2:6] # Rank 3-6
                    combos = [(x, y, z) for x, y in itertools.permutations([int(r1_n), int(r2_n)], 2) for z in opps]
                else: 
                    opps = top_nums[1:6] # Rank 2-6
                    combos = [(int(r1_n), x, y) for x, y in itertools.permutations(opps, 2)]
                    
                race_bet = len(combos)
                race_ret = 0
                for c in combos:
                    k3 = f"{c[0]:02}{c[1]:02}{c[2]:02}"
                    if k3 in payout_map[rid]['sanrentan']:
                        race_ret += payout_map[rid]['sanrentan'][k3] / 100
                
                cost_adapt += race_bet
                return_adapt += race_ret
                if race_ret > 0: hits_adapt += 1
                
            umaren_rate = hits_12 / count * 100
            adapt_roi = return_adapt / cost_adapt * 100 if cost_adapt > 0 else 0
            
            print(f"{str(g_name):<12} | {count:<5} | {r1_win:5.1f}% | {r2_win:5.1f}% | {umaren_rate:5.1f}%      | {adapt_roi:6.1f}% ({hits_adapt/count*100:.1f}%)")

    # Add Date/Month column if missing
    if 'date' in df.columns:
        df['month'] = pd.to_datetime(df['date']).dt.month
        print_segment_report('Month', 'month')

    # Add Surface from 'surface' column or inference ??
    # 1=Turf, 2=Dirt (JRA JVD spec)
    # Check if 'surface' exists
    if 'surface' in df.columns:
        # Map Code
        # 10=Straight Turf?? 10..22 is Turf in JVD spec?
        # Standard: 1=Turf, 2=Dirt?
        # Actually loader query uses: CAST(r.track_code AS INTEGER) BETWEEN 10 AND 22 THEN 'Turf' ELSE 'Dirt' logic (simplified)
        # Wait, the loader output columns were: 
        # r.track_code AS surface
        # Let's map it.
        # JVD Track Codes: 10-22 = Turf, 23-29 = Dirt.
        def map_surface(c):
            s = str(c)
            if 'Turf' in s: return 'Turf'
            if 'Dirt' in s: return 'Dirt'
            try:
                ic = int(c)
                if 10 <= ic <= 22: return 'Turf'
                if 23 <= ic <= 29: return 'Dirt'
                return 'Other'
            except: return 'Unknown'
        
        df['surface_name'] = df['surface'].apply(map_surface)
        print_segment_report('Surface', 'surface_name')

    if 'venue' in df.columns:
        # Map Venue Codes (JRA)
        venues = {
            '01': 'Sapporo', '02': 'Hakodate', '03': 'Fukushima', '04': 'Niigata',
            '05': 'Tokyo', '06': 'Nakayama', '07': 'Chukyo', '08': 'Kyoto',
            '09': 'Hanshin', '10': 'Kokura'
        }
        df['venue_name'] = df['venue'].astype(str).str.zfill(2).map(venues).fillna(df['venue'])
        print_segment_report('Venue', 'venue_name')
        
    
    # Helper for Adaptive
    # If Diff < 0.15: Use "Axis 1-2 -> Flow (Rank 3-7)" (Strong 2 heads)
    # Else: Use "Axis 1 -> Flow (Rank 2-7)" (Strong 1 head)
    
    for st_type, st_name, st_cost, st_func in strategies:
        total_bet = 0
        total_ret = 0
        total_hit = 0
        race_count = 0
        
        for rid, group in races:
            if rid not in payout_map: continue
            
            # Get R1, R2, Top horses
            sorted_g = group.sort_values('score', ascending=False)
            if len(sorted_g) < 7: continue # Need enough horses
                
            top_nums = sorted_g['horse_number'].astype(int).tolist()
            r1_num = top_nums[0]
            r2_num = top_nums[1]
            
            # Stats for adaptive logic
            r1_score = sorted_g.iloc[0]['score']
            r2_score = sorted_g.iloc[1]['score']
            diff = r1_score - r2_score
            
            combos = []
            
            if st_func == 'adaptive_puku':
                if diff < 0.15:
                    # Close race: Axis 1-2 fixed, Flow to Rank 3-7 (5 horses) -> 5 pts
                    opps = top_nums[2:7]
                    combos = [(r1_num, r2_num, x) for x in opps]
                else:
                    # Clear fav: Axis 1 fixed, Flow to Rank 2-7 (6 horses) -> 15 pts
                    opps = top_nums[1:7]
                    combos = [(r1_num, x, y) for x, y in itertools.combinations(opps, 2)]
            
            elif st_func == 'adaptive_tan':
                if diff < 0.15:
                        # Close race: 1st/2nd Fold > 3rd (Rank 3-6) (4 horses) -> 1,2 > 1,2 > 3,4,5,6 (8 pts)
                        # High risk/High return
                        opps = top_nums[2:6]
                        combos = [(x, y, z) for x, y in itertools.permutations([r1_num, r2_num], 2) for z in opps]
                else:
                    # Clear fav: 1st Fixed > 2nd/3rd (Rank 2-6) (5 horses) -> 1 > 2,3,4,5,6 > 2,3,4,5,6 (20 pts)
                    opps = top_nums[1:6]
                    combos = [(r1_num, x, y) for x, y in itertools.permutations(opps, 2)]
                    
            else:
                # Static Strategy
                combos = st_func(r1_num, r2_num, top_nums)
            
            # Evaluate Combos
            race_hit = 0
            race_ret = 0
            race_bet = len(combos)
            
            for c in combos:
                if st_type == 'SanRenPuku':
                    c_sorted = sorted(c)
                    key = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
                    if key in payout_map[rid]['sanrenpuku']:
                        race_ret += payout_map[rid]['sanrenpuku'][key] / 100
                        race_hit = 1
                elif st_type == 'SanRenTan':
                    key = f"{c[0]:02}{c[1]:02}{c[2]:02}"
                    if key in payout_map[rid]['sanrentan']:
                            race_ret += payout_map[rid]['sanrentan'][key] / 100
                            race_hit = 1
                            
            total_bet += race_bet
            total_ret += race_ret
            if race_hit > 0: total_hit += 1
            race_count += 1
            
        roi = total_ret / total_bet * 100 if total_bet > 0 else 0
        hit_rate = total_hit / race_count if race_count > 0 else 0
        avg_cost = total_bet / race_count if race_count > 0 else 0
        
        print(f"{st_type:<12} | {st_name:<35} | {avg_cost:<6.1f} | {hit_rate:8.1%} | {total_ret:<8.1f} | {roi:8.1f}%")

    # 5. EV Analysis (if expected_value exists)

    # 5. EV Analysis (if expected_value exists)
    if 'expected_value' in df.columns:
        print("\n=== 💎 High EV Performance (EV > 1.2) ===")
        high_ev = df[df['expected_value'] > 1.2]
        if not high_ev.empty:
            hr = (high_ev['rank'] == 1).mean()
            fr = (high_ev['rank'] <= 3).mean()
            
            # ROI
            ret = high_ev[high_ev['rank']==1]['odds'].sum()
            cost = len(high_ev)
            ev_roi = ret/cost*100
            
            print(f"Count: {len(high_ev)}")
            print(f"Win Rate: {hr:.1%}")
            print(f"Fuku Rate: {fr:.1%}")
            print(f"ROI: {ev_roi:.1f}%")
        else:
            print("No horses with EV > 1.2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_pattern', type=str, help='Glob pattern for csv files')
    args = parser.parse_args()
    
    analyze(args.file_pattern)

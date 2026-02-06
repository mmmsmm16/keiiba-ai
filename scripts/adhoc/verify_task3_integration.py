
import pandas as pd
import numpy as np
import os
import sys
import time

sys.path.append(os.getcwd())
from src.probability.ticket_probabilities import compute_ticket_probs
from src.tickets.generate_candidates import generate_candidates
from src.odds.join_snapshot_odds import join_snapshot_odds

def main():
    # 1. Setup Mock Race (Use a real race ID from 2024/2025 ideally)
    # Let's find a valid race ID from the parquet file later, or just pick one if we know it.
    # If build is still running, maybe we can't read parquet yet.
    # But let's assume one exists or wait.
    
    # We will try to read 2024 parquet
    year = 2024
    snap_path = f"data/odds_snapshots/{year}/odds_T-10.parquet"
    
    # Wait for file
    print(f"Waiting for {snap_path}...")
    max_wait = 600
    start = time.time()
    while not os.path.exists(snap_path):
        if time.time() - start > max_wait:
            print("Timeout waiting for parquet.")
            return
        time.sleep(10)
        
    print("Loading snapshots to pick a race...")
    df_odds = pd.read_parquet(snap_path)
    race_id = df_odds['race_id'].iloc[0]
    print(f"Target Race: {race_id}")
    
    # 2. Mock Probabilities (We don't need real model here, just structure)
    # We need horse numbers.
    race_odds = df_odds[df_odds['race_id'] == race_id]
    sample_win = race_odds[race_odds['ticket_type'] == 'win']
    horses = sorted([int(c) for c in sample_win['combination'].tolist() if c.isdigit()])
    
    # Fake predicions
    # Assign prob proportional to inverse odds?
    p_fake = []
    for h in horses:
        p_fake.append(np.random.rand())
    
    # Create DF
    # Frames? Need mock frames.
    # 10 horses -> 1..8
    frames = [(h+1)//2 for h in range(len(horses))]
    frames = [f if f<=8 else 8 for f in frames] # limit to 8
    
    race_df = pd.DataFrame({
        'horse_number': horses,
        'frame_number': frames[:len(horses)],
        'pred_prob': p_fake
    })
    
    # 3. Compute Probs
    print("Computing Ticket Probs...")
    probs = compute_ticket_probs(race_df, n_samples=5000)
    
    # 4. Generate Candidates
    print("Generating Candidates...")
    candidates = generate_candidates(race_id, probs)
    print(f"Candidates Count: {len(candidates)}")
    print(candidates['ticket_type'].value_counts())
    
    # 5. Join Odds
    print("Joining Odds...")
    joined = join_snapshot_odds(candidates, year, snapshot_mode='T-10')
    
    # 6. Verify
    print("Verification:")
    
    # Check if Win/Place/Umaren have odds
    for t in ['win', 'place', 'umaren', 'wakuren']:
        subset = joined[joined['ticket_type'] == t]
        if subset.empty:
            print(f"- {t}: No candidates.")
        else:
            has_odds = subset['odds'].notna().sum()
            print(f"- {t}: {len(subset)} candidates, {has_odds} with odds.")
            if has_odds > 0:
                print(f"  Sample: {subset.iloc[0].to_dict()}")
                
    # Check EV
    has_ev = joined['ev'].notna().sum()
    print(f"Total entries with EV: {has_ev}/{len(joined)}")
    
    # Assert some success
    if has_ev > 0:
        print("SUCCESS: Task 3 Integration Verified.")
    else:
        print("FAILURE: No EV calculated.")

if __name__ == "__main__":
    main()

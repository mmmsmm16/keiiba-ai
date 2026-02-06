
import sys
import os
sys.path.append(os.getcwd())
from src.data.realtime_loader import RealTimeDataLoader

def test_loader():
    print("Testing RealTimeDataLoader...")
    loader = RealTimeDataLoader()
    
    # Needs a valid RaceID that exists in apd_sokuho_o1/o2
    # From previous sample: 2025(Y) 09(V) 05(K) 04(N) 05(R) ?? 
    # Sample row had: kaisai_nen=2025, keibajo=09, kai=05, nichi=04, race=05
    # RaceID = 202509050405
    
    test_race_id = "202509050405"
    
    print(f"Fetching WIN odds for {test_race_id}...")
    win_odds = loader.get_latest_odds([test_race_id], 'win')
    print(f"Win Odds: {win_odds}")
    
    print(f"Fetching UMAREN odds for {test_race_id}...")
    uma_odds = loader.get_latest_odds([test_race_id], 'umaren')
    
    print("Umaren Odds (First 5):")
    if uma_odds and test_race_id in uma_odds:
        sorted_keys = sorted(uma_odds[test_race_id].keys())[:5]
        for k in sorted_keys:
            print(f"  {k}: {uma_odds[test_race_id][k]}")
    else:
        print("No Umaren odds found.")

if __name__ == "__main__":
    test_loader()

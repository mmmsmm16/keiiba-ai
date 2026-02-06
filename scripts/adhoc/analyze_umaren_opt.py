import pandas as pd
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.simulation.optimize_umaren_frequency import evaluate_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze():
    df = pd.read_csv("reports/simulations/umaren_optimization_train.csv")
    
    # Filter ROI >= 100%
    valid = df[df['roi'] >= 1.0].copy()
    
    if valid.empty:
        logger.warning("No config with ROI >= 100% found!")
        # Fallback: ROI >= 90%
        valid = df[df['roi'] >= 0.9].copy()
    
    # Sort by Hit Day Rate
    df_sorted = valid.sort_values('hit_day_rate', ascending=False)
    
    print("=== Top 10 by Hit Day Rate (ROI>=1.0) ===")
    print(df_sorted.head(10)[['e_th', 'p_th', 'max_pairs', 'roi', 'hit_day_rate', 'race_hit_rate', 'bets']].to_string())
    
    # Check Test Set for Top 3
    print("\n=== Validating on Test (2024) ===")
    
    # Load Candidates
    cand_path = "reports/simulations/umaren_candidates.parquet"
    if not os.path.exists(cand_path):
        print("Candidates file not found.")
        return
        
    full_cand = pd.read_parquet(cand_path)
    test_cand = full_cand[full_cand['year'] == 2024].copy()
    
    top_configs = df_sorted.head(5)
    
    for _, row in top_configs.iterrows():
        params = {
            'e_th': row['e_th'],
            'p_th': row['p_th'],
            'max_pairs': int(row['max_pairs'])
        }
        res = evaluate_strategy(test_cand, params)
        print(f"Config: {params}")
        print(f"  Train: ROI={row['roi']:.2f}, HitDay={row['hit_day_rate']:.3f}")
        print(f"  Test : ROI={res['roi']:.2f}, HitDay={res['hit_day_rate']:.3f}, RaceHit={res['race_hit_rate']:.3f}, Bets={res['bets']}")
        print("-" * 30)

if __name__ == "__main__":
    analyze()

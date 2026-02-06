
import os
import sys
import pandas as pd
import itertools
from datetime import datetime
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.simulate_kelly_2025 import KellySimulator, RANKER_MODEL_PATH

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_optimization(start_date, end_date):
    # Parameter Grid
    # Based on user findings: fraction ~ 0.05, threshold ~ 1.5, min_prob ~ 0.15 for good ROI.
    # We explore around these values.
    
    thresholds = [1.2, 1.5, 2.0]
    min_probs = [0.05, 0.10, 0.15]
    min_odds_list = [1.5, 2.0]
    fractions = [0.05]
    
    results = []
    
    param_grid = list(itertools.product(thresholds, min_probs, min_odds_list, fractions))
    total_runs = len(param_grid)
    
    logger.info(f"Starting Grid Search. Total combinations: {total_runs}")
    logger.info(f"Period: {start_date} ~ {end_date}")
    
    # Pre-load data
    logger.info("Pre-loading data...")
    dummy_sim = KellySimulator(RANKER_MODEL_PATH)
    preloaded_df = dummy_sim.prepare_data(start_date, end_date)
    
    # Pre-predict to save time (Optional, simulation handles it, but safer to do it once if model deterministic)
    # The modified run_simulation handles prediction if needed.
    
    logger.info("Data loaded. Starting Optimization Loop...")
    
    for i, (th, mp, mo, fr) in enumerate(param_grid):
        logger.info(f"--- Run {i+1}/{total_runs}: Th={th}, MinP={mp}, MinO={mo}, Fr={fr} ---")
        
        sim = KellySimulator(RANKER_MODEL_PATH, kelly_fraction=fr, threshold=th, min_prob=mp, min_odds=mo)
        
        try:
            # Pass preloaded data
            stats = sim.run_simulation(start_date, end_date, preloaded_df=preloaded_df)
            
            res = {
                'threshold': th,
                'min_prob': mp,
                'min_odds': mo,
                'fraction': fr,
                **stats
            }
            results.append(res)
            
            # Intermediate save
            pd.DataFrame(results).to_csv("strategy_benchmark_temp.csv", index=False)
        except Exception as e:
            logger.error(f"Run failed: {e}")
            import traceback
            traceback.print_exc()
        
    df = pd.DataFrame(results)
    df.to_csv("strategy_benchmark.csv", index=False)
    
    print("\n=== Optimization Results ===")
    print(df.sort_values("profit", ascending=False).head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default='2025-01-01')
    parser.add_argument("--end_date", type=str, default='2025-06-30')
    args = parser.parse_args()
    
    run_optimization(args.start_date, args.end_date)

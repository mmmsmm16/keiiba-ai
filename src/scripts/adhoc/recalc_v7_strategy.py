import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.pipeline.config import ExperimentConfig
from src.pipeline.strategy import optimize_strategies
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    config_path = "config/experiments/1stmodel.yaml"
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    config = ExperimentConfig.load(config_path)
    
    # Force strategy enabled
    config.strategy.enabled = True
    
    # Path to v7 output
    run_dir = os.path.join("experiments", "v7_ensemble_full")
    
    if not os.path.exists(run_dir):
        print(f"Run dir not found: {run_dir}")
        return

    print("Running strategy optimization for v7...")
    optimize_strategies(config, run_dir)
    print("Done.")

if __name__ == "__main__":
    main()

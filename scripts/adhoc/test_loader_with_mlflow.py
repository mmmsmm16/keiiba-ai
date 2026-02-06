
import sys
import os
import logging
import mlflow

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Setting MLFlow...")
    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment("debug_experiment")
    
    logger.info("Initializing JraVanDataLoader...")
    loader = JraVanDataLoader()
    
    logger.info("Starting MLFlow run...")
    with mlflow.start_run(run_name="debug_run"):
        logger.info("loading 100 rows with 2-year range scan...")
        try:
            # 2024-2025, no limit (similar to run_experiment failure case)
            df = loader.load(limit=None, history_start_date="2024-01-01", end_date="2025-12-31", skip_odds=True, skip_training=True)
            logger.info(f"Loaded {len(df)} rows.")
            print(df.head())
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()

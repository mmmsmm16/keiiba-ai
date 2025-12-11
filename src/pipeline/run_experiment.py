import argparse
import sys
import os
import logging
from datetime import datetime

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.pipeline.config import ExperimentConfig

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.pipeline.data import prepare_data

def run_data_step(config: ExperimentConfig, run_dir: str):
    logger.info(">>> Step 1: Data Preparation")
    logger.info(f"Target Years: {config.data.train_years}, Valid: {config.data.valid_year}")
    prepare_data(config, run_dir)

from src.pipeline.train import train_model
from src.pipeline.evaluate import evaluate_model

def run_train_step(config: ExperimentConfig, run_dir: str):
    logger.info(">>> Step 2: Model Training")
    logger.info(f"Model Type: {config.model.type}")
    train_model(config, run_dir)

def run_evaluate_step(config: ExperimentConfig, run_dir: str):
    logger.info(">>> Step 3: Evaluation")
    evaluate_model(config, run_dir)

from src.pipeline.strategy import optimize_strategies

def run_strategy_step(config: ExperimentConfig, run_dir: str):
    logger.info(">>> Step 4: Strategy Optimization")
    optimize_strategies(config, run_dir)

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation step")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training step")
    parser.add_argument("--eval-only", action="store_true", help="Run only evaluation and strategy steps")
    args = parser.parse_args()
    
    # Config Load
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
        
    try:
        config = ExperimentConfig.load(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Setup Dirs
    run_dir = config.setup_dirs()
    logger.info(f"🚀 Starting Experiment: {config.experiment_name}")
    logger.info(f"📂 Output Directory: {run_dir}")
    
    # Save current config to run_dir for reproducibility
    config_backup_path = os.path.join(run_dir, "config_snapshot.yaml")
    with open(config_backup_path, "w", encoding='utf-8') as f:
        import yaml
        yaml.dump(config.model_dump(), f, default_flow_style=False, allow_unicode=True)

    # eval-only implies skip-data and skip-train
    if args.eval_only:
        args.skip_data = True
        args.skip_train = True
        logger.info("🎯 eval-only mode: Skipping data and training steps")

    # Run Steps
    try:
        if not args.skip_data:
            run_data_step(config, run_dir)
        else:
            logger.info("⏭️ Step 1: Data Preparation [SKIPPED]")
            
        if not args.skip_train:
            run_train_step(config, run_dir)
        else:
            logger.info("⏭️ Step 2: Model Training [SKIPPED]")
            
        run_evaluate_step(config, run_dir)
        run_strategy_step(config, run_dir)
        
        logger.info("✅ Experiment Completed Successfully!")
        
    except Exception as e:
        logger.error(f"🔥 Experiment Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

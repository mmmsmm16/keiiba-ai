
import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_step(command, description):
    logger.info("="*60)
    logger.info(f"STARTING: {description}")
    logger.info(f"CMD: {command}")
    logger.info("="*60)
    
    start_time = time.time()
    try:
        # Stream output to console
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=sys.stdout, 
            stderr=subprocess.STDOUT
        )
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"❌ FAILED: {description} (Exit Code: {process.returncode})")
            sys.exit(process.returncode)
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION: {description} - {e}")
        sys.exit(1)
        
    duration = time.time() - start_time
    logger.info(f"✅ COMPLETED: {description} (Time: {duration/60:.1f} min)")
    logger.info("-" * 60)

def main():
    logger.info("🚀 STARTING FULL RETRAINING PIPELINE")
    logger.info("Target: Top2, Top3, LambdaRank (Win Model SKIPPED/DONE)")
    
    # 1. Win Model (Binary) - DONE (AUC 0.7973)
    # run_step("python scripts/run_train_optuna_best.py", "1/4 Win Model (Binary)")
    
    # 2. Top2 Model (Optimized + Full Train)
    # Optimized script: Sampling + Pruning -> Faster trials. Increased to 20.
    run_step("python scripts/run_optuna_top2_top3.py --target top2 --train_full --n_trials 20", "2/4 Top2 Model (HPO + Train)")
    
    # 3. Top3 Model (Optimized + Full Train)
    run_step("python scripts/run_optuna_top2_top3.py --target top3 --train_full --n_trials 20", "3/4 Top3 Model (HPO + Train)")
    
    # 4. LambdaRank Model (Ranking)
    run_step("python scripts/experiments/lambdarank_experiment.py", "4/4 LambdaRank Model")
    
    logger.info("🎉 ALL MODELS RETRAINED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

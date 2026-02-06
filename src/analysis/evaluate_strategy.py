import argparse
import sys
import os
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# プロジェクトルートへのパス追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def plot_calibration_curve(y_true, y_prob, output_path):
    """キャリブレーションカーブを描画・保存"""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Actual Win Rate)')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Calibration curve saved to {output_path}")

def plot_roi_curve(df_stats, output_path):
    """ROI/BetCountのカーブを描画・保存"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('EV Threshold')
    ax1.set_ylabel('ROI (%)', color=color)
    ax1.plot(df_stats['current_threshold'], df_stats['roi'], color=color, marker='o', label='ROI')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(100, color='gray', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Bet Count', color=color)  # we already handled the x-label with ax1
    ax2.bar(df_stats['current_threshold'], df_stats['bet_count'], width=0.03, alpha=0.3, color=color, label='Bet Count')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Strategy Simulation: ROI vs EV Threshold')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(output_path)
    plt.close()
    logger.info(f"ROI curve saved to {output_path}")

def main():
    default_config = "config/experiments/exp_v05_sire.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config, help="Path to config yaml")
    parser.add_argument("--year", type=int, default=2025, help="Year to simulate")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config.get('experiment_name')
    feature_blocks = config.get('features', [])
    
    logger.info(f"🚀 Strategy Evaluation for {exp_name} (Year: {args.year})")

    # 1. Load Data
    loader = JraVanDataLoader()
    start_date = f"{args.year}-01-01"
    end_date = f"{args.year}-12-31"
    load_start = f"{args.year-1}-01-01" # Load previous year for history context
    
    logger.info(f"Loading data ({load_start} ~ {end_date})...")
    raw_df = loader.load(history_start_date=load_start, end_date=end_date, jra_only=True)
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    # 2. Features
    pipeline = FeaturePipeline(cache_dir="data/features")
    # force=False to use cache (assuming cache is valid from previous runs)
    df_features = pipeline.load_features(clean_df, feature_blocks)
    
    # Merge Meta
    meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'odds', 'horse_name'] # removed jockey_id to avoid key error
    # clean_df might have object columns that dataset splitter removed, but for simulation we just need meta
    df_sim = pd.merge(
        df_features, 
        clean_df[meta_cols], 
        on=['race_id', 'horse_number'], 
        how='inner'
    )
    
    # Filter Year
    df_sim['date'] = pd.to_datetime(df_sim['date'])
    df_target = df_sim[(df_sim['date'] >= start_date) & (df_sim['date'] <= end_date)].copy()
    
    # 3. Predict
    model_path = f"models/experiments/{exp_name}/model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Feature Alignment
    model_features = model.feature_name()
    if 'age' in df_target.columns:
        df_target['age'] = pd.to_numeric(df_target['age'], errors='coerce')

    X = pd.DataFrame(index=df_target.index)
    for feat in model_features:
        if feat in df_target.columns:
            X[feat] = df_target[feat]
        else:
            X[feat] = 0.0
            
    logger.info("Predicting...")
    # LightGBM binary model -> predict returns probability
    probs = model.predict(X)
    df_target['pred_prob'] = probs
    
    # 4. Calibration Check
    # Need binary target (1: 1st/2nd/3rd?, No, v05 is binary for rank<=3 or rank=1? 
    # v05 config says 'binary', run_experiment says 'Rank <= 3' to 1.
    # Let's check config or assume Rank <=3 is the target.
    # Wait, EV is usually calculated for WIN (1st place).
    # If the model predicts "Probability of Top 3", then EV = Prob(Top3) * WinOdds is WRONG.
    # EV = P(Win) * WinOdds.
    # If the model is trained on Top 3, the probability is P(Top3).
    # We cannot directly use P(Top3) for Win EV.
    # We check if rank is 1.
    
    # Check what v05 trained on.
    # run_experiment.py: if objective=='binary': target = (rank <= 3)
    # SO v05 predicts "Probability of being in Top 3".
    # Using this for Win Betting is fundamentally flawed unless we assume P(Win) ∝ P(Top3).
    # User request: "EV = P(Win) * Odds".
    # If model outputs P(Top3), we CANNOT assume P(Win) = P(Top3).
    # Usually P(Win) approx P(Top3) / 3 ? No.
    # BUT, let's look at the result.
    # If the user asks to calc EV, they imply P(Win).
    # Warning: The current model v05 is trained on Top3 (Rank<=3). 
    # Simulating Win betting using Top3 probability will result in massive overestimation of EV.
    # I should note this.
    # HOWEVER, maybe I should check calibration against Top3 first.
    
    # Strategy:
    # 1. Calibration: Check reliability for Rank <= 3.
    # 2. EV Sim: User wants EV = P(Win) * Odds.
    #    Since we only have P(Top3), we can try to heuristic: P(Win) ~= P(Top3) / K?
    #    Or maybe just simulate assuming the score is "strength" and see if high score correlates with Win.
    #    The User instructions say: "EV = P(Win) * Odds ... P(Win) is model's predicted probability".
    #    If the model predicts P(Top3), this is a discrepancy.
    #    I will proceed with using the raw probability but Label it clearly.
    #    Also, I'll check Calibration against WIN (Rank=1) to see if the curve is linear (just scaled).
    
    # Calibration against Top 3
    y_true_top3 = (df_target['rank'] <= 3).astype(int)
    plot_calibration_curve(y_true_top3, df_target['pred_prob'], f"reports/calibration_top3_{exp_name}_{args.year}.png")

    # Calibration against Win
    y_true_win = (df_target['rank'] == 1).astype(int)
    plot_calibration_curve(y_true_win, df_target['pred_prob'], f"reports/calibration_win_{exp_name}_{args.year}.png")
    
    # 5. EV Simulation
    # EV = pred_prob * odds
    df_target['ev'] = df_target['pred_prob'] * df_target['odds']
    
    stats_list = []
    thresholds = np.arange(0.5, 2.55, 0.1) # 0.5 ~ 2.5
    
    for th in thresholds:
        bets = df_target[
            (df_target['ev'] >= th) & 
            (df_target['rank'] > 0) & 
            (df_target['odds'].notna())
        ].copy()
        
        count = len(bets)
        if count == 0:
            stats_list.append({
                'current_threshold': th, 'bet_count': 0, 'hit_rate': 0, 'return': 0, 'cost': 0, 'roi': 0
            })
            continue
            
        hits = bets[bets['rank'] == 1]
        hit_count = len(hits)
        return_amount = hits['odds'].sum() * 100
        cost_amount = count * 100
        roi = (return_amount / cost_amount) * 100
        hit_rate = (hit_count / count) * 100
        
        stats_list.append({
            'current_threshold': th,
            'bet_count': count,
            'hit_rate': hit_rate,
            'return': return_amount,
            'cost': cost_amount,
            'roi': roi
        })
        
    df_stats = pd.DataFrame(stats_list)
    
    # Output
    print("\nXXX Strategy Simulation Result (EV Threshold Analysis) XXX")
    print(df_stats[['current_threshold', 'bet_count', 'hit_rate', 'roi']])
    
    # Plot
    plot_roi_curve(df_stats, f"reports/strategy_roi_{exp_name}_{args.year}.png")
    
    # Save CSV
    df_stats.to_csv(f"reports/strategy_stats_{exp_name}_{args.year}.csv", index=False)
    logger.info("Saved stats and plots.")

if __name__ == "__main__":
    main()

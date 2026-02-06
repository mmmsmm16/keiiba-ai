import argparse
import sys
import os
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
import joblib
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

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

def load_data_and_features(year_start, year_end, feature_blocks):
    loader = JraVanDataLoader()
    # History contextのため前年からロード
    load_start = f"{year_start-1}-01-01"
    load_end = f"{year_end}-12-31"
    
    logger.info(f"Loading data ({load_start} ~ {load_end})...")
    raw_df = loader.load(history_start_date=load_start, end_date=load_end, jra_only=True)
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    pipeline = FeaturePipeline(cache_dir="data/features")
    df_features = pipeline.load_features(clean_df, feature_blocks)
    
    # Merge Meta
    meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'odds', 'horse_name']
    df_merged = pd.merge(
        df_features, 
        clean_df[meta_cols], 
        on=['race_id', 'horse_number'], 
        how='inner'
    )
    
    # Filter pure range
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    df_filtered = df_merged[
        (df_merged['date'] >= f"{year_start}-01-01") & 
        (df_merged['date'] <= f"{year_end}-12-31")
    ].copy()
    
    return df_filtered

def align_and_predict(model, df):
    model_features = model.feature_name()
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    X = pd.DataFrame(index=df.index)
    for feat in model_features:
        if feat in df.columns:
            X[feat] = df[feat]
        else:
            X[feat] = 0.0
            
    return model.predict(X)

def main():
    default_config = "config/experiments/exp_v05_sire.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config.get('experiment_name')
    feature_blocks = config.get('features', [])
    
    # Model Load
    model_path = f"models/experiments/{exp_name}/model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info("🚀 Isotonic Calibration Strategy")
    
    # 1. Calibration Data (2024)
    # Use 2024 as calibration set (It was valid set, so acceptable to use for calibration training)
    logger.info("--- Phase 1: Training Calibration Model (2024) ---")
    df_calib = load_data_and_features(2024, 2024, feature_blocks)
    
    # Predict Raw (P_Top3)
    raw_preds_calib = align_and_predict(model, df_calib)
    df_calib['raw_prob'] = raw_preds_calib
    
    # Target: Win (Rank=1)
    df_calib['is_win'] = (df_calib['rank'] == 1).astype(int)
    
    # Train Isotonic Regression
    # X: Raw Probability (Top3), y: Actual Win (0/1)
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso_reg.fit(df_calib['raw_prob'], df_calib['is_win'])
    
    # Visualization: Mapping
    x_range = np.linspace(0, 1, 100)
    y_mapped = iso_reg.predict(x_range)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_range, y_mapped, label='Mapping Function')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5)
    plt.xlabel('Original Prediction (P_Top3)')
    plt.ylabel('Calibrated Probability (P_Win)')
    plt.title('Isotonic Calibration Mapping: P(Top3) -> P(Win)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"reports/calibration_mapping_{exp_name}.png")
    logger.info("Saved calibration mapping plot.")
    
    # 2. Simulation (2025)
    logger.info("--- Phase 2: Simulation 2025 with Calibrated Prob ---")
    df_test = load_data_and_features(2025, 2025, feature_blocks)
    
    # Predict Raw
    raw_preds_test = align_and_predict(model, df_test)
    df_test['raw_prob'] = raw_preds_test
    
    # Apply Calibration
    df_test['calib_prob_win'] = iso_reg.predict(df_test['raw_prob'])
    
    # EV Calculation
    df_test['ev'] = df_test['calib_prob_win'] * df_test['odds']
    
    # ROI Simulation
    stats_list = []
    thresholds = np.arange(0.5, 2.05, 0.1)
    
    for th in thresholds:
        bets = df_test[
            (df_test['ev'] >= th) & 
            (df_test['rank'] > 0) & 
            (df_test['odds'].notna())
        ].copy()
        
        count = len(bets)
        if count == 0:
            stats_list.append({'th': th, 'bets': 0, 'roi': 0, 'hit': 0})
            continue
            
        hits = bets[bets['rank'] == 1]
        return_amount = hits['odds'].sum() * 100
        cost_amount = count * 100
        roi = (return_amount / cost_amount) * 100
        hit_rate = (len(hits) / count) * 100
        
        stats_list.append({
            'threshold': th,
            'bet_count': count,
            'hit_rate': hit_rate,
            'roi': roi,
            'profit': return_amount - cost_amount
        })
        
    df_stats = pd.DataFrame(stats_list)
    print("\nXXX Isotonic Win Strategy Result (2025) XXX")
    print(df_stats)
    
    # Save
    out_csv = f"reports/strategy_isotonic_{exp_name}_2025.csv"
    df_stats.to_csv(out_csv, index=False)
    
    # Save Model
    joblib.dump(iso_reg, f"models/experiments/{exp_name}/isotonic_calibrator.pkl")
    logger.info("Saved isotonic model and stats.")

if __name__ == "__main__":
    main()

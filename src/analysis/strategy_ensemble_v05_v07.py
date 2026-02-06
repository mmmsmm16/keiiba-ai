import argparse
import sys
import os
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
import seaborn as sns
import matplotlib.pyplot as plt

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

def load_model(exp_name):
    model_path = f"models/experiments/{exp_name}/model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_with_model(model, df):
    # Model's feature names
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
    # Hardcoded config for v05 as base (assuming features are same for v07)
    base_config_path = "config/experiments/exp_v05_sire.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025, help="Year to simulate")
    args = parser.parse_args()

    config = load_config(base_config_path)
    feature_blocks = config.get('features', [])
    
    logger.info(f"🚀 Ensemble Analysis (v05 Top3 + v07 Win) for Year {args.year}")

    # 1. Load Data
    loader = JraVanDataLoader()
    start_date = f"{args.year}-01-01"
    end_date = f"{args.year}-12-31"
    load_start = f"{args.year-1}-01-01"
    
    logger.info(f"Loading data ({load_start} ~ {end_date})...")
    raw_df = loader.load(history_start_date=load_start, end_date=end_date, jra_only=True)
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    # 2. Features
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
    
    # Filter 2025
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    df_test = df_merged[
        (df_merged['date'] >= start_date) & 
        (df_merged['date'] <= end_date) &
        (df_merged['odds'].notna())
    ].copy()
    
    # 3. Predict v05 (Top3)
    logger.info("Predicting with v05_sire (P_Top3)...")
    model_v05 = load_model("v05_sire")
    df_test['p_top3'] = predict_with_model(model_v05, df_test)
    
    # 4. Predict v07 (Win)
    logger.info("Predicting with v07_win_binary (P_Win)...")
    model_v07 = load_model("v07_win_binary")
    df_test['p_win'] = predict_with_model(model_v07, df_test)
    
    # 5. Calculate Metrics
    # WinPotential = P_Win / P_Top3
    # P_Top3 could be 0, so add epsilon or clip
    df_test['win_potential'] = df_test['p_win'] / (df_test['p_top3'] + 1e-6)
    
    # EV using P_Win
    df_test['ev_win'] = df_test['p_win'] * df_test['odds']
    
    # 6. Segment Analysis (Heatmap)
    logger.info("Generating Heatmap Data...")
    
    # Binning
    df_test['p_win_bin'] = pd.cut(df_test['p_win'], bins=np.arange(0, 1.05, 0.1), labels=np.arange(0, 0.95, 0.1))
    df_test['potential_bin'] = pd.cut(df_test['win_potential'], bins=np.arange(0, 1.05, 0.1), labels=np.arange(0, 0.95, 0.1))
    
    # Check if bins are populated
    logger.info(f"P_Win distribution:\n{df_test['p_win_bin'].value_counts().sort_index()}")
    logger.info(f"Potential distribution:\n{df_test['potential_bin'].value_counts().sort_index()}")

    pivot_roi = df_test.pivot_table(
        index='p_win_bin', 
        columns='potential_bin', 
        values='odds', 
        aggfunc=lambda x: (x[df_test.loc[x.index, 'rank']==1].sum() / len(x) * 100) if len(x) > 10 else np.nan
    )
    
    logger.info(f"Pivot table shape: {pivot_roi.shape}")
    if pivot_roi.isnull().all().all():
         logger.warning("Pivot ROI is empty. Skipping Heatmap.")
    else:
        # Save Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_roi, annot=True, fmt=".1f", cmap="RdYlGn", center=100)
        plt.title("ROI Heatmap: P_Win vs WinPotential")
        plt.ylabel("P_Win (Confidence)")
        plt.xlabel("WinPotential (P_Win / P_Top3)")
        plt.savefig("reports/ensemble_roi_heatmap.png")
        logger.info("Saved ROI heatmap.")
    
    # 7. Simulation with Composite Logic
    logger.info("Simulating Composite Strategy...")
    # Grid Search for best threshold pair
    best_roi = 0
    best_params = {}
    
    results = []
    
    # Sweep EV threshold and Potential threshold
    ev_thresholds = [0.8, 1.0, 1.2, 1.5]
    pot_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    for ev_th in ev_thresholds:
        for pot_th in pot_thresholds:
            # Condition: EV > ev_th AND WinPotential > pot_th
            candidates = df_test[
                (df_test['ev_win'] >= ev_th) & 
                (df_test['win_potential'] >= pot_th)
            ]
            
            count = len(candidates)
            if count < 100: continue # Ignore small sample
            
            hits = candidates[candidates['rank']==1]
            ret = hits['odds'].sum()
            cost = count
            roi = (ret / cost) * 100
            
            results.append({
                'ev_th': ev_th,
                'pot_th': pot_th,
                'count': count,
                'roi': roi
            })
            
            if roi > best_roi:
                best_roi = roi
                best_params = {'ev_th': ev_th, 'pot_th': pot_th, 'count': count, 'roi': roi}

    df_results = pd.DataFrame(results)
    print("\nXXX Ensemble Strategy Results (Top 10) XXX")
    print(df_results.sort_values('roi', ascending=False).head(10))
    print(f"\nBest Params: {best_params}")
    
    df_results.to_csv("reports/strategy_ensemble_results.csv", index=False)
    
    # Save raw data with scores for debug
    df_test[['date', 'race_id', 'horse_name', 'rank', 'odds', 'p_top3', 'p_win', 'win_potential', 'ev_win']].to_csv("reports/ensemble_predictions_2025.csv", index=False)
    logger.info("Saved metrics and logs.")

if __name__ == "__main__":
    main()

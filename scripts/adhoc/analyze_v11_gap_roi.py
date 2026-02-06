import pandas as pd
import numpy as np
import os
import sys
import yaml
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

# Logger Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_v11_models():
    lgbm_path = "models/experiments/v11_lgbm_enhanced/model.pkl"
    cat_path = "models/experiments/v11_cat_enhanced/model.cbm"
    
    logger.info(f"Loading LGBM model from {lgbm_path}...")
    with open(lgbm_path, 'rb') as f:
        lgbm_model = pickle.load(f)
    
    logger.info(f"Loading CatBoost model from {cat_path} (This may take a while as it's ~1.2GB)...")
    from catboost import CatBoostRanker
    cat_model = CatBoostRanker()
    cat_model.load_model(cat_path)
        
    return lgbm_model, cat_model

def get_predictions(config_path, valid_year=2025):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    dataset_cfg = config.get('dataset', {})
    
    loader = JraVanDataLoader()
    # 2025 data for testing
    df_raw = loader.load(history_start_date="2024-01-01", end_date="2025-12-31", jra_only=True)
    
    cleanser = DataCleanser()
    df_clean = cleanser.cleanse(df_raw)
    
    # Only 2025
    df_2025 = df_clean[pd.to_datetime(df_clean['date']).dt.year == valid_year].copy()
    
    pipeline = FeaturePipeline(cache_dir="data/features")
    feature_blocks = config.get('features', [])
    df_features = pipeline.load_features(df_2025, feature_blocks)
    
    # Load models
    lgbm_model, cat_model = load_v11_models()

    # Ensure year and other key-based features are present
    if 'year' not in df_features.columns:
        df_features = pd.merge(df_features, df_2025[['race_id', 'horse_number', 'date']], on=['race_id', 'horse_number'], how='left')
        df_features['year'] = pd.to_datetime(df_features['date']).dt.year

    # Strictly follow the feature order of the model
    feature_names = lgbm_model.feature_name()
    X = df_features[feature_names].copy()
    
    # Categorical features for LGBM (Sync with run_experiment.py logic)
    lgbm_cat_cols = [
        "jockey_id", "trainer_id", "sire_id", "horse_id", "sex", 
        "grade_code", "kyoso_joken_code", "surface", "venue", 
        "prev_grade", "dist_change_category", "interval_category"
    ]
    auto_cat = [c for c in X.columns if X[c].dtype == 'object']
    lgbm_cat_cols = list(set(lgbm_cat_cols + auto_cat))
    
    # Categorical features for CatBoost
    cat_cat_cols = list(set(lgbm_cat_cols + ["age"])) # age was noted in CatBoost config
    
    # LGBM predict
    X_lgbm = X.copy()
    for col in lgbm_cat_cols:
        if col in X_lgbm.columns:
            X_lgbm[col] = X_lgbm[col].astype('category')
    lgbm_preds = lgbm_model.predict(X_lgbm)
    
    # CatBoost predict
    X_cat = X.copy()
    for col in cat_cat_cols:
        if col in X_cat.columns:
            X_cat[col] = X_cat[col].astype(str).replace('nan', '')
    cat_preds = cat_model.predict(X_cat)
    
    # Ensemble (Rank Average or simple mean)
    # Since they are different scales, let's use Rank Average
    res_df = df_2025[['race_id', 'horse_number', 'rank', 'odds']].copy()
    res_df['lgbm_score'] = lgbm_preds
    res_df['cat_score'] = cat_preds
    
    # Rank within race
    res_df['lgbm_rank'] = res_df.groupby('race_id')['lgbm_score'].rank(ascending=False)
    res_df['cat_rank'] = res_df.groupby('race_id')['cat_score'].rank(ascending=False)
    
    res_df['ensemble_rank_score'] = (res_df['lgbm_rank'] + res_df['cat_rank']) / 2
    res_df['model_rank'] = res_df.groupby('race_id')['ensemble_rank_score'].rank(ascending=True, method='min')
    
    # Market Rank
    res_df['market_rank'] = res_df.groupby('race_id')['odds'].rank(ascending=True, method='min')
    
    return res_df

def analyze_patterns(df):
    results = []
    
    # Pattern A: Simple Odds Hole
    logger.info("Analyzing Pattern A: Simple Odds Hole")
    for x in [5.0, 10.0, 15.0, 20.0, 30.0]:
        mask = (df['model_rank'] <= 3) & (df['odds'] >= x)
        subset = df[mask]
        
        if len(subset) > 0:
            roi = (subset[subset['rank'] == 1]['odds'].sum() / len(subset)) * 100
            hit_rate = (len(subset[subset['rank'] == 1]) / len(subset)) * 100
            results.append({
                'Pattern': 'A',
                'Threshold': f'Odds >= {x}',
                'ROI': roi,
                'HitRate': hit_rate,
                'BetCount': len(subset)
            })

    # Pattern B: Gap
    logger.info("Analyzing Pattern B: Gap Strategy")
    for gap in [3, 5, 7, 10]:
        mask = (df['model_rank'] <= 3) & (df['market_rank'] >= df['model_rank'] + gap)
        subset = df[mask]
        
        if len(subset) > 0:
            roi = (subset[subset['rank'] == 1]['odds'].sum() / len(subset)) * 100
            hit_rate = (len(subset[subset['rank'] == 1]) / len(subset)) * 100
            results.append({
                'Pattern': 'B',
                'Threshold': f'Gap >= {gap}',
                'ROI': roi,
                'HitRate': hit_rate,
                'BetCount': len(subset)
            })
            
    # Pattern C: Ultra Gap (Model Rank 1 only)
    logger.info("Analyzing Pattern C: Model Rank 1 + Gap")
    for gap in [3, 5, 7, 10]:
        mask = (df['model_rank'] == 1) & (df['market_rank'] >= df['model_rank'] + gap)
        subset = df[mask]
        
        if len(subset) > 0:
            roi = (subset[subset['rank'] == 1]['odds'].sum() / len(subset)) * 100
            hit_rate = (len(subset[subset['rank'] == 1]) / len(subset)) * 100
            results.append({
                'Pattern': 'C',
                'Threshold': f'Rank1 & Gap >= {gap}',
                'ROI': roi,
                'HitRate': hit_rate,
                'BetCount': len(subset)
            })
            
    return pd.DataFrame(results)

def main():
    config_path = "config/experiments/exp_v11_lgbm.yaml"
    logger.info("Getting predictions for 2025...")
    df = get_predictions(config_path)
    
    logger.info("Analyzing...")
    report_df = analyze_patterns(df)
    
    print("\nROI Analysis Report (v11 Ensemble - 2025 Test Data)")
    print("=====================================================")
    print(report_df.to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # A
    plt.subplot(1, 2, 1)
    df_a = report_df[report_df['Pattern'] == 'A']
    sns.barplot(x='Threshold', y='ROI', data=df_a, palette='viridis')
    plt.axhline(100, color='red', linestyle='--')
    plt.title('Pattern A: Odds Threshold vs ROI')
    plt.xticks(rotation=45)
    
    # B
    plt.subplot(1, 2, 2)
    df_b = report_df[report_df['Pattern'] == 'B']
    sns.barplot(x='Threshold', y='ROI', data=df_b, palette='magma')
    plt.axhline(100, color='red', linestyle='--')
    plt.title('Pattern B: Gap Threshold vs ROI')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('reports/v11_gap_analysis.png')
    logger.info("Visualization saved to reports/v11_gap_analysis.png")
    
    # Save raw results
    report_df.to_csv('reports/v11_gap_analysis_results.csv', index=False)

if __name__ == "__main__":
    main()

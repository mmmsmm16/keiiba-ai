
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import logging
from scipy.stats import spearmanr

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from nar.loader import NarDataLoader
from nar.features import NarFeatureGenerator

logging.basicConfig(level=logging.ERROR) # Suppress logs for cleaner output

def run_baseline_analysis():
    loader = NarDataLoader()
    # Load enough data for stable analysis
    raw_df = loader.load(limit=150000, region='south_kanto')
    feature_gen = NarFeatureGenerator(history_windows=[1, 3, 5])
    df = feature_gen.generate_features(raw_df)
    
    df = df.dropna(subset=['rank']).copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Simple split
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    features = [
        'distance', 'venue', 'state', 'frame_number', 'horse_number', 'weight', 'impost',
        'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate',
        'horse_run_count'
    ] + [col for col in train_df.columns if 'horse_prev' in col]

    # Preprocessing
    categorical_cols = ['venue', 'state']
    for col in features:
        if col in categorical_cols:
            train_df[col] = train_df[col].astype(str).astype('category')
            test_df[col] = test_df[col].astype(str).astype('category')
            test_df[col] = test_df[col].cat.set_categories(train_df[col].cat.categories)
        else:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    X_train = train_df[features]
    y_train = train_df['rank'].astype(float)
    X_test = test_df[features]
    y_test = test_df['rank'].astype(float)

    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # 1. Prediction & Correlation
    preds = model.predict(X_test)
    test_df['pred_rank'] = preds
    corr, p = spearmanr(preds, y_test)
    
    # 2. Top-N Analysis
    test_df['predicted_order'] = test_df.groupby('race_id')['pred_rank'].rank(method='min')
    
    eval_by_rank = test_df.groupby('predicted_order').agg({
        'rank': [lambda x: (x == 1).mean(), lambda x: (x <= 3).mean(), 'count']
    })
    eval_by_rank.columns = ['win_rate', 'place_rate', 'count']
    
    print("--- ANALYSIS RESULTS ---")
    print(f"Total Test Races: {test_df['race_id'].nunique()}")
    print(f"Spearman Correlation: {corr:.4f}")
    print("\nTop-N Performance (Top 5 ranks):")
    print(eval_by_rank.head(5).to_string())
    
    # Feature Importance (Top 10)
    importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    print("\nTop 10 Important Features:")
    print(importances.sort_values('importance', ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    run_baseline_analysis()

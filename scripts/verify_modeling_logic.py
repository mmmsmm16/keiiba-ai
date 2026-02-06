
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import logging

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from nar.loader import NarDataLoader
from nar.features import NarFeatureGenerator

logging.basicConfig(level=logging.INFO)

def run_baseline_model():
    loader = NarDataLoader()
    # Load more data to ensure 2015+ exists
    raw_df = loader.load(limit=100000, region='south_kanto')
    feature_gen = NarFeatureGenerator(history_windows=[1, 3, 5])
    df = feature_gen.generate_features(raw_df)
    
    df = df.dropna(subset=['rank']).copy()
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Total records after dropna: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    split_date = pd.to_datetime('2015-01-01')
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    print(f"Train records: {len(train_df)}")
    print(f"Test records: {len(test_df)}")

    if len(test_df) == 0:
        print("Test set is empty! Adjusting split date or loading more data...")
        split_date = df['date'].quantile(0.8)
        print(f"Using 80% split at: {split_date}")
        train_df = df[df['date'] < split_date].copy()
        test_df = df[df['date'] >= split_date].copy()
    
    features = [
        'distance', 'venue', 'state', 'frame_number', 'horse_number', 'weight', 'impost',
        'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate',
        'horse_run_count'
    ] + [col for col in train_df.columns if 'horse_prev' in col]

    # FORCE NUMERIC
    numeric_features = [f for f in features if f not in ['venue', 'state']]
    for col in numeric_features:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # FORCE CATEGORY
    category_cols = ['venue', 'state']
    for col in category_cols:
        train_df[col] = train_df[col].astype(str).astype('category')
        test_df[col] = test_df[col].astype(str).astype('category')
        test_df[col] = test_df[col].cat.set_categories(train_df[col].cat.categories)

    X_train = train_df[features]
    y_train = train_df['rank'].astype(float)
    X_test = test_df[features]
    y_test = test_df['rank'].astype(float)

    print(f"X_train types:\n{X_train.dtypes}")
    print(f"X_train objects: {X_train.select_dtypes(include=['object']).columns.tolist()}")

    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(10)
        ])
    
    print("Training successful!")

if __name__ == "__main__":
    run_baseline_model()

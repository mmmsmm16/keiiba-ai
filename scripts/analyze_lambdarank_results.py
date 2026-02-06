import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# プロジェクトのsrcディレクトリをパスに追加
src_path = os.path.abspath(os.path.join(os.getcwd(), 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from nar.loader import NarDataLoader
from nar.features import NarFeatureGenerator

def run_evaluation():
    loader = NarDataLoader()
    print("Loading data...")
    raw_df = loader.load(limit=150000, region='south_kanto')

    print("Generating features (11 LambdaRank)...")
    generator = NarFeatureGenerator(history_windows=[1, 2, 3, 4, 5])
    df = generator.generate_features(raw_df)

    df = df.dropna(subset=['rank']).copy()
    df['date'] = pd.to_datetime(df['date'])

    # Features list (same as 10)
    baseline_features = [
        'distance', 'venue', 'state', 'frame_number', 'horse_number', 'weight', 'impost',
        'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate',
        'horse_run_count'
    ] + [col for col in df.columns if 'horse_prev' in col]

    advanced_features = [
        'gender', 'age', 'days_since_prev_race', 'weight_diff',
        'horse_jockey_place_rate', 'is_consecutive_jockey',
        'distance_diff', 'horse_venue_place_rate',
        'trainer_30d_win_rate',
        'impost_diff', 'was_accident_prev1', 'weighted_si_momentum', 'weighted_rank_momentum',
        'class_rank', 'class_diff', 'is_promoted', 'is_demoted'
    ]
    features = list(set(baseline_features + advanced_features))

    # Preprocessing
    categorical_cols = ['venue', 'state', 'gender']
    for col in features:
        if col in categorical_cols:
            df[col] = df[col].astype(str).astype('category')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Split
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()

    # LambdaRank specific processing
    train_df = train_df.sort_values('race_id')
    test_df = test_df.sort_values('race_id')
    
    train_groups = train_df.groupby('race_id').size().values
    test_groups = test_df.groupby('race_id').size().values
    
    # Label: High relevance for low rank (higher is better)
    train_label = 20 - train_df['rank']
    test_label = 20 - test_df['rank']

    print(f"Training LambdaRank (LGBMRanker) on {len(train_df)} rows...")
    model = lgb.LGBMRanker(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=6,
        random_state=42,
        importance_type='gain',
        verbosity=-1
    )
    model.fit(
        train_df[features], train_label,
        group=train_groups,
        eval_set=[(test_df[features], test_label)],
        eval_group=[test_groups],
        eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )

    # Prediction (Ranker outputs scores where higher is better)
    preds = model.predict(test_df[features])
    
    # Calculate Spearman Correlation
    # Note: preds is higher-is-better, rank is lower-is-better, so we correlate with negative rank or invert
    corr, _ = spearmanr(preds, -test_df['rank'])
    
    # Hit rates
    test_df['pred_score'] = preds
    # Important: Ranker's pred_score is higher-is-better, so we rank descending for pred_rank
    test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(method='min', ascending=False)
    
    top1 = test_df[test_df['pred_rank'] == 1]
    win_rate = (top1['rank'] == 1).mean()
    place_rate = (top1['rank'] <= 3).mean()

    print(f"\n--- Evaluation Results (11 LambdaRank) ---")
    print(f"Spearman Correlation: {corr:.4f}")
    print(f"Top 1 Win Rate: {win_rate:.2%}")
    print(f"Top 1 Place Rate: {place_rate:.2%}")

    # Top importances
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 20 Features:")
    print(importances.head(20))

if __name__ == "__main__":
    run_evaluation()

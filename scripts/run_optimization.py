import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# プロジェクトのsrcディレクトリをパスに追加
src_path = os.path.abspath(os.path.join(os.getcwd(), 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from nar.loader import NarDataLoader
from nar.features import NarFeatureGenerator

def run_tuning():
    loader = NarDataLoader()
    print("Loading data...")
    raw_df = loader.load(limit=150000, region='south_kanto')

    generator = NarFeatureGenerator(history_windows=[1, 2, 3, 4, 5])
    df = generator.generate_features(raw_df)

    df = df.dropna(subset=['rank']).copy()
    df['date'] = pd.to_datetime(df['date'])

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
        'impost_diff', 'was_accident_prev1', 'weighted_si_momentum', 'weighted_rank_momentum'
    ]
    features = list(set(baseline_features + advanced_features))

    categorical_cols = ['venue', 'state', 'gender']
    for col in features:
        if col in categorical_cols:
            df[col] = df[col].astype(str).astype('category')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            train_df[features], train_df['rank'],
            eval_set=[(test_df[features], test_df['rank'])],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        preds = model.predict(test_df[features])
        rmse = np.sqrt(mean_squared_error(test_df['rank'], preds))
        return rmse

    print("Starting Optuna optimization (20 trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print("\nBest Parameters:")
    print(study.best_params)

    # Evaluate Final
    best_params = study.best_params
    best_params['objective'] = 'regression'
    best_model = lgb.LGBMRegressor(n_estimators=1000, **best_params)
    best_model.fit(
        train_df[features], train_df['rank'],
        eval_set=[(test_df[features], test_df['rank'])],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)]
    )

    test_df['pred_score'] = best_model.predict(test_df[features])
    corr, _ = spearmanr(test_df['pred_score'], test_df['rank'])
    
    test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(method='min')
    top1 = test_df[test_df['pred_rank'] == 1]
    win_rate = (top1['rank'] == 1).mean()
    place_rate = (top1['rank'] <= 3).mean()

    print(f"\n--- Final Results (Optimized) ---")
    print(f"Spearman Correlation: {corr:.4f}")
    print(f"Top 1 Win Rate: {win_rate:.2%}")
    print(f"Top 1 Place Rate: {place_rate:.2%}")

if __name__ == "__main__":
    run_tuning()

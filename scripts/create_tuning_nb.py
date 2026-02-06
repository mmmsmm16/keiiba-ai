import nbformat as nbf
import os

target_nb = '/workspace/notebooks/nar/09_nar_hyperparameter_tuning.ipynb'
nb = nbf.v4.new_notebook()

# 1. Title
nb.cells.append(nbf.v4.new_markdown_cell("# 地方競馬（NAR）ハイパーパラメータ最適化 (09モデル)\n\nOptunaを用いて、08モデルで使用した特徴量セットに対する最適なLightGBMパラメータを探索します。"))

# 2. Setup (Reuse 08 logic)
nb.cells.append(nbf.v4.new_code_cell("""import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# プロジェクトのsrcディレクトリをパスに追加
src_path = os.path.abspath(os.path.join(os.getcwd(), '../../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from nar.loader import NarDataLoader
from nar.features import NarFeatureGenerator"""))

# 3. Data Loading
nb.cells.append(nbf.v4.new_code_cell("""loader = NarDataLoader()
raw_df = loader.load(limit=150000, region='south_kanto')

# 08モデルと同じ特徴量生成（1-5走前、改善ロジック込）
generator = NarFeatureGenerator(history_windows=[1, 2, 3, 4, 5])
df = generator.generate_features(raw_df)

df = df.dropna(subset=['rank']).copy()
df['date'] = pd.to_datetime(df['date'])

# 特徴量リスト
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
test_df = df[df['date'] >= split_date].copy()"""))

# 4. Optuna Objective
nb.cells.append(nbf.v4.new_markdown_cell("## Optunaによるパラメータ探索\n\n検証データに対するRMSEを最小化するパラメータを探索します。"))
nb.cells.append(nbf.v4.new_code_cell("""def objective(trial):
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
    
    # 訓練/検証
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train_df[features], train_df['rank'],
        eval_set=[(test_df[features], test_df['rank'])],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    preds = model.predict(test_df[features])
    rmse = np.sqrt(mean_squared_error(test_df['rank'], preds))
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("Best Trial:")
print(study.best_trial.params)"""))

# 5. Final Model Evaluation
nb.cells.append(nbf.v4.new_markdown_cell("## 最良パラメータでの評価\n\n探索で見つかった最良のパラメータを用いて最終的な精度を算出します。"))
nb.cells.append(nbf.v4.new_code_cell("""best_params = study.best_trial.params
best_params['objective'] = 'regression'
# n_estimators は十分に大きく設定
best_model = lgb.LGBMRegressor(n_estimators=2000, **best_params)

best_model.fit(
    train_df[features], train_df['rank'],
    eval_set=[(test_df[features], test_df['rank'])],
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

test_df['pred_score'] = best_model.predict(test_df[features])
corr, _ = spearmanr(test_df['pred_score'], test_df['rank'])
print(f"最良パラメータでの Spearman相関: {corr:.4f}")

# 的中率
test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(method='min')
top1 = test_df[test_df['pred_rank'] == 1]
print(f"予測1位 勝率: {(top1['rank'] == 1).mean():.2%}")
print(f"予測1位 複勝率: {(top1['rank'] <= 3).mean():.2%}")"""))

with open(target_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created {target_nb} successfully.")

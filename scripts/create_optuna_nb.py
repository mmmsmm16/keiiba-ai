
import nbformat as nbf
import os

def create_optuna_notebook():
    nb = nbf.v4.new_notebook()
    
    # Markdown - Title
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# 地方競馬（NAR）最新モデルのハイパーパラメータ最適化 (13モデル)\n\n"
        "フェーズ 9 で導入した高度な特徴量に対し、Optuna を用いて LightGBM (LGBMRanker) のパラメータを最適化します。\n\n"
        "### 最適化対象\n"
        "- **目的関数**: `lambdarank` (NDCG)\n"
        "- **評価指標**: NDCG@3 (上位3着の順位付け精度)"
    ))
    
    # Code - Setup
    nb.cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import lightgbm as lgb\n"
        "import optuna\n"
        "from scipy.stats import spearmanr\n"
        "import logging\n\n"
        "# プロジェクトのsrcディレクトリをパスに追加\n"
        "src_path = os.path.abspath(os.path.join(os.getcwd(), '../../src'))\n"
        "if src_path not in sys.path:\n"
        "    sys.path.append(src_path)\n\n"
        "from nar.loader import NarDataLoader\n"
        "from nar.features import NarFeatureGenerator\n\n"
        "optuna.logging.set_verbosity(optuna.logging.WARNING)\n"
        "logging.getLogger('lightgbm').setLevel(logging.ERROR)"
    ))
    
    # Code - Data Prepare
    nb.cells.append(nbf.v4.new_code_cell(
        "loader = NarDataLoader()\n"
        "raw_df = loader.load(limit=150000, region='south_kanto')\n\n"
        "generator = NarFeatureGenerator(history_windows=[1, 2, 3, 4, 5])\n"
        "df = generator.generate_features(raw_df)\n\n"
        "df = df.dropna(subset=['rank']).copy()\n"
        "df['date'] = pd.to_datetime(df['date'])\n\n"
        "baseline_features = [\n"
        "    'distance', 'venue', 'state', 'frame_number', 'horse_number', 'weight', 'impost',\n"
        "    'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate',\n"
        "    'horse_run_count'\n"
        "] + [col for col in df.columns if 'horse_prev' in col]\n\n"
        "advanced_features = [\n"
        "    'gender', 'age', 'days_since_prev_race', 'weight_diff',\n"
        "    'horse_jockey_place_rate', 'is_consecutive_jockey',\n"
        "    'distance_diff', 'horse_venue_place_rate',\n"
        "    'trainer_30d_win_rate',\n"
        "    'impost_diff', 'was_accident_prev1', 'weighted_si_momentum', 'weighted_rank_momentum',\n"
        "    'class_rank', 'class_diff', 'is_promoted', 'is_demoted'\n"
        "]\n\n"
        "phase9_features = [\n"
        "    'weighted_si_momentum_race_rank', 'weighted_si_momentum_diff_from_avg', 'weighted_si_momentum_zscore',\n"
        "    'weighted_rank_momentum_race_rank', 'weighted_rank_momentum_diff_from_avg', 'weighted_rank_momentum_zscore',\n"
        "    'class_rank_race_rank', 'class_rank_diff_from_avg', 'class_rank_zscore',\n"
        "    'horse_state_place_rate', 'season', 'is_night_race', 'trainer_momentum_bias'\n"
        "]\n\n"
        "features = list(set(baseline_features + advanced_features + phase9_features))\n\n"
        "categorical_cols = ['venue', 'state', 'gender', 'season']\n"
        "for col in features:\n"
        "    if col in df.columns:\n"
        "        if col in categorical_cols:\n"
        "            df[col] = df[col].astype(str).astype('category')\n"
        "        else:\n"
        "            df[col] = pd.to_numeric(df[col], errors='coerce')\n\n"
        "features = [f for f in features if f in df.columns]\n\n"
        "split_date = df['date'].quantile(0.8)\n"
        "train_df = df[df['date'] < split_date].sort_values('race_id').copy()\n"
        "test_df = df[df['date'] >= split_date].sort_values('race_id').copy()\n\n"
        "train_groups = train_df.groupby('race_id').size().values\n"
        "test_groups = test_df.groupby('race_id').size().values\n\n"
        "train_label = 20 - train_df['rank']\n"
        "test_label = 20 - test_df['rank']\n\n"
        "print(f'訓練データ: {len(train_df)}')\n"
        "print(f'テストデータ: {len(test_df)}')"
    ))
    
    # Code - Objective Function
    nb.cells.append(nbf.v4.new_code_cell(
        "def objective(trial):\n"
        "    params = {\n"
        "        'objective': 'lambdarank',\n"
        "        'metric': 'ndcg',\n"
        "        'ndcg_at': [3],\n"
        "        'verbosity': -1,\n"
        "        'boosting_type': 'gbdt',\n"
        "        'random_state': 42,\n"
        "        'num_leaves': trial.suggest_int('num_leaves', 32, 128),\n"
        "        'max_depth': trial.suggest_int('max_depth', 4, 10),\n"
        "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),\n"
        "        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),\n"
        "        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),\n"
        "        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),\n"
        "        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),\n"
        "    }\n"
        "    \n"
        "    model = lgb.LGBMRanker(**params, n_estimators=500)\n"
        "    \n"
        "    model.fit(\n"
        "        train_df[features], train_label,\n"
        "        group=train_groups,\n"
        "        eval_set=[(test_df[features], test_label)],\n"
        "        eval_group=[test_groups],\n"
        "        eval_at=[3],\n"
        "        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]\n"
        "    )\n"
        "    \n"
        "    return model.best_score_['valid_0']['ndcg@3']\n\n"
        "study = optuna.create_study(direction='maximize')\n"
        "study.optimize(objective, n_trials=50, show_progress_bar=True)\n\n"
        "print('Best trial:')\n"
        "trial = study.best_trial\n"
        "print(f'  Value: {trial.value}')\n"
        "print('  Params: ')\n"
        "for key, value in trial.params.items():\n"
        "    print(f'    {key}: {value}')"
    ))
    
    # Code - Evaluation with Best Params
    nb.cells.append(nbf.v4.new_code_cell(
        "best_params = study.best_params\n"
        "model = lgb.LGBMRanker(**best_params, n_estimators=1000, random_state=42)\n\n"
        "model.fit(\n"
        "    train_df[features], train_label,\n"
        "    group=train_groups,\n"
        "    eval_set=[(test_df[features], test_label)],\n"
        "    eval_group=[test_groups],\n"
        "    eval_at=[1, 3, 5],\n"
        "    callbacks=[lgb.early_stopping(stopping_rounds=100)]\n"
        ")\n\n"
        "test_df['pred_score'] = model.predict(test_df[features])\n"
        "test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(method='min', ascending=False)\n\n"
        "eval_list = []\n"
        "for r in range(1, 6):\n"
        "    matches = test_df[test_df['pred_rank'] == r]\n"
        "    win_rate = (matches['rank'] == 1).mean()\n"
        "    place_rate = (matches['rank'] <= 3).mean()\n"
        "    eval_list.append({'predicted_rank': r, 'win_rate': win_rate, 'place_rate': place_rate})\n\n"
        "eval_df = pd.DataFrame(eval_list)\n"
        "print('\\n予測順位別 的中率:')\n"
        "print(eval_df)"
    ))
    
    # Save
    path = '/workspace/notebooks/nar/13_nar_optuna_relative_model.ipynb'
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook created at {path}')

if __name__ == '__main__':
    create_optuna_notebook()

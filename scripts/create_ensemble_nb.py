
import nbformat as nbf
import os

def create_ensemble_notebook():
    nb = nbf.v4.new_notebook()
    
    # Markdown - Title
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# 地方競馬（NAR）アンサンブル戦略 (15モデル)\n\n"
        "これまでに構築した強力な2つのモデルを組み合わせることで、さらなる予測精度の向上（特に複勝率と安定性）を目指します。\n\n"
        "### 統合するモデル\n"
        "1. **回帰モデル (Regression)**: 全体的な着順傾向を学習 (Spearman相関が高い)。`10_nar_class_features_model` 相当。\n"
        "2. **ランク学習モデル (LambdaRank)**: 1位の特定に特化 (Top-1勝率が高い)。`13_nar_optuna_relative_model` 相当。\n\n"
        "### 戦略\n"
        "- **Weighted Blending**: 両モデルの予測スコア（またはランク）を加重平均します。"
    ))
    
    # Code - Setup
    nb.cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import lightgbm as lgb\n"
        "from scipy.stats import spearmanr\n"
        "import matplotlib.pyplot as plt\n"
        "import japanize_matplotlib\n\n"
        "# プロジェクトのsrcディレクトリをパスに追加\n"
        "src_path = os.path.abspath(os.path.join(os.getcwd(), '../../src'))\n"
        "if src_path not in sys.path:\n"
        "    sys.path.append(src_path)\n\n"
        "from nar.loader import NarDataLoader\n"
        "from nar.features import NarFeatureGenerator\n\n"
        "%matplotlib inline"
    ))
    
    # Code - Data Load
    nb.cells.append(nbf.v4.new_code_cell(
        "loader = NarDataLoader()\n"
        "# データ量は多めに確保\n"
        "raw_df = loader.load(limit=150000, region='south_kanto')\n\n"
        "generator = NarFeatureGenerator(history_windows=[1, 2, 3, 4, 5])\n"
        "df = generator.generate_features(raw_df)\n\n"
        "df = df.dropna(subset=['rank']).copy()\n"
        "df['date'] = pd.to_datetime(df['date'])\n\n"
        "# 特徴量定義 (Phase 9 base)\n"
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
        "# カテゴリ処理\n"
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
        "print(f'Train: {len(train_df)}, Test: {len(test_df)}')"
    ))
    
    # Code - Train Models (Regression & LambdaRank)
    nb.cells.append(nbf.v4.new_code_cell(
        "# Model 1: Regression (LGBMRegressor)\n"
        "print('Training Regression Model...')\n"
        "reg_model = lgb.LGBMRegressor(\n"
        "    n_estimators=1000,\n"
        "    learning_rate=0.05,\n"
        "    num_leaves=64,\n"
        "    max_depth=6,\n"
        "    random_state=42\n"
        ")\n"
        "reg_model.fit(\n"
        "    train_df[features], train_df['rank'],\n"
        "    eval_set=[(test_df[features], test_df['rank'])],\n"
        "    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]\n"
        ")\n\n"
        "# Model 2: LambdaRank (LGBMRanker)\n"
        "print('Training LambdaRank Model...')\n"
        "train_groups = train_df.groupby('race_id').size().values\n"
        "test_groups = test_df.groupby('race_id').size().values\n"
        "train_label = 20 - train_df['rank']\n"
        "test_label = 20 - test_df['rank']\n\n"
        "rank_model = lgb.LGBMRanker(\n"
        "    n_estimators=1000,\n"
        "    learning_rate=0.05,\n"
        "    num_leaves=64,\n"
        "    max_depth=6,\n"
        "    random_state=42,\n"
        "    metric='ndcg',\n"
        "    importance_type='gain'\n"
        ")\n"
        "rank_model.fit(\n"
        "    train_df[features], train_label,\n"
        "    group=train_groups,\n"
        "    eval_set=[(test_df[features], test_label)],\n"
        "    eval_group=[test_groups],\n"
        "    eval_at=[3],\n"
        "    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]\n"
        ")"
    ))
    
    # Code - Ensemble Logic
    nb.cells.append(nbf.v4.new_code_cell(
        "# 予測の実行\n"
        "test_df['pred_reg'] = reg_model.predict(test_df[features])\n"
        "test_df['pred_rank'] = rank_model.predict(test_df[features])\n\n"
        "# スケーリング (0-1に正規化して合わせる)\n"
        "# Regressionは「順位」なので小さい方が良い -> 反転させる\n"
        "# LambdaRankは「スコア」なので大きい方が良い\n"
        "from sklearn.preprocessing import MinMaxScaler\n"
        "scaler = MinMaxScaler()\n\n"
        "# Reg: 1位=1.0, 最下位=0.0 になるように反転変換\n"
        "test_df['score_reg_inv'] = -test_df['pred_reg']\n"
        "test_df['norm_reg'] = scaler.fit_transform(test_df[['score_reg_inv']])\n\n"
        "# Rank: そのまま正規化\n"
        "test_df['norm_rank'] = scaler.fit_transform(test_df[['pred_rank']])\n\n"
        "# アンサンブル (加重平均)\n"
        "# Rankerの方が勝率が高いので比重を重くする (例: 0.3 : 0.7)\n"
        "alpha = 0.3\n"
        "test_df['ensemble_score'] = alpha * test_df['norm_reg'] + (1 - alpha) * test_df['norm_rank']\n\n"
        "# レースごとの順位付け\n"
        "test_df['final_rank'] = test_df.groupby('race_id')['ensemble_score'].rank(ascending=False, method='min')"
    ))
    
    # Code - Evaluation
    nb.cells.append(nbf.v4.new_code_cell(
        "eval_list = []\n"
        "for r in range(1, 6):\n"
        "    matches = test_df[test_df['final_rank'] == r]\n"
        "    win_rate = (matches['rank'] == 1).mean()\n"
        "    place_rate = (matches['rank'] <= 3).mean()\n"
        "    eval_list.append({'predicted_rank': r, 'win_rate': win_rate, 'place_rate': place_rate})\n\n"
        "eval_df = pd.DataFrame(eval_list)\n"
        "print('アンサンブルモデル予測精度:')\n"
        "print(eval_df)\n\n"
        "corr, _ = spearmanr(test_df['ensemble_score'], 20-test_df['rank'])\n"
        "print(f'Spearman相関係数 (Ensemble): {corr:.4f}')"
    ))

    # Save
    path = '/workspace/notebooks/nar/15_nar_ensemble_strategy.ipynb'
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook created at {path}')

if __name__ == '__main__':
    create_ensemble_notebook()


import nbformat as nbf
import os

def create_pedigree_notebook():
    nb = nbf.v4.new_notebook()
    
    # Markdown - Title
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# 地方競馬（NAR）血統導入モデル (17モデル)\n\n"
        "父馬 (sire_id) の過去成績を集計し、血統傾向（Target Encoding）を特徴量として導入します。\n\n"
        "### 仮説\n"
        "- **仮説**: 特にクラス変動時や初コース時、または若駒において、自身の過去成績よりも血統的なポテンシャルが結果に影響する。\n"
        "- **実装**: 時系列に沿った expanding mean で `sire_win_rate`, `sire_place_rate` を計算しモデルに投入。"
    ))
    
    # Code - Setup
    nb.cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import lightgbm as lgb\n"
        "import seaborn as sns\n"
        "from scipy.stats import spearmanr\n"
        "import matplotlib.pyplot as plt\n"
        "import japanize_matplotlib\n\n"
        "# プロジェクトのsrcディレクトリをパスに追加\n"
        "src_path = os.path.abspath(os.path.join(os.getcwd(), '../../src'))\n"
        "if src_path not in sys.path:\n"
        "    sys.path.append(src_path)\n\n"
        "from nar.loader import NarDataLoader\n"
        "from nar.features import NarFeatureGenerator\n\n"
        "%matplotlib inline\n"
        "sns.set(font='IPAexGothic', style='whitegrid')"
    ))
    
    # Code - Data Load
    nb.cells.append(nbf.v4.new_code_cell(
        "loader = NarDataLoader()\n"
        "# データ量は多めに確保\n"
        "raw_df = loader.load(limit=200000, region='south_kanto')\n\n"
        "generator = NarFeatureGenerator(history_windows=[1, 2, 3, 4, 5])\n"
        "df = generator.generate_features(raw_df)\n\n"
        "df = df.dropna(subset=['rank']).copy()\n"
        "df['date'] = pd.to_datetime(df['date'])\n\n"
        "# 特徴量定義\n"
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
        "pedigree_features = [\n"
        "    'sire_win_rate', 'sire_place_rate'\n"
        "]\n\n"
        "features = list(set(baseline_features + advanced_features + phase9_features + pedigree_features))\n\n"
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
    
    # Code - Train (Optuna Params based)
    nb.cells.append(nbf.v4.new_code_cell(
        "# NB13の最適パラメータを使用\n"
        "params = {\n"
        "    'objective': 'lambdarank',\n"
        "    'metric': 'ndcg',\n"
        "    'ndcg_at': [1, 3, 5],\n"
        "    'n_estimators': 1000,\n"
        "    'learning_rate': 0.05, # NB13 tuned value approx\n"
        "    'num_leaves': 64,\n"
        "    'max_depth': 6,\n"
        "    'random_state': 42,\n"
        "    'importance_type': 'gain'\n"
        "}\n\n"
        "train_groups = train_df.groupby('race_id').size().values\n"
        "test_groups = test_df.groupby('race_id').size().values\n"
        "train_label = 20 - train_df['rank']\n"
        "test_label = 20 - test_df['rank']\n\n"
        "model = lgb.LGBMRanker(**params)\n"
        "model.fit(\n"
        "    train_df[features], train_label,\n"
        "    group=train_groups,\n"
        "    eval_set=[(test_df[features], test_label)],\n"
        "    eval_group=[test_groups],\n"
        "    eval_at=[1, 3, 5],\n"
        "    callbacks=[lgb.early_stopping(stopping_rounds=50)]\n"
        ")"
    ))
    
    # Code - Evaluation
    nb.cells.append(nbf.v4.new_code_cell(
        "test_df['pred_score'] = model.predict(test_df[features])\n"
        "test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(method='min', ascending=False)\n\n"
        "eval_list = []\n"
        "for r in range(1, 6):\n"
        "    matches = test_df[test_df['pred_rank'] == r]\n"
        "    win_rate = (matches['rank'] == 1).mean()\n"
        "    place_rate = (matches['rank'] <= 3).mean()\n"
        "    eval_list.append({'predicted_rank': r, 'win_rate': win_rate, 'place_rate': place_rate})\n\n"
        "eval_df = pd.DataFrame(eval_list)\n"
        "print('血統導入モデル予測精度:')\n"
        "print(eval_df)\n\n"
        "corr, _ = spearmanr(test_df['pred_score'], 20-test_df['rank'])\n"
        "print(f'Spearman相関係数: {corr:.4f}')\n"
        "\n"
        "# 重要度表示\n"
        "plt.figure(figsize=(10, 10))\n"
        "lgb.plot_importance(model, max_num_features=30, importance_type='gain')\n"
        "plt.title('Feature Importance (Gain)')\n"
        "plt.show()"
    ))

    # Save
    path = '/workspace/notebooks/nar/17_nar_pedigree_model.ipynb'
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook created at {path}')

if __name__ == '__main__':
    create_pedigree_notebook()

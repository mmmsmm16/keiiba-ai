
import nbformat as nbf
import os

def create_calibration_notebook():
    nb = nbf.v4.new_notebook()
    
    # Markdown - Title
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# 予測確率の較正 (Calibration) 分析 (19モデル)\n\n"
        "現在の「期待値フィルタ (EV Filter)」の精度をさらに高めるため、モデルが出力する予測確率 (Softmax Prob) が、現実の勝率とどれくらい一致しているか (Calibration) を分析します。\n\n"
        "### 目的\n"
        "- **カバレッジの向上**: 過小評価されている高期待値ゾーンを見つけ、購入レース数を増やせるか検証する。\n"
        "- **複勝への応用**: 3着内率 (Place Prob) の精度も確認する。\n"
    ))
    
    # Code - Setup
    nb.cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import lightgbm as lgb\n"
        "import seaborn as sns\n"
        "import matplotlib.pyplot as plt\n"
        "import japanize_matplotlib\n"
        "from scipy.special import softmax\n"
        "from sklearn.calibration import calibration_curve\n\n"
        "# プロジェクトのsrcディレクトリをパスに追加\n"
        "src_path = os.path.abspath(os.path.join(os.getcwd(), '../../src'))\n"
        "if src_path not in sys.path:\n"
        "    sys.path.append(src_path)\n\n"
        "from nar.loader import NarDataLoader\n"
        "from nar.features import NarFeatureGenerator\n\n"
        "%matplotlib inline\n"
        "sns.set(font='IPAexGothic', style='whitegrid')"
    ))
    
    # Code - Data Load & Model Train (NB17 Copy)
    nb.cells.append(nbf.v4.new_code_cell(
        "# --- モデル構築 (NB17相当) ---\n"
        "loader = NarDataLoader()\n"
        "raw_df = loader.load(limit=200000, region='south_kanto')\n\n"
        "generator = NarFeatureGenerator(history_windows=[1, 2, 3, 4, 5])\n"
        "df = generator.generate_features(raw_df)\n\n"
        "df = df.dropna(subset=['rank']).copy()\n"
        "df['date'] = pd.to_datetime(df['date'])\n\n"
        "# 特徴量定義 (NB17)\n"
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
        "track_bias_features = [\n"
        "    'track_bias_inner_win_rate', 'track_bias_outer_win_rate', 'track_bias_front_win_rate'\n"
        "]\n\n"
        "features = list(set(baseline_features + advanced_features + phase9_features + track_bias_features + pedigree_features))\n"
        "features = [f for f in features if f in df.columns]\n\n"
        "# 型変換\n"
        "categorical_cols = ['venue', 'state', 'gender', 'season']\n"
        "for col in features:\n"
        "    if col in categorical_cols:\n"
        "        df[col] = df[col].astype(str).astype('category')\n"
        "    else:\n"
        "        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)\n\n"
        "split_date = df['date'].quantile(0.8)\n"
        "train_df = df[df['date'] < split_date].sort_values('race_id').copy()\n"
        "test_df = df[df['date'] >= split_date].sort_values('race_id').copy()\n\n"
        "train_groups = train_df.groupby('race_id').size().values\n"
        "test_groups = test_df.groupby('race_id').size().values\n"
        "train_label = 20 - train_df['rank']\n"
        "test_label = 20 - test_df['rank']\n\n"
        "# NB17 Params\n"
        "params = {\n"
        "    'objective': 'lambdarank',\n"
        "    'metric': 'ndcg',\n"
        "    'ndcg_at': [1, 3, 5],\n"
        "    'n_estimators': 1000,\n"
        "    'learning_rate': 0.05,\n"
        "    'num_leaves': 64,\n"
        "    'max_depth': 6,\n"
        "    'random_state': 42,\n"
        "    'importance_type': 'gain'\n"
        "}\n\n"
        "model = lgb.LGBMRanker(**params)\n"
        "model.fit(\n"
        "    train_df[features], train_label,\n"
        "    group=train_groups,\n"
        "    eval_set=[(test_df[features], test_label)],\n"
        "    eval_group=[test_groups],\n"
        "    callbacks=[lgb.early_stopping(stopping_rounds=50)]\n"
        ")"
    ))
    
    # Code - Prob Calc & Calibration Plot
    nb.cells.append(nbf.v4.new_code_cell(
        "# Softmax確率計算\n"
        "test_df['pred_score'] = model.predict(test_df[features])\n"
        "def calc_prob(df_group):\n"
        "    df_group['pred_prob'] = softmax(df_group['pred_score'])\n"
        "    return df_group\n"
        "test_df = test_df.groupby('race_id', group_keys=False).apply(calc_prob)\n\n"
        "# 1. Win Calibration (1着率)\n"
        "prob_true, prob_pred = calibration_curve(test_df['rank'] == 1, test_df['pred_prob'], n_bins=10)\n\n"
        "plt.figure(figsize=(8, 8))\n"
        "plt.plot(prob_pred, prob_true, marker='o', label='Model Softmax')\n"
        "plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')\n"
        "plt.xlabel('Mean Predicted Probability')\n"
        "plt.ylabel('Fraction of Positives (Actual Win Rate)')\n"
        "plt.title('Calibration Curve (Win)')\n"
        "plt.legend()\n"
        "plt.show()"
    ))
    
    # Code - Place Simulation (Simple EV check)
    nb.cells.append(nbf.v4.new_code_cell(
        "# 2. 複勝 (Place) への応用\n"
        "# 現状のモデルは「1着順位」を学習しているが、複勝圏内率もスコアに相関するはず。\n"
        "# 複勝オッズもロードして、'Place EV' を計算してみる\n"
        "\n"
        "# 払戻金ロード (再利用できればするが、ここでは簡易的に一部ロード)\n"
        "# 時間がかかるので、今回の検証では「予測スコア分布」と「3着内率」の関係だけ見る\n"
        "\n"
        "# スコアをビン分割\n"
        "test_df['score_bin'] = pd.qcut(test_df['pred_score'], 10, labels=False)\n"
        "place_rates = test_df.groupby('score_bin')['rank'].apply(lambda x: (x <= 3).mean())\n"
        "\n"
        "plt.figure(figsize=(10, 5))\n"
        "plt.plot(place_rates.index, place_rates.values, marker='o')\n"
        "plt.title('Score Percentile vs Place Rate')\n"
        "plt.xlabel('Score Decile (0=Low, 9=High)')\n"
        "plt.ylabel('Actual Place Rate (Rank <= 3)')\n"
        "plt.grid(True)\n"
        "plt.show()"
    ))

    # Save
    path = '/workspace/notebooks/nar/19_nar_calibration_analysis.ipynb'
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook created at {path}')

if __name__ == '__main__':
    create_calibration_notebook()

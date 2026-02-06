
import nbformat as nbf
import os

def create_viz_notebook():
    nb = nbf.v4.new_notebook()
    
    # Markdown - Title
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# 地方競馬（NAR）新規特徴量の可視化と性能分析 (14モデル)\n\n"
        "フェーズ 9 で導入した高度な特徴量（相対指標、状況適性）の中身を詳しく分析し、モデルの動作原理とデータの分布を確認します。"
    ))
    
    # Code - Setup
    nb.cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import seaborn as sns\n"
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
        "raw_df = loader.load(limit=100000, region='south_kanto')\n\n"
        "generator = NarFeatureGenerator(history_windows=[1, 2, 3, 4, 5])\n"
        "df = generator.generate_features(raw_df)\n\n"
        "df = df.dropna(subset=['rank']).copy()\n"
        "df['date'] = pd.to_datetime(df['date'])\n\n"
        "print(f'データ件数: {len(df)}')"
    ))
    
    # Markdown - Relative Metrics Viz
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 1. レース内相対指標の分布分析\n\n"
        "スピード指数偏差値 (`weighted_si_momentum_zscore`) や相対順位が、実際の着順とどのように関連しているかを確認します。"
    ))
    
    # Code - Relative Metrics Plot
    nb.cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(12, 6))\n"
        "sns.boxplot(x='rank', y='weighted_si_momentum_zscore', data=df[df['rank'] <= 10])\n"
        "plt.title('着順別のスピード指数偏差値 (z-score) 分布')\n"
        "plt.xlabel('確定着順')\n"
        "plt.ylabel('SI偏差値 (z-score)')\n"
        "plt.show()\n\n"
        "print('考察: 着順が良い（小さい）ほど、z-scoreが高い（右肩下がり）傾向があれば、相対指標が強力な予測因子であることを示します。')"
    ))
    
    # Markdown - Situational Aptitude Viz
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 2. 状況適性の分析\n\n"
        "馬場状態別成績 (`horse_state_place_rate`) や季節・ナイターが的中率に与える影響を確認します。"
    ))
    
    # Code - Situational Plot
    nb.cells.append(nbf.v4.new_code_cell(
        "df['state_place_rate_bin'] = pd.cut(df['horse_state_place_rate'], bins=5)\n"
        "plt.figure(figsize=(10, 6))\n"
        "df.groupby('state_place_rate_bin', observed=True)['rank'].apply(lambda x: (x==1).mean()).plot(kind='bar', color='skyblue')\n"
        "plt.title('馬場状態別実績スコアと実際の勝率の関係')\n"
        "plt.xlabel('馬場別実績スコア (bin)')\n"
        "plt.ylabel('勝率')\n"
        "plt.xticks(rotation=45)\n"
        "plt.show()\n\n"
        "plt.figure(figsize=(10, 6))\n"
        "sns.countplot(x='season', hue='is_night_race', data=df)\n"
        "plt.title('季節・ナイター別のレース数分布')\n"
        "plt.show()"
    ))
    
    # Markdown - Human Bias Viz
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 3. 人間系の勢いバイアス分析\n\n"
        "`trainer_momentum_bias` (直近30日の上振れ) が結果にどう影響するかを確認します。"
    ))
    
    # Code - Human Bias Plot
    nb.cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(12, 6))\n"
        "sns.violinplot(x='rank', y='trainer_momentum_bias', data=df[df['rank'] <= 5])\n"
        "plt.title('上位着順別の調教師勢いバイアス分布')\n"
        "plt.show()"
    ))
    
    # Save
    path = '/workspace/notebooks/nar/14_nar_feature_visualization.ipynb'
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook created at {path}')

if __name__ == '__main__':
    create_viz_notebook()

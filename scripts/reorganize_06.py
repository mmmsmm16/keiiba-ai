import nbformat as nbf
import os

target_nb = '/workspace/notebooks/nar/06_nar_advanced_model.ipynb'
nb = nbf.v4.new_notebook()

# 1. Title
nb.cells.append(nbf.v4.new_markdown_cell("# 地方競馬（NAR）高度特徴量モデル構築\n\nベースラインモデルに高度な特徴量を追加し、予測精度を向上させたモデルを構築します。"))

# 2. Setup
nb.cells.append(nbf.v4.new_code_cell("""import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# プロジェクトのsrcディレクトリをパスに追加
src_path = os.path.abspath(os.path.join(os.getcwd(), '../../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from nar.loader import NarDataLoader
from nar.features import NarFeatureGenerator

%matplotlib inline
sns.set(font="IPAexGothic", style="whitegrid")"""))

# 3. Data Loading
nb.cells.append(nbf.v4.new_markdown_cell("## 1. データのロードと特徴量生成\n\n高度な特徴量（休養期間、馬体重増減、人馬相性など）を含めて生成します。"))
nb.cells.append(nbf.v4.new_code_cell("""loader = NarDataLoader()
# データのロード（学習用に十分な量を取得）
raw_df = loader.load(limit=150000, region='south_kanto')

# 特徴量生成（高度特徴量機能を含む）
generator = NarFeatureGenerator(history_windows=[1, 3, 5])
df = generator.generate_features(raw_df)

# ラベル（着順）が存在するデータのみに絞り込み
df = df.dropna(subset=['rank']).copy()
df['date'] = pd.to_datetime(df['date'])

print(f"データ件数: {len(df)}")"""))

# 4. Data Preprocessing
nb.cells.append(nbf.v4.new_markdown_cell("## 2. 特徴量の整理と時系列分割\n\n使用する特徴量を定義し、カテゴリ変数の処理を行います。"))
nb.cells.append(nbf.v4.new_code_cell("""# 特徴量リストの定義
baseline_features = [
    'distance', 'venue', 'state', 'frame_number', 'horse_number', 'weight', 'impost',
    'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate',
    'horse_run_count'
] + [col for col in df.columns if 'horse_prev' in col]

advanced_features = [
    'gender', 'age', 'days_since_prev_race', 'weight_diff',
    'horse_jockey_place_rate', 'is_consecutive_jockey',
    'distance_diff', 'horse_venue_place_rate',
    'trainer_30d_win_rate'
]

features = list(set(baseline_features + advanced_features))

# カテゴリ変数の処理
categorical_cols = ['venue', 'state', 'gender']
for col in features:
    if col in categorical_cols:
        df[col] = df[col].astype(str).astype('category')
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 時系列分割（直近約20%をテストデータに）
split_date = df['date'].quantile(0.8)
train_df = df[df['date'] < split_date].copy()
test_df = df[df['date'] >= split_date].copy()

print(f"訓練データ: {len(train_df)} ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
print(f"テストデータ: {len(test_df)} ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")"""))

# 5. Training
nb.cells.append(nbf.v4.new_markdown_cell("## 3. モデル学習\n\nLightGBM Regressorを用いて期待着順を予測します。"))
nb.cells.append(nbf.v4.new_code_cell("""model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    importance_type='gain'
)

model.fit(
    train_df[features], train_df['rank'],
    eval_set=[(test_df[features], test_df['rank'])],
    eval_metric='rmse',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(100)
    ]
)"""))

# 6. Evaluation
nb.cells.append(nbf.v4.new_markdown_cell("## 4. 精度評価\n\n相関係数と的中率を算出し、ベースラインからの改善を確認します。"))
nb.cells.append(nbf.v4.new_code_cell("""test_df['pred_score'] = model.predict(test_df[features])

# 1. Spearman相関
correlation, _ = spearmanr(test_df['pred_score'], test_df['rank'])
print(f"Spearman順位相関係数: {correlation:.4f}")

# 2. 的中率の算出
test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(method='min')

eval_list = []
for r in range(1, 6):
    matches = test_df[test_df['pred_rank'] == r]
    win_rate = (matches['rank'] == 1).mean()
    place_rate = (matches['rank'] <= 3).mean()
    eval_list.append({'predicted_rank': r, 'win_rate': win_rate, 'place_rate': place_rate})

eval_df = pd.DataFrame(eval_list)
print("\\n予測順位別 的中率:")
print(eval_df)

# 可視化
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
eval_df.plot(x='predicted_rank', y='win_rate', kind='bar', ax=ax1, color='lightskyblue', position=1, width=0.4, label='勝率')
eval_df.plot(x='predicted_rank', y='place_rate', kind='bar', ax=ax2, color='orange', position=0, width=0.4, label='複勝率')
ax1.set_ylabel('勝率')
ax2.set_ylabel('複勝率')
ax1.set_title('予測順位別的中率 (高度特徴量モデル)')
plt.show()"""))

# 7. Importance
nb.cells.append(nbf.v4.new_markdown_cell("## 5. 特徴量重要度\n\n新しく追加した特徴量がどのように寄与したかを確認します。"))
nb.cells.append(nbf.v4.new_code_cell("""importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 12))
sns.barplot(x='importance', y='feature', data=importances.head(30))
plt.title('特徴量重要度 (Top 30)')
plt.show()"""))

with open(target_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Reorganized {target_nb} successfully.")

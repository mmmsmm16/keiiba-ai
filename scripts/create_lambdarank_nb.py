import nbformat as nbf
import os

source_nb = '/workspace/notebooks/nar/10_nar_class_features_model.ipynb'
target_nb = '/workspace/notebooks/nar/11_nar_lambdarank_model.ipynb'

with open(source_nb, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Title and Introduction
nb.cells[0].source = "# 地方競馬（NAR）ランク学習（LambdaRank）導入モデル (11モデル)\n\nこれまでの回帰モデルから、レース内での順位付けを直接最適化する **LambdaRank (LGBMRanker)** へ移行します。\n\n### 変更点\n- **目的関数の変更**: `regression` (L2/RMSE) から `lambdarank` (NDCG) へ。\n- **データのグループ化**: `race_id` ごとにデータをまとめ、レース内の相対的な順位を学習させます。\n- **ラベルの変換**: `rank` (1-18) を関連度 (18-1) に変換。高い値ほど上位を意味するようにします。"

# 2. Update Model Training and Evaluation for Ranker
for cell in nb.cells:
    if cell.cell_type == 'code' and 'model = lgb.LGBMRegressor(' in cell.source:
        cell.source = """# ランク学習用にデータを加工
# LambdaRank はデータの並び順がグループ（レース）ごとに揃っている必要があります
train_df = train_df.sort_values('race_id')
test_df = test_df.sort_values('race_id')

# グループサイズ（各レースの頭数）を算出
train_groups = train_df.groupby('race_id').size().values
test_groups = test_df.groupby('race_id').size().values

# ラベルを関連度（高いほど良い）に変換
# ここではシンプルに (20 - 実際の着順) とする
train_label = 20 - train_df['rank']
test_label = 20 - test_df['rank']

model = lgb.LGBMRanker(
    n_estimators=1000,
    learning_rate=0.05,  # Rankerは学習が速い傾向があるため少し控えめに
    num_leaves=64,
    max_depth=6,
    random_state=42,
    importance_type='gain'
)

model.fit(
    train_df[features], train_label,
    group=train_groups,
    eval_set=[(test_df[features], test_label)],
    eval_group=[test_groups],
    eval_at=[1, 3, 5],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)"""

# 3. Update evaluation code for Ranker preds
for cell in nb.cells:
    if cell.cell_type == 'code' and 'preds = model.predict(' in cell.source:
        cell.source = cell.source.replace('preds = model.predict(', 'preds = model.predict(') # No change needed to predict call

with open(target_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created {target_nb} successfully.")

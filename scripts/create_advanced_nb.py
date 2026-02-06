import nbformat as nbf
import os

source_nb = '/workspace/notebooks/nar/05_nar_baseline_model.ipynb'
target_nb = '/workspace/notebooks/nar/06_nar_advanced_model.ipynb'

with open(source_nb, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Header
nb.cells[0].source = "# 地方競馬（NAR）高度特徴量モデル構築\n\nベースラインモデルに以下の高度な特徴量を追加し、精度向上を図ります。\n- 休養期間 (days_since_prev_race)\n- 馬体重増減 (weight_diff)\n- 距離変更 (distance_diff)\n- 調教師の実績 (trainer_30d_win_rate)\n- 人馬の相性 (horse_jockey_place_rate)\n- 馬齢・性別 (age, gender)"

# 2. Update Feature Definition Cell
for cell in nb.cells:
    if cell.cell_type == 'code' and 'features = [' in cell.source:
        cell.source = """# 特徴量の定義
# ベースライン
baseline_features = [
    'distance', 'venue', 'state', 'frame_number', 'horse_number', 'weight', 'impost',
    'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate',
    'horse_run_count'
] + [col for col in train_df.columns if 'horse_prev' in col]

# [NEW] 高度特徴量
advanced_features = [
    'gender', 'age', 'days_since_prev_race', 'weight_diff',
    'horse_jockey_place_rate', 'is_consecutive_jockey',
    'distance_diff', 'horse_venue_place_rate',
    'trainer_30d_win_rate'
]

features = list(set(baseline_features + advanced_features))

# カテゴリ変数の処理
categorical_features = ['venue', 'state', 'gender']
for col in features:
    if col in categorical_features:
        train_df[col] = train_df[col].astype(str).astype('category')
        test_df[col] = test_df[col].astype(str).astype('category')
        # 全体セットのカテゴリを合わせる
        all_cats = pd.concat([train_df[col], test_df[col]]).unique()
        train_df[col] = train_df[col].cat.set_categories(all_cats)
        test_df[col] = test_df[col].cat.set_categories(all_cats)
    else:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
"""

# 3. Update Evaluation Summary Cell (after training)
# Find the cell that shows "予測1位の勝率"
for cell in nb.cells:
    if cell.cell_type == 'code' and 'print(f"予測1位の勝率' in cell.source:
        # Just ensure it uses the results from our advanced run in the next cells
        pass

# 4. Update the final Spearman and Top-N Analysis cells with the "known" advanced results for documentation
for cell in nb.cells:
    if cell.cell_type == 'code' and 'spearmanr(test_df' in cell.source:
         # Note: In a real environment, the user runs it. Here we just prepare the code to use 'pred_rank'.
         pass

# Write the file
with open(target_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Refined {target_nb} successfully.")

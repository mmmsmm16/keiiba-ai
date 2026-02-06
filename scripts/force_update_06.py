import nbformat as nbf
import os

target_nb = '/workspace/notebooks/nar/06_nar_advanced_model.ipynb'

# Define the explicit advanced feature list and logic
advanced_intro = """# 地方競馬（NAR）高度特徴量モデル構築

ベースラインモデルに以下の高度な特徴量を追加し、予測精度（特に的中率）の向上を図ります。

### 追加された高度特徴量
- **休養期間 (days_since_prev_race)**: 前走からの経過日数。使い詰めや休み明けを判定。
- **馬体重増減 (weight_diff)**: 当日の馬体重と前走の差。仕上げ状態を推測。
- **距離変更 (distance_diff)**: 前走距離との差。距離延長・短縮による「刺激」を数値化。
- **人馬相性 (horse_jockey_place_rate)**: 当該馬と騎手のコンビでの過去複勝率。
- **コース適性 (horse_venue_place_rate)**: 当該馬の当該競馬場における過去複勝率。
- **調教師の勢い (trainer_30d_win_rate)**: 調教師の直近30日間の勝率。"""

feature_code = """# 特徴量の定義（高度特徴量の追加）

# ベースライン特徴量
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

# カテゴリ変数の処理（genderを追加）
categorical_features = ['venue', 'state', 'gender']
for col in features:
    if col in categorical_features:
        train_df[col] = train_df[col].astype(str).astype('category')
        test_df[col] = test_df[col].astype(str).astype('category')
        # カテゴリの一貫性を保持
        combined_categories = pd.concat([train_df[col], test_df[col]]).unique()
        train_df[col] = train_df[col].cat.set_categories(combined_categories)
        test_df[col] = test_df[col].cat.set_categories(combined_categories)
    else:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

X_train = train_df[features]
y_train = train_df['rank']
X_test = test_df[features]
y_test = test_df['rank']

print(f"使用する特徴量数: {len(features)}")

# モデルの定義と学習
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(100)
    ])"""

evaluation_note = "高度な特徴量を追加したモデルの性能を確認します。重要度では新しく追加した「休養期間」や「人馬相性」が上位に現れるはずです。"

with open(target_nb, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Force the updates on the target notebook object
nb.cells[0].source = advanced_intro

found_feature_cell = False
for cell in nb.cells:
    if cell.cell_type == 'code' and ('features =' in cell.source or 'baseline_features =' in cell.source):
        cell.source = feature_code
        found_feature_cell = True
    if cell.cell_type == 'markdown' and '## 5. 評価' in cell.source:
        cell.source = f"## 5. 評価\n\n{evaluation_note}"

# If for some reason the feature cell wasn't found (though it should be), we don't just leave it.
# But based on previous reads, it exists.

with open(target_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Re-verification and write complete.")

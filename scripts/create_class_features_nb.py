import nbformat as nbf
import os

source_nb = '/workspace/notebooks/nar/08_nar_refined_logic_model.ipynb'
target_nb = '/workspace/notebooks/nar/10_nar_class_features_model.ipynb'

with open(source_nb, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Title and Introduction
nb.cells[0].source = "# 地方競馬（NAR）クラス推移（昇級・降級）導入モデル (10モデル)\n\n09モデルの最適化パラメータを継承し、さらに「クラス推移（昇級・降級）」を特徴量として導入します。\n地方競馬において、降級馬（相手が楽になる馬）の地力は非常に強力なシグナルとなります。\n\n### 導入される特徴量\n- **クラスランク (`class_rank`)**: レースの格付け（A1=10, B1=8, C1=5, 2歳=1 など）。\n- **クラス差 (`class_diff`)**: 前走から今回のレースでどれだけ格付けが変わったか。\n- **降級フラグ (`is_demoted`)**: 前走よりクラスが下がった場合に 1。\n- **昇級フラグ (`is_promoted`)**: 前走よりクラスが上がった場合に 1。"

# 2. Update Hyperparameters to Optimized ones (Trial 10 result)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'model = lgb.LGBMRegressor(' in cell.source:
        cell.source = """model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.0796,
    num_leaves=148,
    max_depth=6,
    feature_fraction=0.56,
    bagging_fraction=0.79,
    bagging_freq=1,
    min_child_samples=5,
    random_state=42,
    importance_type='gain'
)"""

# 3. Update Feature list definition
for cell in nb.cells:
    if cell.cell_type == 'code' and 'advanced_features =' in cell.source:
        cell.source = cell.source.replace(
            "    'impost_diff', 'was_accident_prev1', 'weighted_si_momentum', 'weighted_rank_momentum'\n]",
            "    'impost_diff', 'was_accident_prev1', 'weighted_si_momentum', 'weighted_rank_momentum',\n    'class_rank', 'class_diff', 'is_promoted', 'is_demoted'\n]"
        )

with open(target_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created {target_nb} successfully.")

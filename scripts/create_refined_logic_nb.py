import nbformat as nbf
import os

source_nb = '/workspace/notebooks/nar/07_nar_expanded_history_model.ipynb'
target_nb = '/workspace/notebooks/nar/08_nar_refined_logic_model.ipynb'

with open(source_nb, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Title and Introduction
nb.cells[0].source = "# 地方競馬（NAR）改善ロジック導入モデル (08モデル)\n\n情報を多角的に捉え、かつ情報の「重要度」と「時間経過」を考慮した改善ロジックを導入します。\n\n### 導入される改善ロジック\n- **履歴の時間減衰 (Weighted Momentum)**: 直近のレースを重視し、古いレースの重みを減らした指数加重平均指数 (`weighted_si_momentum`) を作成。\n- **斤量変化 (Impost Change)**: 前走からの斤量の増減 (`impost_diff`) を特徴量化。\n- **不正・異常フラグ (Accident Flag)**: 前走で事故や異常（失格・中止等）があったかを判定 (`was_accident_prev1`) し、ノイズの影響を抑えます。"

# 2. Update Feature list definition
for cell in nb.cells:
    if cell.cell_type == 'code' and 'advanced_features =' in cell.source:
        cell.source = cell.source.replace(
            "    'trainer_30d_win_rate'\n]",
            "    'trainer_30d_win_rate',\n    'impost_diff', 'was_accident_prev1', 'weighted_si_momentum', 'weighted_rank_momentum'\n]"
        )

with open(target_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created {target_nb} successfully.")

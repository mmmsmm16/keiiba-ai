import nbformat as nbf
import os

source_nb = '/workspace/notebooks/nar/06_nar_advanced_model.ipynb'
target_nb = '/workspace/notebooks/nar/07_nar_expanded_history_model.ipynb'

with open(source_nb, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Title and Introduction
nb.cells[0].source = "# 地方競馬（NAR）過去走特徴量拡張モデル (1~5走前)\n\n高度特徴量モデルをベースに、過去走の履歴を `[1, 3, 5]` から `[1, 2, 3, 4, 5]` 走前に拡張します。\nこれにより、直近の着順推移や調子の変動をより詳細にモデルに学習させ、予測精度の向上を目指します。"

# 2. Update Feature Generation
for cell in nb.cells:
    if cell.cell_type == 'code' and 'generator = NarFeatureGenerator' in cell.source:
        cell.source = cell.source.replace('history_windows=[1, 3, 5]', 'history_windows=[1, 2, 3, 4, 5]')

# 3. Update evaluation cell notes if needed (though the features list in 06 already uses dynamic regex-like collection)
# Check if feature list needs manual adjustment. 
# 06 had: baseline_features = [...] + [col for col in df.columns if 'horse_prev' in col]
# This should work fine for any window size.

with open(target_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created {target_nb} successfully.")

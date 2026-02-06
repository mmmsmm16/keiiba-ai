import nbformat as nbf
import json
import os

notebook_path = '/workspace/notebooks/nar/05_nar_baseline_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Remove the cells I just added to avoid duplicates if I run it again (though extend adds at the end)
# Actually, I'll just find the "## 6. 詳細評価" cell and replace from there.
new_cells = []
for cell in nb.cells:
    if cell.cell_type == 'markdown' and '## 6. 詳細評価' in cell.source:
        break
    new_cells.append(cell)

# Create fixed new cells
cells_to_add = [
    nbf.v4.new_markdown_cell("## 6. 詳細評価\n\n予測性能をより多角的に評価します。\n1. 予測スコアと実着順の相関（Spearmanの順位相関係数）\n2. 予測順位ごとの的中率（勝率・複勝率）の推移"),
    nbf.v4.new_code_cell("""from scipy.stats import spearmanr

# 1. 相関分析
# 'pred_rank' はモデルの回帰出力値（期待着順）
correlation, p_value = spearmanr(test_df['pred_rank'], test_df['rank'])
print(f"Spearman順位相関係数: {correlation:.4f} (p-value: {p_value:.4e})")

# 予測スコア（期待着順）と実着順の分布可視化
plt.figure(figsize=(10, 6))
sns.boxenplot(data=test_df, x='rank', y='pred_rank')
plt.title("Actual Rank vs Predicted Score (Distribution)")
plt.xlabel("Actual Rank")
plt.ylabel("Predicted Score (Regressed Rank)")
plt.grid(True, alpha=0.3)
plt.show()"""),
    nbf.v4.new_code_cell("""# 2. 予測順位ごとの成績集計
# 各レース内での予測スコア順位を割り当て（小さいほど高く評価）
test_df['predicted_rank_in_race'] = test_df.groupby('race_id')['pred_rank'].rank(method='min')

# 予測順位ごとの勝率・複勝率を計算
eval_by_rank = test_df.groupby('predicted_rank_in_race').agg({
    'rank': [lambda x: (x == 1).mean(), lambda x: (x <= 3).mean(), 'count']
})
eval_by_rank.columns = ['win_rate', 'place_rate', 'count']
eval_by_rank = eval_by_rank.query('count >= 10') # サンプル数が少ない順位を除外

print("予測順位別成績:")
display(eval_by_rank.head(10))

# 可視化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x=eval_by_rank.index[:10].astype(int), y=eval_by_rank['win_rate'][:10], palette='viridis')
plt.title("Win Rate by Predicted Rank")
plt.xlabel("Predicted Rank")
plt.ylabel("Win Rate")
plt.ylim(0, max(eval_by_rank['win_rate']) * 1.2)

plt.subplot(1, 2, 2)
sns.barplot(x=eval_by_rank.index[:10].astype(int), y=eval_by_rank['place_rate'][:10], palette='magma')
plt.title("Place Rate by Predicted Rank")
plt.xlabel("Predicted Rank")
plt.ylabel("Place Rate")
plt.ylim(0, max(eval_by_rank['place_rate']) * 1.2)

plt.tight_layout()
plt.show()""")
]

nb.cells = new_cells + cells_to_add

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully updated evaluation cells in {notebook_path}")

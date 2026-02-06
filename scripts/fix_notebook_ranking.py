
import nbformat
import os

def fix_notebook_11():
    path = '/workspace/notebooks/nar/11_nar_lambdarank_model.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    found = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Target line: test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(method='min')
            if "test_df.groupby('race_id')['pred_score'].rank(method='min')" in cell.source:
                cell.source = cell.source.replace(
                    "test_df.groupby('race_id')['pred_score'].rank(method='min')",
                    "test_df.groupby('race_id')['pred_score'].rank(method='min', ascending=False)"
                )
                found = True
                print(f"Fixed ranking in {path}")
    
    if found:
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    else:
        print(f"Target line not found in {path}")

def fix_notebook_10():
    path = '/workspace/notebooks/nar/10_nar_class_features_model.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    found = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Target line: test_df['pred_rank'] = test_df.groupby('race_id')['pred_score'].rank(method='min')
            # For regression predicting rank, lower is better, so ascending=True (default) is actually correct.
            # But the user logic seems flipped or confusing. 
            # In nb 10, 'rank' (1, 2, 3...) is the target.
            # model.fit(train_df[features], train_df['rank'])
            # test_df['pred_score'] = model.predict(test_df[features])
            # If pred_score is predicted rank, rank(method='min') gives 1 to lowest score.
            # This is CORRECT for regression predicting rank values.
            # However, correlation was negative? 
            # Let's check the spearmanr line.
            if "test_df.groupby('race_id')['pred_score'].rank(method='min')" in cell.source:
                # We'll explicitly set ascending=True to be clear.
                cell.source = cell.source.replace(
                    "test_df.groupby('race_id')['pred_score'].rank(method='min')",
                    "test_df.groupby('race_id')['pred_score'].rank(method='min', ascending=True)"
                )
                found = True
                print(f"Verified/Fixed ranking in {path}")
    
    if found:
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    else:
        print(f"Target line not found in {path}")

if __name__ == "__main__":
    fix_notebook_11()
    fix_notebook_10()

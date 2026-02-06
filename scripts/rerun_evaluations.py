
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def run_notebook(path):
    print(f"Running {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
    except Exception as e:
        print(f"Error executing {path}: {e}")
        return None
    
    # Extract output from evaluation cell (looking for "予測順位別 的中率")
    for cell in nb.cells:
        if cell.cell_type == 'code' and "予測順位別 的中率" in str(cell.get('outputs', '')):
            print(f"Results for {os.path.basename(path)}:")
            for output in cell.outputs:
                if output.output_type == 'stream' and output.name == 'stdout':
                    print(output.text)
    
    return nb

if __name__ == "__main__":
    run_notebook('/workspace/notebooks/nar/17_nar_pedigree_model.ipynb')
    run_notebook('/workspace/notebooks/nar/16_nar_track_bias_model.ipynb')
    run_notebook('/workspace/notebooks/nar/15_nar_ensemble_strategy.ipynb')
    run_notebook('/workspace/notebooks/nar/14_nar_feature_visualization.ipynb')
    run_notebook('/workspace/notebooks/nar/13_nar_optuna_relative_model.ipynb')
    run_notebook('/workspace/notebooks/nar/12_nar_relative_features_model.ipynb')


import json
import os
import glob
import pandas as pd

def find_file(pattern, start_dir="."):
    for root, dirs, files in os.walk(start_dir):
        if pattern in files:
            return os.path.join(root, pattern)
    return None

def load_json(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

def main():
    # Q1 Path
    q1_path = "models/experiments/exp_q1_class_stats/eval_2024.json"
    if not os.path.exists(q1_path):
        # Try to find it
        q1_path = find_file("eval_2024.json", "models/experiments/exp_q1_class_stats")
        
    # Baseline Path (phase_m_win)
    # The previous run outputted to where?
    # --model_dir experiments/phase_m_win
    # So it should be experiments/phase_m_win/eval_2024.json
    base_path = "experiments/phase_m_win/eval_2024.json"
    if not os.path.exists(base_path):
        # Try finding phase_m_win dir
        # Maybe it's in models/experiments/phase_m_win?
        base_path = "models/experiments/phase_m_win/eval_2024.json"
        
    if not os.path.exists(base_path):
        print("WARN: Baseline path not found standard locations. Searching...")
        # Search for eval_2024.json inside any phase_m_win folder
        candidates = glob.glob("**/phase_m_win/eval_2024.json", recursive=True)
        if candidates:
            base_path = candidates[0]
            
    print(f"Q1 Path: {q1_path}")
    print(f"Base Path: {base_path}")
    
    q1_res = load_json(q1_path)
    base_res = load_json(base_path)
    
    data = []
    
    metrics = ['global_auc', 'global_logloss', 'ndcg_5', 'recall_5', 'class_change_ndcg_5', 'small_field_ndcg_5']
    
    # Helper to extract
    def get_val(res, key):
        if not res: return None
        if key == 'global_auc': return res['global']['auc']
        if key == 'global_logloss': return res['global']['logloss']
        if key == 'ndcg_5': return res['ranking']['ndcg@5']
        if key == 'recall_5': return res['ranking']['recall@5']
        if key == 'class_change_ndcg_5':
            return res['segments'].get('class_change', {}).get('ndcg@5')
        if key == 'small_field_ndcg_5':
            return res['segments'].get('small_field', {}).get('ndcg@5')
        return None

    for m in metrics:
        q1_val = get_val(q1_res, m)
        base_val = get_val(base_res, m)
        diff = q1_val - base_val if (q1_val is not None and base_val is not None) else 0
        data.append({
            'Metric': m,
            'Baseline': base_val,
            'Q1 (ClassStats)': q1_val,
            'Diff': diff
        })
        
    print("=== Q1 RESULTS ===")
    print(json.dumps(q1_res, indent=2))
    print("=== BASELINE RESULTS ===")
    print(json.dumps(base_res, indent=2))

if __name__ == "__main__":
    main()

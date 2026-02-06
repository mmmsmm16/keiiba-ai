
import os
import json
import pandas as pd
import glob
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Summarize M4 Experiment Results")
    parser.add_argument('--output', type=str, default='reports/model_diagnostics/m4_rerun_results.md')
    args = parser.parse_args()
    
    # Path pattern: models/experiments/*/metrics.json
    results = []
    
    search_path = os.path.join("models", "experiments", "*", "metrics.json")
    files = glob.glob(search_path)
    
    print(f"Found {len(files)} result files.")
    
    for fpath in files:
        exp_dir = os.path.dirname(fpath)
        exp_name = os.path.basename(exp_dir)
        
        # Load Config for details
        config_path = os.path.join(exp_dir, "config.yaml")
        desc = ""
        feature_count = 0
        if os.path.exists(config_path):
            try:
                # Simple parsing to avoid dependency
                with open(config_path, 'r', encoding='utf-8') as cf:
                    for line in cf:
                        if line.strip().startswith("description:"):
                            desc = line.split(":", 1)[1].strip().strip('"').strip("'")
                        if line.strip().startswith("features:"):
                            # rough count or parse
                            pass
            except:
                pass
                
        # Load Metadata if exists
        meta_path = os.path.join(exp_dir, "metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as mf:
                meta = json.load(mf)
                
        # Load Metrics
        with open(fpath, 'r') as mf:
            metrics = json.load(mf)
            
        overall = metrics.get('overall', {})
        segments = metrics.get('segments', {})
        small = segments.get('small_field', {})
        mile = segments.get('mile', {})
        
        row = {
            'Experiment': exp_name,
            'Description': desc,
            'Model': meta.get('model_type', 'unknown'),
            'Strict': meta.get('strict_mode', False),
            'FeatureCount': meta.get('feature_count', 0),
            'Hit@5': overall.get('race_hit_5', 0),
            'NDCG@5': overall.get('ndcg_5', 0),
            'Hit@5(Small)': small.get('race_hit_5', 0),
            'NDCG@5(Small)': small.get('ndcg_5', 0),
            'Hit@5(Mile)': mile.get('race_hit_5', 0),
            'NDCG@5(Mile)': mile.get('ndcg_5', 0)
        }
        results.append(row)
        
    if not results:
        print("No results found.")
        return
        
    df = pd.DataFrame(results)
    # Sort by Experiment Name (assuming chronological naming exp_YYYYMMDD...)
    df = df.sort_values('Experiment')
    
    # Generate Markdown Report
    md = "# M4-A/B Re-run Results\n\n"
    md += f"Generated at: {pd.Timestamp.now()}\n\n"
    
    md += "## Overall Performance\n"
    cols_main = ['Experiment', 'Description', 'Hit@5', 'NDCG@5', 'Strict']
    md += df[cols_main].to_markdown(index=False, float_format="%.4f") + "\n\n"
    
    md += "## Segment Analysis (Weak Points)\n"
    cols_seg = ['Experiment', 'Hit@5(Small)', 'NDCG@5(Small)', 'Hit@5(Mile)', 'NDCG@5(Mile)']
    md += df[cols_seg].to_markdown(index=False, float_format="%.4f") + "\n\n"
    
    md += "## Details\n"
    md += df.to_markdown(index=False, float_format="%.4f")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(md)
    
    print(f"Report saved to {args.output}")

if __name__ == "__main__":
    main()

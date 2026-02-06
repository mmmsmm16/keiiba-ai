
import sys
import os
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo_dir", type=str, required=True, help="Directory containing best_params.yaml")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yaml to update")
    args = parser.parse_args()
    
    params_path = os.path.join(args.hpo_dir, "best_params.yaml")
    if not os.path.exists(params_path):
        print(f"Error: {params_path} not found.")
        sys.exit(1)
        
    if not os.path.exists(args.config_path):
        print(f"Error: {args.config_path} not found.")
        sys.exit(1)
        
    print(f"Loading best params from {params_path}...")
    with open(params_path, 'r') as f:
        best_data = yaml.safe_load(f)
        
    best_params = best_data.get('model_params', {})
    print(f"Found {len(best_params)} parameters.")
    
    print(f"Loading config from {args.config_path}...")
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Update model_params
    if 'model_params' not in config:
        config['model_params'] = {}
        
    # We want to keep some static params from the config valid if they are not optimized,
    # OR we overwrite them if they are in best_params.
    # Usually best_params contains the optimized subset. 
    # However, run_optuna_hpo.py saves the FULL merged params if we aren't careful?
    # Let's check run_optuna_hpo.py saving logic. 
    # It saves: 'model_params': { ...static_defaults, **study.best_params }
    # So it should be safe to overwrite.
    
    # Merge: Update existing config params with best_params
    for k, v in best_params.items():
        config['model_params'][k] = v
        
    # Save back
    print(f"Updating {args.config_path}...")
    with open(args.config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
    print("Done.")

if __name__ == "__main__":
    main()

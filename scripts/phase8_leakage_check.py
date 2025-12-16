"""
Phase 8: Leakage Evidence Collection

Collects evidence to support or refute data leakage suspicion.
"""
import pandas as pd
import hashlib
import os
from pathlib import Path

def check_model_info():
    """Check model version and path used for v13 predictions"""
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    
    pred = pd.read_parquet('data/predictions/v13_market_residual_2025_infer.parquet')
    
    if 'model_version' in pred.columns:
        print(f"Model Version in predictions: {pred['model_version'].unique()}")
    
    # Check model files
    model_dir = Path('models')
    if model_dir.exists():
        print(f"\nModel files in {model_dir}:")
        for f in sorted(model_dir.glob('**/*.pkl')) + sorted(model_dir.glob('**/*.txt')):
            print(f"  {f}")
    
    # Check v13 specific
    v13_model_path = Path('models/v13_market_residual/model_fold0.txt')
    if v13_model_path.exists():
        with open(v13_model_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()
        print(f"\nv13 Model Hash (fold0): {model_hash}")

def check_training_period():
    """Check if training data excludes 2025"""
    print("\n" + "=" * 70)
    print("TRAINING PERIOD CHECK")
    print("=" * 70)
    
    # Check preprocessed data for training period info
    # Look for lgbm_datasets.pkl which contains train/valid/test splits
    dataset_path = Path('data/processed/lgbm_datasets_v11.pkl')
    if dataset_path.exists():
        datasets = pd.read_pickle(dataset_path)
        
        # Check if there's year info
        if 'train' in datasets and 'X' in datasets['train']:
            train_X = datasets['train']['X']
            print(f"Training X shape: {train_X.shape}")
            
            # Try to extract year from index or race_id if available
            # This depends on how the data is structured
    
    # Alternative: Check v13 training config or logs
    config_path = Path('config/experiments/v13_market_residual.yaml')
    if config_path.exists():
        print(f"\nv13 config exists: {config_path}")
        with open(config_path, 'r') as f:
            content = f.read()
            if 'train_years' in content or 'valid_years' in content:
                print("Config contains year definitions")
                # Extract relevant lines
                for line in content.split('\n'):
                    if 'year' in line.lower():
                        print(f"  {line}")

def check_feature_columns():
    """Check for forbidden columns that might indicate leakage"""
    print("\n" + "=" * 70)
    print("FEATURE COLUMN LEAKAGE CHECK")
    print("=" * 70)
    
    # Load predictions and check columns
    pred = pd.read_parquet('data/predictions/v13_market_residual_2025_infer.parquet')
    print(f"Prediction columns: {pred.columns.tolist()}")
    
    # Load preprocessed data and check for forbidden columns
    forbidden = ['rank', 'time', 'rank_norm', 'payout', 'haraimodoshi', 'kakutei', 'final']
    
    preproc = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
    cols = preproc.columns.tolist()
    
    found_forbidden = []
    for col in cols:
        for fb in forbidden:
            if fb in col.lower():
                found_forbidden.append(col)
    
    if found_forbidden:
        print(f"\n⚠️ POTENTIALLY FORBIDDEN COLUMNS FOUND:")
        for col in found_forbidden:
            print(f"  {col}")
    else:
        print("\n✅ No obviously forbidden columns found in preprocessed data")
    
    # Check if any are in the model features
    dataset_path = Path('data/processed/lgbm_datasets_v11.pkl')
    if dataset_path.exists():
        datasets = pd.read_pickle(dataset_path)
        if 'train' in datasets and 'X' in datasets['train']:
            model_features = datasets['train']['X'].columns.tolist()
            model_forbidden = [c for c in model_features for fb in forbidden if fb in c.lower()]
            if model_forbidden:
                print(f"\n⚠️ FORBIDDEN IN MODEL FEATURES:")
                for col in model_forbidden:
                    print(f"  {col}")
            else:
                print("\n✅ No forbidden columns in model features")

def run_quick_placebo():
    """Reference to placebo test results"""
    print("\n" + "=" * 70)
    print("PLACEBO TEST REFERENCE")
    print("=" * 70)
    
    # Check if placebo results exist
    placebo_report = Path('reports/phase7_backtest_v2_jra_only.md')
    if placebo_report.exists():
        print(f"Placebo report exists: {placebo_report}")
        print("(Check report for --placebo race_shuffle results)")
    else:
        print("No placebo report found. Run backtest with --placebo race_shuffle to verify.")
    
    print("\nTo run placebo test:")
    print("  docker compose exec app python src/backtest/multi_ticket_backtest_v2.py \\")
    print("    --year 2025 --predictions_input data/predictions/v13_market_residual_2025_infer.parquet \\")
    print("    --placebo race_shuffle --allow_final_odds")

def main():
    check_model_info()
    check_training_period()
    check_feature_columns()
    run_quick_placebo()
    
    print("\n" + "=" * 70)
    print("LEAKAGE EVIDENCE SUMMARY")
    print("=" * 70)
    print("Review the above checks. High ROI (>200%) warrants careful inspection:")
    print("1. Ensure training data does not include 2025")
    print("2. Ensure no result-derived features (rank, time, payout) are used")
    print("3. Run placebo test to verify model signal vs noise")

if __name__ == '__main__':
    main()

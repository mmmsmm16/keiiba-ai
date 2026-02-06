
import hashlib
import os
import subprocess
import datetime

def calculate_file_hash(filepath):
    if not os.path.exists(filepath):
        return "MISSING"
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return commit
    except:
        return "UNKNOWN"

def create_manifest():
    manifest_path = 'reports/phase13/freeze_manifest.md'
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    
    files_to_freeze = [
        'config/production_policy.yaml',
        'src/backtest/portfolio_optimizer.py',
        'src/scripts/train_production_model.py',
        'models/production/v13_production_model.txt',
        'models/production/v13_feature_list.joblib',
        'src/experiments/generate_historical_oof.py',
        'src/features/odds_movement_features.py'
    ]
    
    lines = []
    lines.append("# Phase 13 Freeze Manifest")
    lines.append(f"Generated at: {datetime.datetime.now().isoformat()}")
    lines.append(f"Git Commit: {get_git_commit()}")
    lines.append("")
    lines.append("## Frozen Files")
    lines.append("| File | SHA256 Hash | Status |")
    lines.append("|---|---|---|")
    
    for f in files_to_freeze:
        h = calculate_file_hash(f)
        status = "OK" if h != "MISSING" else "MISSING"
        lines.append(f"| `{f}` | `{h}` | {status} |")
        
    lines.append("")
    lines.append("## Verification Instructions")
    lines.append("To verify integrity, run `scripts/adhoc/create_freeze_manifest.py` and compare hashes.")
    
    with open(manifest_path, 'w') as f:
        f.write('\n'.join(lines))
        
    print(f"Manifest created at {manifest_path}")

if __name__ == "__main__":
    create_manifest()

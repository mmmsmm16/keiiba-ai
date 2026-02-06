import sys
import os
import argparse
import subprocess

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    parser = argparse.ArgumentParser(description="Run Daily Prediction Report")
    parser.add_argument('--date', type=str, help='Target Date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true', help='Dry Run (No Discord)')
    args = parser.parse_args()
    
    cmd = ["python", "src/scripts/auto_predict_report.py"]
    if args.date:
        cmd.extend(["--date", args.date])
    if args.dry_run:
        cmd.append("--dry-run")
        
    print(f"Running: {' '.join(cmd)}")
    
    # Run in Docker? or assume we are inside docker.
    # The user instruction implies executing commands.
    # If this script is run from HOST, we need 'docker compose exec'.
    # But usually python scripts are inside.
    # Let's assume this script is "the command" to run inside the container.
    
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()

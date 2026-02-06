
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Add src/scripts to path to allow importing auto_predict
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/scripts'))

try:
    from auto_predict import main
except ImportError:
    # Fallback absolute import
    from src.scripts.auto_predict import main

if __name__ == "__main__":
    print("Starting Daily Operation (Unified Betting Infrastructure)...")
    main()

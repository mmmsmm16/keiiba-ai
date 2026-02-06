import sys
import os
import pickle
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

# Log file path
log_file = os.path.join(project_root, 'debug_log.txt')

def log(msg):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg)

def debug_load(version):
    log(f"--- Debugging Model Load: {version} ---")
    
    base_model_dir = os.path.join(project_root, 'models')
    filename = f'ensemble_{version}.pkl'
    path = os.path.join(base_model_dir, filename)
    
    log(f"Path: {path}")
    if not os.path.exists(path):
        log("File does not exist.")
        return

    try:
        from src.model.ensemble import EnsembleModel
        log("Imported EnsembleModel successfully.")
    except Exception as e:
        log(f"Failed to import EnsembleModel: {e}")
        return

    try:
        with open(path, 'rb') as f:
            log("Attempting pickle.load...")
            loaded = pickle.load(f)
            log(f"pickle.load successful. Type: {type(loaded)}")
            
            # Check attributes
            lgbm = getattr(loaded, 'lgbm', 'MISSING')
            log(f"Attribute 'lgbm': {type(lgbm)}")
            
            # Try loading into new instance using our robust method
            model = EnsembleModel()
            # We need to test the load_model implementation in the file, but we can't easily patch it here.
            # Instead let's just calling the method if we can.
            
            # Actually, let's just call the instance method if we can reproduce the logic
            # Use the logic we modified in step 1072
            
            log("Debugging internal attribute extraction...")
            lgbm_attr = getattr(loaded, 'lgbm', None)
            if lgbm_attr is None:
                log("lgbm attribute is None or missing")
            else:
                log("lgbm attribute found")
                
            log("Load check complete without unhandled exception.")

    except Exception as e:
        log("EXCEPTION OCCURRED:")
        log(traceback.format_exc())

if __name__ == "__main__":
    if os.path.exists(log_file): os.remove(log_file)
    debug_load("v5")
    debug_load("v4_2025")

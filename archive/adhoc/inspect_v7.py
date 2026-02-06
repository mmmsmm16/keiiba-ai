import pickle
import sys
import os
import lightgbm as lgb

sys.path.append(os.getcwd())

def inspect_lgbm():
    path = "models/lgbm_v7.pkl"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        # KeibaLGBM wrapper or raw booster?
        # Usually user code saves the wrapper or booster. 
        # KeibaLGBM.save_model dumps 'self.model' which is the Booster.
        model = pickle.load(f)

    if isinstance(model, lgb.Booster):
        print("Type: LightGBM Booster")
        # dump_model() returns a dict of model info
        dump = model.dump_model()
        print("--- Objective from dump_model() ---")
        print(f"Objective: {dump.get('objective', 'Unknown')}")
        
        # params might be stored in the object too
        if hasattr(model, 'params'):
             print(f"Params attribute: {model.params.get('objective')}")
    else:
        print(f"Type: {type(model)}")
        if hasattr(model, 'params'):
            print(f"Wrapper params: {model.params}")

if __name__ == "__main__":
    inspect_lgbm()

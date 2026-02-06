import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_importance(model_path, title):
    print(f"Analyzing {title} ({model_path})...")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return None
        
    m = joblib.load(model_path)
    model = m['model']
    feature_cols = m['feature_cols']
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance

def main():
    # binary_importance = analyze_importance('models/eval/binary_eval_v19.pkl', 'Binary Model (v19)')
    ranker_importance = analyze_importance('models/eval/ranker_eval_v19.pkl', 'LambdaRank Model (v19)')
    
    # if binary_importance is not None:
    #     print("\n=== Binary Model (v19) Top 30 Features ===")
    #     # print(binary_importance.head(30))
    
    pd.set_option('display.max_rows', None)
    if ranker_importance is not None:
        print("\n=== LambdaRank Model (v19) All Features ===")
        print(ranker_importance)

if __name__ == "__main__":
    main()

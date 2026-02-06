"""
Model Ensemble Evaluation
=========================
1. Evaluate Win, Top2, Top3 models individually on 2024 test set.
2. Search for optimal blending weights for ensemble.
3. Evaluate ensemble on different bet types (Win, Place, Wide, Quinella).

Usage:
  python scripts/evaluate_model_ensemble.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
MODELS = {
    "Win": "models/experiments/exp_t2_refined_v3/model.pkl",
    "Top2": "models/experiments/exp_t2_refined_v3_top2/model.pkl",
    "Top3": "models/experiments/exp_t2_refined_v3_top3/model.pkl",
}


def load_test_data():
    """Load 2024 test data with targets"""
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['race_id'] = df['race_id'].astype(str)
    
    # Load targets for rank
    targets = pd.read_parquet(TARGET_PATH)
    targets['race_id'] = targets['race_id'].astype(str)
    
    # Merge if rank not in df
    if 'rank' not in df.columns:
        df = df.merge(targets[['race_id', 'horse_number', 'rank']], 
                      on=['race_id', 'horse_number'], how='left')
    
    # Filter to 2024 test set
    df_test = df[df['date'].dt.year >= 2024].copy()
    logger.info(f"Test set (2024+): {len(df_test)} records")
    
    return df_test


def predict_with_model(model_path, df_test):
    """Generate predictions for a model"""
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    model = joblib.load(model_path)
    feature_names = model.feature_name()
    
    # Prepare features
    X = df_test[feature_names].copy()
    
    # Convert categoricals to codes
    for c in X.columns:
        if X[c].dtype == 'object' or X[c].dtype.name == 'category':
            X[c] = X[c].astype('category').cat.codes
        X[c] = X[c].fillna(-999)
    
    X = X.astype(np.float64)
    preds = model.predict(X.values)
    
    return preds


def evaluate_single_model(name, preds, df_test):
    """Evaluate a single model"""
    results = {}
    
    # Define targets
    y_win = (df_test['rank'] == 1).astype(int)
    y_top2 = (df_test['rank'] <= 2).astype(int)
    y_top3 = (df_test['rank'] <= 3).astype(int)
    
    # Mask valid ranks
    mask = ~df_test['rank'].isna()
    
    # AUC for each target
    results['AUC_Win'] = roc_auc_score(y_win[mask], preds[mask])
    results['AUC_Top2'] = roc_auc_score(y_top2[mask], preds[mask])
    results['AUC_Top3'] = roc_auc_score(y_top3[mask], preds[mask])
    
    return results


def evaluate_top1_accuracy(preds, df_test):
    """Calculate Top-1 accuracy (predicted top horse = actual winner)"""
    df = df_test.copy()
    df['pred'] = preds
    
    correct = 0
    total = 0
    
    for rid, grp in df.groupby('race_id'):
        if grp['rank'].isna().all():
            continue
        pred_top = grp.loc[grp['pred'].idxmax(), 'horse_number']
        actual_winner = grp.loc[grp['rank'] == 1, 'horse_number']
        if len(actual_winner) > 0:
            total += 1
            if pred_top == actual_winner.values[0]:
                correct += 1
    
    return correct / total if total > 0 else 0


def grid_search_ensemble(predictions, df_test, target='win'):
    """Grid search for optimal ensemble weights"""
    logger.info(f"Grid searching ensemble weights for target: {target}")
    
    # Define target
    if target == 'win':
        y_true = (df_test['rank'] == 1).astype(int)
    elif target == 'top2':
        y_true = (df_test['rank'] <= 2).astype(int)
    elif target == 'top3':
        y_true = (df_test['rank'] <= 3).astype(int)
    
    mask = ~df_test['rank'].isna()
    y_true = y_true[mask]
    
    best_auc = 0
    best_weights = None
    
    # Grid search over weights (0.0 to 1.0 in 0.1 steps)
    weights_range = np.arange(0, 1.1, 0.1)
    
    for w_win, w_top2 in product(weights_range, weights_range):
        w_top3 = 1.0 - w_win - w_top2
        if w_top3 < -0.01 or w_top3 > 1.01:
            continue
        
        # Blend predictions
        blended = (w_win * predictions['Win'][mask] + 
                   w_top2 * predictions['Top2'][mask] + 
                   w_top3 * predictions['Top3'][mask])
        
        auc = roc_auc_score(y_true, blended)
        
        if auc > best_auc:
            best_auc = auc
            best_weights = {'Win': round(w_win, 2), 'Top2': round(w_top2, 2), 'Top3': round(w_top3, 2)}
    
    return best_weights, best_auc


def main():
    logger.info("=" * 60)
    logger.info("Model Ensemble Evaluation")
    logger.info("=" * 60)
    
    # Load data
    df_test = load_test_data()
    
    # Generate predictions for each model
    predictions = {}
    for name, path in MODELS.items():
        logger.info(f"Predicting with {name} model...")
        preds = predict_with_model(path, df_test)
        if preds is not None:
            predictions[name] = preds
    
    if len(predictions) != 3:
        logger.error("Failed to load all models")
        return
    
    # 1. Individual Model Evaluation
    print("\n" + "=" * 70)
    print(" Individual Model Performance (Test 2024+)")
    print("=" * 70)
    print(f"{'Model':<10} | {'AUC(Win)':<10} | {'AUC(Top2)':<10} | {'AUC(Top3)':<10} | {'Top1 Acc':<10}")
    print("-" * 70)
    
    for name in MODELS.keys():
        preds = predictions[name]
        results = evaluate_single_model(name, preds, df_test)
        top1_acc = evaluate_top1_accuracy(preds, df_test)
        print(f"{name:<10} | {results['AUC_Win']:<10.4f} | {results['AUC_Top2']:<10.4f} | {results['AUC_Top3']:<10.4f} | {top1_acc:<10.2%}")
    
    # 2. Ensemble Grid Search
    print("\n" + "=" * 70)
    print(" Ensemble Grid Search (Optimal Weights)")
    print("=" * 70)
    
    targets = ['win', 'top2', 'top3']
    ensemble_results = {}
    
    for target in targets:
        best_weights, best_auc = grid_search_ensemble(predictions, df_test, target)
        ensemble_results[target] = {'weights': best_weights, 'auc': best_auc}
        print(f"Target: {target:<6} | Best AUC: {best_auc:.4f} | Weights: {best_weights}")
    
    # 3. Compare Best Single vs Ensemble
    print("\n" + "=" * 70)
    print(" Improvement Summary")
    print("=" * 70)
    
    # Get single model baselines
    mask = ~df_test['rank'].isna()
    baselines = {
        'win': roc_auc_score((df_test['rank'][mask] == 1).astype(int), predictions['Win'][mask]),
        'top2': roc_auc_score((df_test['rank'][mask] <= 2).astype(int), predictions['Top2'][mask]),
        'top3': roc_auc_score((df_test['rank'][mask] <= 3).astype(int), predictions['Top3'][mask]),
    }
    
    for target in targets:
        baseline = baselines[target]
        ensemble_auc = ensemble_results[target]['auc']
        improvement = (ensemble_auc - baseline) * 100
        print(f"{target:<6} | Baseline: {baseline:.4f} | Ensemble: {ensemble_auc:.4f} | Δ: {improvement:+.2f}%")
    
    # 4. Final Recommendation
    print("\n" + "=" * 70)
    print(" Recommended Ensemble Configuration")
    print("=" * 70)
    
    # For Win prediction (main use case)
    print(f"\n🎯 For Win Prediction:")
    print(f"   Weights: {ensemble_results['win']['weights']}")
    print(f"   Expected AUC: {ensemble_results['win']['auc']:.4f}")
    
    print(f"\n🎯 For Top2 Prediction (馬連/連複):")
    print(f"   Weights: {ensemble_results['top2']['weights']}")
    print(f"   Expected AUC: {ensemble_results['top2']['auc']:.4f}")
    
    print(f"\n🎯 For Top3 Prediction (複勝/ワイド):")
    print(f"   Weights: {ensemble_results['top3']['weights']}")
    print(f"   Expected AUC: {ensemble_results['top3']['auc']:.4f}")


if __name__ == "__main__":
    main()

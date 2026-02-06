"""
SHAP Analysis for Model Interpretability
=========================================
Analyzes feature importance using SHAP values to understand:
1. Which features contribute most to predictions
2. How each feature affects predictions (direction and magnitude)
3. Potential data leakage or suspicious patterns

Usage:
  python scripts/experiments/shap_analysis.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
MODEL_PATH = "models/experiments/exp_lambdarank/model.pkl"
OUTPUT_DIR = "reports/shap_analysis"


def load_data():
    """Load and prepare data"""
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    targets = pd.read_parquet(TARGET_PATH)
    
    df['race_id'] = df['race_id'].astype(str)
    targets['race_id'] = targets['race_id'].astype(str)
    
    df = df.merge(targets[['race_id', 'horse_number', 'rank']], 
                  on=['race_id', 'horse_number'], how='left')
    df['date'] = pd.to_datetime(df['date'])
    df['is_win'] = (df['rank'] == 1).astype(int)
    
    return df


def run_shap_analysis():
    """Run comprehensive SHAP analysis"""
    logger.info("=" * 60)
    logger.info("SHAP Analysis for Model Interpretability")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model and data
    model = joblib.load(MODEL_PATH)
    feature_names = model.feature_name()
    logger.info(f"Model has {len(feature_names)} features")
    
    df = load_data()
    df_test = df[df['date'].dt.year == 2024].copy()
    logger.info(f"Test set (2024): {len(df_test)} records")
    
    # Prepare features
    X_test = df_test[feature_names].copy()
    
    for c in X_test.columns:
        if X_test[c].dtype == 'object' or X_test[c].dtype.name == 'category':
            X_test[c] = X_test[c].astype('category').cat.codes
        X_test[c] = X_test[c].fillna(-999)
    
    X_test = X_test.astype(np.float64)
    
    # Sample for SHAP (too slow on full dataset)
    sample_size = min(5000, len(X_test))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[sample_idx]
    
    logger.info(f"SHAP analysis on {sample_size} samples...")
    
    # ========================================
    # 1. Calculate SHAP values
    # ========================================
    logger.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample.values)
    
    # ========================================
    # 2. Feature Importance (Mean |SHAP|)
    # ========================================
    logger.info("Calculating feature importance...")
    
    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_importance
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 70)
    print(" Top 30 Features by SHAP Importance")
    print("=" * 70)
    print(f"\n{'Rank':<6} {'Feature':<45} {'Importance':<15}")
    print("-" * 70)
    for i, (_, row) in enumerate(importance_df.head(30).iterrows(), 1):
        print(f"{i:<6} {row['feature']:<45} {row['importance']:<15.4f}")
    
    # Save to file
    importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
    
    # ========================================
    # 3. Feature Categories Analysis
    # ========================================
    print("\n" + "=" * 70)
    print(" Feature Categories Summary")
    print("=" * 70)
    
    # Categorize features
    categories = {
        'base': ['age', 'sex', 'horse_number', 'weight', 'distance', 'grade'],
        'history': ['win_rate', 'top3_rate', 'avg_rank', 'n_races', 'last_'],
        'jockey': ['jockey_'],
        'trainer': ['trainer_'],
        'pace': ['pace_', 'last_3f', 'early_', 'late_'],
        'bloodline': ['sire_', 'mare_', 'bms_', 'bloodline'],
        'elo': ['elo_', 'rating_'],
        'odds': ['odds_', 'popularity'],
        'aptitude': ['aptitude_', 'surface_', 'distance_', 'course_'],
        'form': ['form_', 'momentum', 'trend'],
        'lag': ['lag1_', 'lag2_', 'lag3_', 'lag4_', 'lag5_'],
        'race': ['race_', 'field_', 'nige_', 'R_']
    }
    
    category_importance = {}
    for cat, prefixes in categories.items():
        cat_features = [f for f in feature_names 
                        if any(p in f.lower() for p in [p.lower() for p in prefixes])]
        cat_imp = importance_df[importance_df['feature'].isin(cat_features)]['importance'].sum()
        category_importance[cat] = {
            'count': len(cat_features),
            'total_importance': cat_imp,
            'avg_importance': cat_imp / len(cat_features) if cat_features else 0
        }
    
    cat_df = pd.DataFrame(category_importance).T
    cat_df = cat_df.sort_values('total_importance', ascending=False)
    
    print(f"\n{'Category':<15} {'Features':<10} {'Total Imp':<12} {'Avg Imp':<12}")
    print("-" * 50)
    for cat, row in cat_df.iterrows():
        print(f"{cat:<15} {int(row['count']):<10} {row['total_importance']:<12.3f} {row['avg_importance']:<12.4f}")
    
    # ========================================
    # 4. Low Importance Features
    # ========================================
    print("\n" + "=" * 70)
    print(" Low Importance Features (Candidates for Removal)")
    print("=" * 70)
    
    low_importance = importance_df[importance_df['importance'] < 0.001]
    print(f"\nFeatures with importance < 0.001: {len(low_importance)}")
    
    if len(low_importance) > 0:
        print("\nTop 20 lowest importance features:")
        for i, (_, row) in enumerate(low_importance.tail(20).iterrows(), 1):
            print(f"  {row['feature']}: {row['importance']:.6f}")
    
    # ========================================
    # 5. Potential Leakage Detection
    # ========================================
    print("\n" + "=" * 70)
    print(" Potential Leakage/Suspicious Patterns")
    print("=" * 70)
    
    # Check for suspiciously high importance on specific features
    suspicious_keywords = ['rank', 'is_win', 'is_top', 'result', 'payout', 'odds_final']
    suspicious = importance_df[
        importance_df['feature'].str.lower().str.contains('|'.join(suspicious_keywords))
    ]
    
    if len(suspicious) > 0:
        print("\n⚠️ Features with potentially leaky keywords:")
        for _, row in suspicious.iterrows():
            print(f"  {row['feature']}: importance = {row['importance']:.4f}")
    else:
        print("\n✅ No obvious leaky features detected")
    
    # Check top 10 for any suspiciously high values
    top_features = importance_df.head(10)
    print("\n📊 Top 10 features (check for unexpected patterns):")
    for _, row in top_features.iterrows():
        flag = "⚠️" if row['importance'] > 0.05 else "✓"
        print(f"  {flag} {row['feature']}: {row['importance']:.4f}")
    
    # ========================================
    # 6. Save SHAP Summary Plot
    # ========================================
    logger.info("Generating SHAP summary plot...")
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                      max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved SHAP summary plot to {OUTPUT_DIR}/shap_summary.png")
    
    # ========================================
    # 7. Feature Correlation with Target
    # ========================================
    print("\n" + "=" * 70)
    print(" Feature-Target Correlation Analysis")
    print("=" * 70)
    
    # Get direction of impact for top features
    print("\nDirection of impact for top features:")
    mean_shap = shap_values.mean(axis=0)
    
    for feat in importance_df.head(15)['feature']:
        idx = list(feature_names).index(feat)
        direction = "↑" if mean_shap[idx] > 0 else "↓"
        print(f"  {feat}: mean SHAP = {mean_shap[idx]:.4f} {direction}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print(" Summary & Recommendations")
    print("=" * 70)
    
    print(f"""
📊 Analysis Complete:
   - Total features: {len(feature_names)}
   - Low importance features (< 0.001): {len(low_importance)}
   - Output saved to: {OUTPUT_DIR}/

🎯 Recommendations:
   1. Consider removing {len(low_importance)} low-importance features
   2. Focus on top feature categories: {', '.join(cat_df.head(3).index.tolist())}
   3. Review SHAP summary plot for feature behavior
""")


if __name__ == "__main__":
    run_shap_analysis()

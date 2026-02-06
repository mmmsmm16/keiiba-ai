
import pandas as pd
import numpy as np
import logging
import sys
import os
import pickle
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.runtime.model_wrapper import ModelWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CheckImportance")

def check_importance():
    model_path = "models/experiments/exp_20251222_040356/model.pkl"
    logger.info(f"Loading model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        wrapper = pickle.load(f)
        
    model = wrapper # Assume it's the model itself if not wrapper
    if hasattr(wrapper, 'models'):
        model = wrapper.models[0]
    
    # Check if LightGBM or CatBoost
    # LightGBM has feature_importance()
    if hasattr(model, 'feature_importance'):
        logger.info(f"Model type: LightGBM ({type(model)})")
        imp = model.feature_importance(importance_type='gain')
        names = model.feature_name()
    elif hasattr(model, 'get_feature_importance'):
        logger.info(f"Model type: CatBoost ({type(model)})")
        imp = model.get_feature_importance()
        names = model.feature_names_
    else:
        logger.error(f"Unknown model type: {type(model)}")
        return

    df_imp = pd.DataFrame({'feature': names, 'importance': imp})
    df_imp = df_imp.sort_values('importance', ascending=False)
    
    logger.info("=== Top 20 Features ===")
    logger.info("\n" + df_imp.head(20).to_string())
    
    # Check for suspicious names
    suspicious = ['rank', 'is_top3', 'is_win', 'time', 'score', 'odds_10min'] # odds_10min is allowed?
    # odds_10min is allowed if not dropped? But dataset drops it?
    # Dataset drops 'odds'.
    
    for s in suspicious:
        if s in df_imp['feature'].values:
            logger.warning(f"⚠️ SUSPICIOUS FEATURE FOUND: {s}")
            
    # Check dominance
    top_share = df_imp.iloc[0]['importance'] / df_imp['importance'].sum()
    logger.info(f"Top Feature Share: {top_share:.2%}")
    if top_share > 0.5:
         logger.warning("⚠️ DOMINANT FEATURE DETECTED (>50%)")

if __name__ == "__main__":
    check_importance()

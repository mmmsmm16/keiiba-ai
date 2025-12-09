from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
import os
import sys
from typing import Optional, Dict, Any

# Ensure project root is in path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.inference.loader import InferenceDataLoader
from src.inference.preprocessor import InferencePreprocessor
from src.model.ensemble import EnsembleModel

router = APIRouter()

# Global model cache to avoid reloading heavy models
MODEL_CACHE: Dict[str, Any] = {}
BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../../models')

def get_model(version: str):
    """
    Load and cache the specified model version.
    Supports specific versions like 'v7', 'v5'.
    Defaults to EnsembleModel.
    """
    global MODEL_CACHE
    
    if version in MODEL_CACHE:
        return MODEL_CACHE[version]
    
    print(f"Loading model version: {version}...")
    try:
        model = EnsembleModel()
        
        # Determine filename based on version ID
        filename = f'ensemble_{version}.pkl'
        
        if version == 'v4_2025':
            filename = 'ensemble_v4_2025.pkl'
        elif version == 'v5':
            filename = 'ensemble_v5.pkl'
            
        model_path = os.path.join(BASE_MODEL_DIR, filename)
        
        if not os.path.exists(model_path):
             # Fallback logic
             if version == 'latest':
                fallback_path = os.path.join(BASE_MODEL_DIR, 'ensemble_model.pkl')
                if os.path.exists(fallback_path):
                    model_path = fallback_path
                else:
                    raise FileNotFoundError(f"Latest model not found")
             else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model.load_model(model_path)
        MODEL_CACHE[version] = model
        print(f"Model {version} loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model {version}: {e}")
        raise e

@router.get("/races/{race_id}/predictions", tags=["predictions"])
def get_race_predictions(
    race_id: str, 
    model_id: str = Query("v5", description="Model version (v5=JRA Only, v4_2025=JRA+Combine)")
):
    """
    Get AI predictions using the specified model version.
    """
    try:
        # 1. Load Data
        loader = InferenceDataLoader()
        df = loader.load(race_ids=[race_id])
        
        if df.empty:
            # Check if strictly empty or just no rows
            return {"race_id": race_id, "predictions": [], "message": "Race not found."}
        
        df_race = df[df['race_id'] == race_id].copy()
        
        # 2. Preprocess
        preprocessor = InferencePreprocessor()
        # Note: preprocess expects raw_df and returns X, ids
        try:
             X, ids = preprocessor.preprocess(df_race)
        except Exception as e:
             import traceback
             return {
                 "race_id": race_id, 
                 "predictions": [], 
                 "message": f"Preprocessing error: {str(e)}",
                 "debug_trace": traceback.format_exc()
             }
        
        if X.empty:
             return {"race_id": race_id, "predictions": [], "message": "Preprocessing failed (no features generated)."}

        # Combine for easy handling
        processed_df = pd.concat([ids, X], axis=1)
        
        # Remove duplicate columns if any (e.g. horse_number in X and ids)
        processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
        
        # 3. Load Model & Predict
        model = get_model(model_id)
        
        scores = model.predict(X)
        
        # 4. Format Results
        from scipy.special import softmax
        
        # Assign scores back to processed_df
        processed_df['score'] = scores
        
        # Calculate Probabilities (Softmax within the race)
        processed_df['prob'] = softmax(processed_df['score'].values)
        
        # Prepare response list
        predictions = []
        
        # Ensure types for merge
        df_race['horse_number'] = df_race['horse_number'].astype(int)
        processed_df['horse_number'] = processed_df['horse_number'].astype(int)
        
        # Merge results with display info
        result_df = pd.merge(
            processed_df[['horse_number', 'score', 'prob']],
            df_race[['horse_number', 'horse_name', 'popularity', 'odds', 'frame_number']],
            on='horse_number',
            how='inner'
        )
        
        # Helper for types
        def safe_float(v):
            if pd.isna(v): return 0.0
            return float(v)

        result_df = result_df.sort_values('score', ascending=False)
        
        for i, row in result_df.iterrows():
            score = float(row['score'])
            prob = float(row['prob'])
            odds = safe_float(row['odds'])
            
            # Expected Value = Probability * Odds
            expected_value = prob * odds
            
            predictions.append({
                "horse_number": int(row['horse_number']),
                "frame_number": int(row['frame_number']) if not pd.isna(row['frame_number']) else 0,
                "horse_name": str(row['horse_name']),
                "score": round(score, 4), # Raw score
                "probability": round(prob * 100, 1), # %
                "expected_value": round(expected_value, 2),
                "predicted_rank": 0, # To be filled
                "odds": odds,
                "popularity": safe_float(row['popularity'])
            })
            
        # Assign ranks
        for i, p in enumerate(predictions):
            p['predicted_rank'] = i + 1
            
        return {
            "race_id": race_id,
            "predictions": predictions,
            "model_version": f"Ensemble {model_id}",
            "feature_count": X.shape[1],
            "timestamp": pd.Timestamp.now().isoformat()
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Prediction failed: {error_details}")
        # Return 200 with error message for debugging purposes
        return {
            "race_id": race_id,
            "predictions": [],
            "message": f"Server Error: {str(e)}",
            "debug_trace": error_details
        }

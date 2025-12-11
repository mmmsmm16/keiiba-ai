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
PREDICTIONS_CACHE: Dict[str, pd.DataFrame] = {}
PREPROCESSED_CACHE: Dict[str, pd.DataFrame] = {}
BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../../models')
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), '../../../experiments')

def load_cached_predictions(model_id: str) -> pd.DataFrame:
    """Load cached predictions from parquet file."""
    global PREDICTIONS_CACHE
    
    if model_id in PREDICTIONS_CACHE:
        return PREDICTIONS_CACHE[model_id]
    
    # Map model_id to prediction file
    pred_path = None
    if model_id == 'v12':
        pred_path = os.path.join(EXPERIMENTS_DIR, 'v12_tabnet_revival', 'reports', 'predictions.parquet')
    elif model_id == 'v7':
        pred_path = os.path.join(EXPERIMENTS_DIR, 'v7_ensemble_full', 'reports', 'predictions.parquet')
    elif model_id == 'v4_2025':
        pred_path = os.path.join(EXPERIMENTS_DIR, 'predictions_ensemble_v4_2025.parquet')
    
    if pred_path and os.path.exists(pred_path):
        print(f"Loading cached predictions from: {pred_path}")
        df = pd.read_parquet(pred_path)
        PREDICTIONS_CACHE[model_id] = df
        return df
    
    return pd.DataFrame()

def load_preprocessed_data(model_id: str) -> pd.DataFrame:
    """Load preprocessed data with all features for real-time prediction."""
    global PREPROCESSED_CACHE
    
    if model_id in PREPROCESSED_CACHE:
        return PREPROCESSED_CACHE[model_id]
    
    # Map model_id to preprocessed data
    data_path = None
    if model_id == 'v12':
        data_path = os.path.join(EXPERIMENTS_DIR, 'v12_tabnet_revival', 'data', 'preprocessed_data.parquet')
    elif model_id == 'v7':
        data_path = os.path.join(EXPERIMENTS_DIR, 'v7_ensemble_full', 'data', 'preprocessed_data.parquet')
    
    if data_path and os.path.exists(data_path):
        print(f"Loading preprocessed data from: {data_path}")
        df = pd.read_parquet(data_path)
        PREPROCESSED_CACHE[model_id] = df
        return df
    
    return pd.DataFrame()

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
        
        if version == 'v12':
            # v12 is in experiments directory
            model_path = os.path.join(EXPERIMENTS_DIR, 'v12_tabnet_revival', 'models', 'ensemble.pkl')
        elif version == 'v4_2025':
            filename = 'ensemble_v4_2025.pkl'
            model_path = os.path.join(BASE_MODEL_DIR, filename)
        elif version == 'v5':
            filename = 'ensemble_v5.pkl'
            model_path = os.path.join(BASE_MODEL_DIR, filename)
        else:
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
    model_id: str = Query("v12", description="Model version (v12=High ROI, v7=2025 Best, v5=JRA Only)")
):
    """
    Get AI predictions using the specified model version.
    For past races with cached predictions, uses pre-computed data.
    For future races, performs real-time prediction.
    """
    try:
        from scipy.special import softmax
        
        # 1. Check for cached predictions first (for past races)
        cached_df = load_cached_predictions(model_id)
        
        if not cached_df.empty and race_id in cached_df['race_id'].values:
            # Use cached predictions
            race_preds = cached_df[cached_df['race_id'] == race_id].copy()
            race_preds = race_preds.sort_values('score', ascending=False)
            
            predictions = []
            for i, row in race_preds.iterrows():
                score = float(row['score']) if pd.notna(row.get('score')) else 0
                prob = softmax(race_preds['score'].values)[list(race_preds.index).index(i)]
                odds = float(row['odds']) if pd.notna(row.get('odds')) else 1.0
                expected_value = prob * odds
                
                predictions.append({
                    "horse_number": int(row['horse_number']),
                    "frame_number": int(row.get('frame_number', 0)) if pd.notna(row.get('frame_number')) else 0,
                    "horse_name": str(row.get('horse_name', '-')),
                    "score": round(score, 4),
                    "probability": round(prob * 100, 1),
                    "expected_value": round(expected_value, 2),
                    "predicted_rank": 0,
                    "odds": odds,
                    "popularity": float(row.get('popularity', 0)) if pd.notna(row.get('popularity')) else 0,
                    "actual_rank": int(row.get('rank', 0)) if pd.notna(row.get('rank')) else None
                })
            
            for i, p in enumerate(predictions):
                p['predicted_rank'] = i + 1
            
            return {
                "race_id": race_id,
                "predictions": predictions,
                "model_version": f"Ensemble {model_id} (cached)",
                "source": "cached",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        # 2. Check for preprocessed data (for races in training data)
        preprocessed_df = load_preprocessed_data(model_id)
        
        if not preprocessed_df.empty and race_id in preprocessed_df['race_id'].values:
            # Use preprocessed features and run prediction
            race_data = preprocessed_df[preprocessed_df['race_id'] == race_id].copy()
            
            # Get model and required features
            model = get_model(model_id)
            
            # Get feature names from the model
            required_features = model.lgbm.model.feature_name()
            
            # Prepare X
            available_features = [f for f in required_features if f in race_data.columns]
            X = race_data[available_features].copy()
            
            # Run prediction
            scores = model.predict(X)
            
            race_data['score'] = scores
            race_data['prob'] = softmax(race_data['score'].values)
            race_data = race_data.sort_values('score', ascending=False)
            
            predictions = []
            for i, (idx, row) in enumerate(race_data.iterrows()):
                score = float(row['score'])
                prob = float(row['prob'])
                odds = float(row['odds']) if pd.notna(row.get('odds')) else 1.0
                expected_value = prob * odds
                
                predictions.append({
                    "horse_number": int(row['horse_number']),
                    "frame_number": int(row.get('frame_number', 0)) if pd.notna(row.get('frame_number')) else 0,
                    "horse_name": str(row.get('horse_name', '-')),
                    "score": round(score, 4),
                    "probability": round(prob * 100, 1),
                    "expected_value": round(expected_value, 2),
                    "predicted_rank": i + 1,
                    "odds": odds,
                    "popularity": float(row.get('popularity', 0)) if pd.notna(row.get('popularity')) else 0,
                    "actual_rank": int(row.get('rank', 0)) if pd.notna(row.get('rank')) else None
                })
            
            return {
                "race_id": race_id,
                "predictions": predictions,
                "model_version": f"Ensemble {model_id} (preprocessed)",
                "source": "preprocessed",
                "feature_count": len(available_features),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        # 3. No preprocessed data - try real-time prediction (may not have all features)
        loader = InferenceDataLoader()
        df = loader.load(race_ids=[race_id])
        
        if df.empty:
            return {"race_id": race_id, "predictions": [], "message": "Race not found."}
        
        df_race = df[df['race_id'] == race_id].copy()
        
        # Preprocess
        preprocessor = InferencePreprocessor()
        try:
             X, ids = preprocessor.preprocess(df_race)
        except Exception as e:
             import traceback
             return {
                 "race_id": race_id, 
                 "predictions": [], 
                 "message": f"Preprocessing error: {str(e)}. This race may require the full preprocessed dataset.",
                 "debug_trace": traceback.format_exc()
             }
        
        if X.empty:
             return {"race_id": race_id, "predictions": [], "message": "Preprocessing failed."}

        processed_df = pd.concat([ids, X], axis=1)
        processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
        
        # Load Model & Predict
        model = get_model(model_id)
        scores = model.predict(X)
        
        processed_df['score'] = scores
        processed_df['prob'] = softmax(processed_df['score'].values)
        
        predictions = []
        df_race['horse_number'] = df_race['horse_number'].astype(int)
        processed_df['horse_number'] = processed_df['horse_number'].astype(int)
        
        result_df = pd.merge(
            processed_df[['horse_number', 'score', 'prob']],
            df_race[['horse_number', 'horse_name', 'popularity', 'odds', 'frame_number']],
            on='horse_number',
            how='inner'
        )
        
        def safe_float(v):
            if pd.isna(v): return 0.0
            return float(v)

        result_df = result_df.sort_values('score', ascending=False)
        
        for i, row in result_df.iterrows():
            score = float(row['score'])
            prob = float(row['prob'])
            odds = safe_float(row['odds'])
            expected_value = prob * odds
            
            predictions.append({
                "horse_number": int(row['horse_number']),
                "frame_number": int(row['frame_number']) if not pd.isna(row['frame_number']) else 0,
                "horse_name": str(row['horse_name']),
                "score": round(score, 4),
                "probability": round(prob * 100, 1),
                "expected_value": round(expected_value, 2),
                "predicted_rank": 0,
                "odds": odds,
                "popularity": safe_float(row['popularity'])
            })
            
        for i, p in enumerate(predictions):
            p['predicted_rank'] = i + 1
            
        return {
            "race_id": race_id,
            "predictions": predictions,
            "model_version": f"Ensemble {model_id}",
            "source": "realtime",
            "feature_count": X.shape[1],
            "timestamp": pd.Timestamp.now().isoformat()
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Prediction failed: {error_details}")
        return {
            "race_id": race_id,
            "predictions": [],
            "message": f"Server Error: {str(e)}",
            "debug_trace": error_details
        }


@router.get("/daily-roi", tags=["predictions"])
def get_daily_roi(date: str = Query(..., description="Date in YYYY-MM-DD format"), model_id: str = "v12"):
    """
    Calculate ROI for Option C strategy on all races for a given date.
    Returns overall ROI and ROI per venue.
    """
    from itertools import permutations
    
    try:
        # Load cached predictions
        predictions_df = load_cached_predictions(model_id)
        if predictions_df.empty:
            return {"error": "No predictions available", "date": date}
        
        # Filter by date
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        target_date = pd.to_datetime(date)
        df = predictions_df[predictions_df['date'].dt.date == target_date.date()].copy()
        
        if df.empty:
            return {"error": "No races found for this date", "date": date}
        
        # Load payout data - try multiple possible paths
        payout_df = None
        payout_paths = [
            os.path.join(EXPERIMENTS_DIR, 'payouts_2024_2025.parquet'),
            os.path.join(EXPERIMENTS_DIR, 'payouts_2024.parquet'),
            os.path.join(EXPERIMENTS_DIR, 'v7_ensemble_full', 'data', 'payout_data.parquet'),
        ]
        for payout_path in payout_paths:
            if os.path.exists(payout_path):
                payout_df = pd.read_parquet(payout_path)
                print(f"Loaded payout data from: {payout_path}")
                break
        
        if payout_df is None:
            return {"error": "Payout data not available", "date": date}
        
        # Build payout map
        payout_map = {}
        for _, row in payout_df.iterrows():
            rid = str(row.get('race_id', ''))
            if not rid:
                continue
            if rid not in payout_map:
                payout_map[rid] = {'tansho': {}, 'sanrentan': {}, 'sanrenpuku': {}}
            
            # Tansho
            for k in range(1, 4):
                comb = row.get(f'haraimodoshi_tansho_{k}a')
                pay = row.get(f'haraimodoshi_tansho_{k}b')
                if comb and pay:
                    try:
                        payout_map[rid]['tansho'][str(int(float(comb))).zfill(2)] = int(float(pay))
                    except:
                        pass
            
            # Sanrentan
            for k in range(1, 7):
                comb = row.get(f'haraimodoshi_sanrentan_{k}a')
                pay = row.get(f'haraimodoshi_sanrentan_{k}b')
                if comb and pay:
                    try:
                        payout_map[rid]['sanrentan'][str(comb).strip()] = int(float(pay))
                    except:
                        pass
            
            # Sanrenpuku (三連複)
            for k in range(1, 4):
                comb = row.get(f'haraimodoshi_sanrenpuku_{k}a')
                pay = row.get(f'haraimodoshi_sanrenpuku_{k}b')
                if comb and pay:
                    try:
                        payout_map[rid]['sanrenpuku'][str(comb).strip()] = int(float(pay))
                    except:
                        pass
        
        # Calculate pred_rank
        df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
        
        # Get venue from race_id (position 4-5)
        df['venue'] = df['race_id'].astype(str).str[4:6]
        
        # Stats tracking
        stats_total = {'cost': 0, 'return': 0, 'races': 0, 'hits': 0}
        stats_by_venue = {}
        stats_by_race = {}  # Per race stats
        
        for race_id, group in df.groupby('race_id'):
            race_id_str = str(race_id)
            venue = race_id_str[4:6] if len(race_id_str) >= 6 else "00"
            
            if venue not in stats_by_venue:
                stats_by_venue[venue] = {'cost': 0, 'return': 0, 'races': 0, 'hits': 0}
            
            if race_id_str not in payout_map:
                continue
            
            sorted_group = group.sort_values('pred_rank')
            if len(sorted_group) < 5:
                continue
            
            top1 = sorted_group.iloc[0]
            top1_pop = int(top1.get('popularity', 99)) if pd.notna(top1.get('popularity')) else 99
            
            # Calculate Expected Value (EV) for v12 strategy
            # Use expected_value column if available, otherwise calculate from probability * odds
            if 'expected_value' in sorted_group.columns and pd.notna(top1.get('expected_value')):
                top1_ev = float(top1.get('expected_value', 0))
            else:
                # Calculate probability using softmax and multiply by odds
                from scipy.special import softmax
                probs = softmax(sorted_group['score'].values)
                top1_prob = probs[0]
                top1_odds = float(top1.get('odds', 1)) if pd.notna(top1.get('odds')) else 1.0
                top1_ev = top1_prob * top1_odds
            
            # Initialize
            race_cost = 0
            race_return = 0
            hit = False
            bet_type = "見送り"
            
            # v12 EV-based Strategy
            axis = int(sorted_group.iloc[0]['horse_number'])
            opps = [int(sorted_group.iloc[i]['horse_number']) for i in range(1, min(5, len(sorted_group)))]
            
            if top1_ev >= 1.2:
                # High Value: 三連複 Top1→3頭 (ROI 142.7%)
                bet_type = "三連複"
                # 三連複 1頭軸流し (3点)
                from itertools import combinations
                tickets = list(combinations([axis] + opps[:3], 3))
                tickets = [t for t in tickets if axis in t]  # 軸馬を含むもののみ
                race_cost = len(tickets) * 100
                
                for t in tickets:
                    sorted_t = tuple(sorted(t))
                    key = f"{sorted_t[0]:02}{sorted_t[1]:02}{sorted_t[2]:02}"
                    if key in payout_map[race_id_str].get('sanrenpuku', {}):
                        race_return += payout_map[race_id_str]['sanrenpuku'][key]
                        hit = True
                        
            elif top1_ev >= 0.8:
                # Mid Value: 三連単 Top1→3頭 (ROI 116.5%)
                bet_type = "三連単"
                # 三連単 1着固定流し (6点)
                tickets = [(axis, o1, o2) for o1, o2 in permutations(opps[:3], 2)]
                race_cost = len(tickets) * 100
                
                for t in tickets:
                    key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                    if key in payout_map[race_id_str].get('sanrentan', {}):
                        race_return += payout_map[race_id_str]['sanrentan'][key]
                        hit = True
            else:
                # Low Value: 見送り (EV < 0.8)
                race_cost = 0
                race_return = 0
                bet_type = "見送り"
                hit = False
            
            # Store per-race stats
            stats_by_race[race_id_str] = {
                "cost": race_cost,
                "return": int(race_return),
                "hit": hit,
                "bet_type": bet_type
            }
            
            stats_total['cost'] += race_cost
            stats_total['return'] += race_return
            stats_total['races'] += 1
            if hit:
                stats_total['hits'] += 1
            
            stats_by_venue[venue]['cost'] += race_cost
            stats_by_venue[venue]['return'] += race_return
            stats_by_venue[venue]['races'] += 1
            if hit:
                stats_by_venue[venue]['hits'] += 1
        
        # Calculate ROI
        total_roi = (stats_total['return'] / stats_total['cost'] * 100) if stats_total['cost'] > 0 else 0
        hit_rate = (stats_total['hits'] / stats_total['races'] * 100) if stats_total['races'] > 0 else 0
        
        venue_results = []
        for venue, stats in stats_by_venue.items():
            roi = (stats['return'] / stats['cost'] * 100) if stats['cost'] > 0 else 0
            hr = (stats['hits'] / stats['races'] * 100) if stats['races'] > 0 else 0
            venue_results.append({
                "venue": venue,
                "races": stats['races'],
                "cost": stats['cost'],
                "return": int(stats['return']),
                "profit": int(stats['return'] - stats['cost']),
                "roi": round(roi, 1),
                "hit_rate": round(hr, 1)
            })
        
        return {
            "date": date,
            "model_id": model_id,
            "total": {
                "races": stats_total['races'],
                "cost": stats_total['cost'],
                "return": int(stats_total['return']),
                "profit": int(stats_total['return'] - stats_total['cost']),
                "roi": round(total_roi, 1),
                "hit_rate": round(hit_rate, 1)
            },
            "by_venue": venue_results,
            "by_race": stats_by_race
        }
        
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}

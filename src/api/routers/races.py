from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from src.inference.loader import InferenceDataLoader
import pandas as pd
from datetime import datetime

router = APIRouter()

@router.get("/races",  tags=["races"])
def get_races(date: str = Query(..., description="Target date in YYYY-MM-DD format"), limit: int = 100):
    """
    Get list of races for a specific date.
    """
    loader = InferenceDataLoader()
    
    # Convert YYYY-MM-DD to YYYYMMDD
    try:
        target_date = date.replace("-", "")
        # Basic validation
        if len(target_date) != 8 or not target_date.isdigit():
             raise ValueError("Invalid date format")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    try:
        df = loader.load_race_list(target_date)
        if df.empty:
            return {"races": [], "message": "No races found for this date."}
        
        # Helper to convert to list of dicts
        # Adjust types for JSON serialization
        records = df.to_dict(orient='records')
        
        # Clean up types if necessary (e.g. numpy int to int)
        # Pandas to_dict usually handles basic types well, but let's ensure.
        return {"races": records, "count": len(records), "date": date}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/races/{race_id}", tags=["races"])
def get_race_detail(race_id: str):
    """
    Get detailed information for a specific race including horse entries.
    """
    loader = InferenceDataLoader()
    
    try:
        # Extract date from race_id (format: YYYYJJKKNNRR)
        # YYYY = year, JJ = venue, KK = kai, NN = nichime, RR = race number
        # We need the full date which requires looking up the race
        
        # Load race data for this specific race_id
        df = loader.load(race_ids=[race_id])
        
        if df.empty:
            return {"race_id": race_id, "entries": [], "message": "Race not found or no entries."}
        
        # Filter to just this race
        df_race = df[df['race_id'] == race_id].copy()
        
        if df_race.empty:
            return {"race_id": race_id, "entries": [], "message": "Race not found."}
        
        # Helper for safe type conversion
        def safe_int(val):
            try:
                if pd.isna(val): return None
                return int(val)
            except:
                return None

        def safe_str(val):
            if pd.isna(val): return None
            return str(val)

        # Select relevant columns for race card display
        columns_to_include = [
            'horse_number', 'frame_number', 'horse_name', 'sex', 'age',
            'jockey_name', 'trainer_name', 'weight', 'weight_diff',
            'odds', 'popularity', 'horse_id'
        ]
        
        # Filter columns that exist
        available_cols = [c for c in columns_to_include if c in df_race.columns]
        df_entries = df_race[available_cols].copy()
        
        # Sort by horse number
        df_entries = df_entries.sort_values('horse_number')
        
        # Convert to records, handling NaN
        # Use simple fillna for entries table
        entries = df_entries.fillna('').to_dict(orient='records')
        
        # Get race metadata from first row
        # Ensure we handle numpy types (int64, etc.) by converting to native python types or None
        row = df_race.iloc[0]
        
        race_info = {
            "race_id": race_id,
            "venue": safe_str(row.get('venue')),
            "race_number": safe_int(row.get('race_number')),
            "title": safe_str(row.get('title')),
            "distance": safe_int(row.get('distance')),
            "track_type": safe_str(row.get('track_type')),
            "track_condition": safe_str(row.get('track_condition')),
        }
        
        return {
            "race": race_info,
            "entries": entries,
            "count": len(entries)
        }
        
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

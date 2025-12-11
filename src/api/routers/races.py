from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from src.inference.loader import InferenceDataLoader
import pandas as pd
from datetime import datetime

router = APIRouter()

@router.get("/races/latest-date", tags=["races"])
def get_latest_race_date():
    """
    Get the most recent date that has race data.
    """
    from sqlalchemy import create_engine, text
    import os
    
    try:
        # Use the same DB settings as InferenceDataLoader
        user = os.environ.get('POSTGRES_USER', 'user')
        password = os.environ.get('POSTGRES_PASSWORD', 'password')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT MAX(TO_DATE(kaisai_nen || kaisai_tsukihi, 'YYYYMMDD')) as latest_date
                FROM jvd_ra
                WHERE kaisai_nen IS NOT NULL
                  AND keibajo_code IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
            """))
            row = result.fetchone()
            
            if row and row[0]:
                latest_date = row[0].strftime('%Y-%m-%d')
                return {"latest_date": latest_date}
            else:
                return {"latest_date": None, "message": "No race data found"}
                
    except Exception as e:
        return {"latest_date": None, "error": str(e)}

@router.get("/races/adjacent-dates", tags=["races"])
def get_adjacent_race_dates(date: str = Query(..., description="Current date in YYYY-MM-DD format")):
    """
    Get the previous and next race dates relative to the given date (JRA only).
    """
    from sqlalchemy import create_engine, text
    import os
    
    try:
        user = os.environ.get('POSTGRES_USER', 'user')
        password = os.environ.get('POSTGRES_PASSWORD', 'password')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(db_url)
        
        target_date = date.replace("-", "")
        
        with engine.connect() as conn:
            # Get previous race date
            prev_result = conn.execute(text(f"""
                SELECT MAX(TO_DATE(kaisai_nen || kaisai_tsukihi, 'YYYYMMDD')) as prev_date
                FROM jvd_ra
                WHERE TO_DATE(kaisai_nen || kaisai_tsukihi, 'YYYYMMDD') < TO_DATE('{target_date}', 'YYYYMMDD')
                  AND keibajo_code IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
            """))
            prev_row = prev_result.fetchone()
            prev_date = prev_row[0].strftime('%Y-%m-%d') if prev_row and prev_row[0] else None
            
            # Get next race date
            next_result = conn.execute(text(f"""
                SELECT MIN(TO_DATE(kaisai_nen || kaisai_tsukihi, 'YYYYMMDD')) as next_date
                FROM jvd_ra
                WHERE TO_DATE(kaisai_nen || kaisai_tsukihi, 'YYYYMMDD') > TO_DATE('{target_date}', 'YYYYMMDD')
                  AND keibajo_code IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
            """))
            next_row = next_result.fetchone()
            next_date = next_row[0].strftime('%Y-%m-%d') if next_row and next_row[0] else None
            
            return {"prev_date": prev_date, "next_date": next_date, "current_date": date}
                
    except Exception as e:
        return {"prev_date": None, "next_date": None, "error": str(e)}

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
        
        # Get date from the row if available
        race_date = None
        if 'date' in row and pd.notna(row['date']):
            try:
                race_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            except:
                pass
        
        race_info = {
            "race_id": race_id,
            "venue": safe_str(row.get('venue')),
            "race_number": safe_int(row.get('race_number')),
            "title": safe_str(row.get('title')),
            "distance": safe_int(row.get('distance')),
            "track_type": safe_str(row.get('track_type')),
            "track_condition": safe_str(row.get('track_condition')),
            "date": race_date,
        }
        
        # Get available venues for this date
        available_venues = []
        if race_date:
            try:
                venue_loader = InferenceDataLoader()
                venue_df = venue_loader.load_race_list(race_date.replace('-', ''))
                available_venues = list(venue_df['venue'].unique()) if not venue_df.empty else []
            except:
                pass
        
        return {
            "race": race_info,
            "entries": entries,
            "count": len(entries),
            "available_venues": available_venues
        }
        
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

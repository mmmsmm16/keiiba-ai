from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import os
import logging
from datetime import datetime

from src.simulation.simulator import BettingSimulator

router = APIRouter()
logger = logging.getLogger(__name__)

# Cache loader to avoid re-init
simulator = BettingSimulator()

class DateRange(BaseModel):
    start: str
    end: str

class Filters(BaseModel):
    surface: Optional[str] = None # 芝, ダート
    place: Optional[List[str]] = None

class Strategy(BaseModel):
    type: str
    name: str
    formation: List[List[int]]

class SimulationRequest(BaseModel):
    model_keys: List[str] = ["v5_2025"] # e.g. ["v5", "v4"]
    year: int = 2025
    filters: Optional[Filters] = None
    strategies: List[Strategy]

@router.post("/run")
async def run_simulation(req: SimulationRequest):
    try:
        results = {}

        for model_key in req.model_keys:
            # Determine File Path
            model_path = ""
            if "v5" in model_key:
                model_path = "reports/predictions_v5_2025.csv"
            elif "v4" in model_key:
                model_path = "reports/predictions_v4_2025.csv"
            else:
                # Default fallback or skip
                logger.warning(f"Unknown model key: {model_key}, skipping")
                continue
            
            if not os.path.exists(model_path):
                 results[model_key] = {"error": "Model file not found"}
                 continue

            df = pd.read_csv(model_path)
            df['race_id'] = df['race_id'].astype(str)
            
            # --- Filtering ---
            
            # Surface Filter
            if req.filters and req.filters.surface and req.filters.surface != "All" and 'surface' in df.columns:
                df = df[df['surface'] == req.filters.surface]

            # Year Filter
            # Assuming 'date' column exists in YYYY-MM-DD or similar
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Filter by Year
                df = df[df['date'].dt.year == req.year]
                # Convert back to string for response consistency
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            else:
                 # If no date, maybe can't filter by year? or rely on race_id?
                 # race_id for JRA usually starts with Year: '2025...'
                 # Let's try to filter by race_id prefix if date missing
                 df = df[df['race_id'].str.startswith(str(req.year))]
            
            if df.empty:
                results[model_key] = {"warning": "No data found for this period"}
                continue

            # Convert Pydantic strategies to dict for simulator
            strategies_dict = [s.dict() for s in req.strategies]
            
            # Run Simulation (Returns detailed stats now)
            sim_result = simulator.run(df, strategies_dict)
            results[model_key] = sim_result
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

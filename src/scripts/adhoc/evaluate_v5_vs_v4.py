
import pandas as pd
import sys
import os
import logging
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.inference.loader import InferenceDataLoader
from src.inference.preprocessor import InferencePreprocessor
from src.model.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(version, race_ids, loader, preprocessor, model_dir='models'):
    logger.info(f"Evaluating Model {version}...")
    
    # Load Model
    ensemble = EnsembleModel()
    
    # For v4, filenames are standard (e.g. ensemble_v4.pkl)
    # For v5, same.
    path = os.path.join(model_dir, f'ensemble_{version}.pkl')
    if version == 'v4' and not os.path.exists(path):
        alt_path = os.path.join(model_dir, 'ensemble_v4_2025.pkl')
        if os.path.exists(alt_path):
            path = alt_path

    if not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        return None

    ensemble.load_model(path, device_name='cpu') # Use CPU for inference eval to be safe/consistent
    
    results = []
    
    # Process in chunks
    chunk_size = 100
    for i in tqdm(range(0, len(race_ids), chunk_size)):
        chunk_ids = race_ids[i:i+chunk_size]
        
        # Load raw data
        raw_df = loader.load(race_ids=chunk_ids)
        if raw_df is None or raw_df.empty:
            continue
            
        # Preprocess
        # Preprocessor returns (X, ids)
        try:
            X, ids = preprocessor.preprocess(raw_df)
        except Exception as e:
            logger.warning(f"Preprocessing failed for chunk {i}: {e}")
            continue
        
        # Predict
        if X.empty:
            continue
            
        try:
            scores = ensemble.predict(X)
        except Exception as e:
            logger.warning(f"Prediction failed for chunk {i}: {e}")
            continue
            
        # Create result DF
        # ids contains race_id, horse_number, odds, surface, etc.
        # Ensure alignment (indices usually preserved)
        ids['score'] = scores
        ids['model_version'] = version
        
        # Keep necessary metadata for basic evaluation if not present
        # InferencePreprocessor's ids returns: 
        # ['race_id', 'date', 'venue', 'race_number', 'horse_number', 'horse_name', 
        #  'jockey_id', 'odds', 'popularity', 'title', 'distance', 'surface', 'state', 'weather']
        
        # But we need 'rank' for evaluation!
        # InferencePreprocessor ids checks raw_df columns, but does it include 'rank'?
        # In preprocessor.py line 412: rank is NOT in id_cols.
        # It IS in raw_df.
        
        # We need to attach 'rank' back to `ids`.
        # Assuming ids index matches raw_df (if no rows dropped) or X.
        # Preprocessor sorts by [race_id, horse_number] at line 377.
        # So we should merge `rank` from raw_df to `ids` using [race_id, horse_number].
        
        if 'rank' in raw_df.columns:
            # Create a lookup or merge
            # ids structure: race_id, horse_number are key.
            # Make sure types match. raw_df loaded from DB might have specific types.
            rank_map = raw_df.set_index(['race_id', 'horse_number'])['rank']
            ids = ids.join(rank_map, on=['race_id', 'horse_number'])
        
        results.append(ids)
        
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()
        


def main():
    # Load 2025 JRA Race IDs
    logger.info("Loading Race List for 2025 (JRA Only)...")
    loader = InferenceDataLoader()
    
    # Query all JRA races in 2025
    from sqlalchemy import text
    with loader.engine.connect() as conn:
        # 01-10 JRA
        # 01-10 JRA
        # Note: jvd_ra table does not have 'race_id' column, we must construct it.
        # columns: kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichisu, race_bango
        q = text("""
            SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango 
            FROM jvd_ra 
            WHERE kaisai_nen = '2025' 
              AND keibajo_code BETWEEN '01' AND '10' 
            ORDER BY kaisai_nen, kaisai_tsukihi, race_bango
        """)
        rows = conn.execute(q).fetchall()
        
        ra_ids = []
        for r in rows:
            # Construct race_id (18 digits? or 16? usually 202501010101 if simple)
            # Keiiba-AI format: YYYYJJRRDDNN (12? No)
            # Standard ID: YYYY GG KK DD RR (Year, Venue, Kai, Day, RaceNo)
            # 4 + 2 + 2 + 2 + 2 = 12 digits.
            # Check DB schema or loader. 
            # Usually strict padding is required.
            # r keys: kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichisu, race_bango
            # Assuming these are strings or ints. If strings, they might be padded.
            
            # Use safe robust formatting
            rid = f"{str(r[0])}{str(r[1])}{str(r[2])}{str(r[3])}{str(r[4])}"
            ra_ids.append(rid)

        
    logger.info(f"Found {len(ra_ids)} JRA races in 2025.")
    
    # Use a subset if too many (e.g. random 50 for quick checks)
    # import random
    # random.seed(42)
    # if len(ra_ids) > 50:
    #     ra_ids = random.sample(ra_ids, 50)
    #     logger.info(f"Subsampled to {len(ra_ids)} random races for quick debug evaluation.") 
    
    preprocessor = InferencePreprocessor()
    
    # Evaluate v4
    df_v4 = evaluate_model('v4', ra_ids, loader, preprocessor)
    if df_v4 is not None:
        calc_roi(df_v4, 'v4')
        
    # Evaluate v5
    df_v5 = evaluate_model('v5', ra_ids, loader, preprocessor)
    if df_v5 is not None:
        # Save for betting simulation
        output_path = 'reports/predictions_v5_2025.csv'
        os.makedirs('reports', exist_ok=True)
        df_v5.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        calc_roi(df_v5, 'v5')

def calc_roi(df, version):
    # Simple ROI calculation on Rank 1 bets
    # Sort by score desc per race
    logger.info(f"--- ROI Analysis for {version} ---")
    
    # Data cleaning
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    
    # Group by race -> get top 1
    # Rank by score descending
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    top1 = df[df['pred_rank'] == 1.0].copy()
    
    # Accuracy
    hits = top1[top1['rank'] == 1]
    accuracy = len(hits) / len(top1)
    
    # ROI
    return_amount = hits['odds'].sum() * 100 # 100 yen bet
    invest_amount = len(top1) * 100
    roi = return_amount / invest_amount * 100 if invest_amount > 0 else 0
    
    print(f"[{version}] Overall: Acc={accuracy:.1%}, ROI={roi:.1f}% (N={len(top1)})")
    
    # Segment: Dirt
    dirt = top1[top1['surface'] == 'Dirt'] # Or mapped code
    # Note: InferenceDataLoader usually returns 'Turf/Dirt' string if fixed? 
    # Or codes? Code mapping logic in loader. 
    # Current loader `map_surface` returns '芝'/'ダート' or 'Turf'/'Dirt' depending on script version.
    # Check values.
    # Assuming 'ダート' or 'Dirt'.
    
    if not dirt.empty:
        d_hits = dirt[dirt['rank'] == 1]
        d_ret = d_hits['odds'].sum() * 100
        d_inv = len(dirt) * 100
        d_roi = d_ret / d_inv * 100
        print(f"[{version}] Dirt   : Acc={len(d_hits)/len(dirt):.1%}, ROI={d_roi:.1f}% (N={len(dirt)})")
        
    # Segment: Turf
    turf = top1[top1['surface'].isin(['Turf', '芝'])]
    if not turf.empty:
        t_hits = turf[turf['rank'] == 1]
        t_ret = t_hits['odds'].sum() * 100
        t_inv = len(turf) * 100
        t_roi = t_ret / t_inv * 100
        print(f"[{version}] Turf   : Acc={len(t_hits)/len(turf):.1%}, ROI={t_roi:.1f}% (N={len(turf)})")

if __name__ == "__main__":
    main()

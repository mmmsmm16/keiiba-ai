import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    
    print("Inspecting jvd_ra columns...")
    try:
        df = pd.read_sql("SELECT * FROM jvd_ra LIMIT 1", loader.engine)
        cols = df.columns.tolist()
        print("Columns:", cols)
        
        # Identify time column
        candidates = [c for c in cols if 'jikan' in c or 'time' in c]
        print("Time candidates:", candidates)
        
        target_col = None
        if 'hassoh_jikan' in cols: target_col = 'hassoh_jikan'
        elif 'hasso_jikan' in cols: target_col = 'hasso_jikan'
        elif candidates: target_col = candidates[0]
        
        if target_col:
            print(f"Using column: {target_col}")
            # Run the check with this column
            q = f"""
            WITH ra AS (
                SELECT 
                    kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
                    CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
                    {target_col} as start_str
                FROM jvd_ra 
                LIMIT 10
            )
            SELECT 
                ra.race_id,
                ra.start_str,
                o1.happyo_tsukihi_jifun,
                o1.race_bango
            FROM ra
            JOIN apd_sokuho_o1 o1 
            ON ra.kaisai_nen = o1.kaisai_nen 
            AND ra.keibajo_code = o1.keibajo_code 
            AND ra.kaisai_kai = o1.kaisai_kai 
            AND ra.kaisai_nichime = o1.kaisai_nichime
            AND ra.race_bango = o1.race_bango
            LIMIT 20
            """
            joined = pd.read_sql(q, loader.engine)
            print("Joined Data Sample:")
            print(joined)
            
            if not joined.empty:
                row = joined.iloc[0]
                start = str(row['start_str'])
                publish = str(row['happyo_tsukihi_jifun'])
                print(f"Start: {start}, Publish: {publish}")
                
        else:
            print("Could not find start time column!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

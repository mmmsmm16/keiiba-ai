import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    print("Fetching state sample...")
    
    q = """
    SELECT 
        CASE WHEN CAST(r.track_code AS INTEGER) BETWEEN 10 AND 22 THEN r.babajotai_code_shiba ELSE r.babajotai_code_dirt END AS state,
        COUNT(*) as cnt
    FROM jvd_ra r
    WHERE r.kaisai_nen = '2024'
    GROUP BY state
    ORDER BY cnt DESC
    LIMIT 20
    """
    
    try:
        df = pd.read_sql(q, loader.engine)
        print("Top values for state:")
        print(df)
        for val in df['state']:
            print(f"'{val}' -> Type: {type(val)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

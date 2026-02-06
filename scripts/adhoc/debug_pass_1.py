import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    print("Fetching pass_1 sample...")
    
    # Check what col_pass is aliased to
    # Usually p.corner_1 etc.
    q = """
    SELECT 
        res.corner_1 as pass_1, 
        res.corner_1,
        tr.baba_jotai_code as going
    FROM jvd_se res
    JOIN jvd_ra tr ON res.record_id = tr.record_id -- Join on Record ID? No, Race ID components
    WHERE res.kaisai_nen = '2024'
    LIMIT 20
    """
    # Join properly
    q_join = """
    SELECT 
        res.corner_1 as pass_1
    FROM jvd_se res
    LIMIT 20
    """
    
    df = pd.read_sql(q_join, loader.engine)
    print("Sample pass_1 values:")
    print(df['pass_1'].unique())
    
    for val in df['pass_1']:
        print(f"'{val}' -> Type: {type(val)}")

if __name__ == "__main__":
    main()

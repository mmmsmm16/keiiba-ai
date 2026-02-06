import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    print("Fetching pass_1 sample with Value='01'...")
    
    q = """
    SELECT 
        res.corner_1 as pass_1,
        COUNT(*) as cnt
    FROM jvd_se res
    WHERE res.kaisai_nen = '2024'
    GROUP BY res.corner_1
    ORDER BY cnt DESC
    LIMIT 10
    """
    
    df = pd.read_sql(q, loader.engine)
    print("Top values for corner_1:")
    print(df)

if __name__ == "__main__":
    main()

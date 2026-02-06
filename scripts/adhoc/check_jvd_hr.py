
import pandas as pd
import sys
import os
from sqlalchemy import create_engine

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    print("Checking jvd_hr (Payouts) count...")
    
    query = "SELECT kaisai_nen, count(*) as cnt FROM jvd_hr GROUP BY kaisai_nen ORDER BY kaisai_nen DESC"
    
    try:
        df = pd.read_sql(query, loader.engine)
        print(df)
        
        # Check sample IDs
        q2 = "SELECT * FROM jvd_hr WHERE kaisai_nen = '2024' LIMIT 1"
        df2 = pd.read_sql(q2, loader.engine)
        if not df2.empty:
            print("Sample 2024 Row:")
            print(df2.iloc[0])
        else:
            print("No 2024 rows in jvd_hr")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

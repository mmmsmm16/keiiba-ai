
import os
import sys
import pandas as pd
from sqlalchemy import text

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.preprocessing.loader import JraVanDataLoader

def inspect_payouts():
    print("Connecting to DB...")
    loader = JraVanDataLoader()
    
    # Check for table name
    tbl = 'apd_sokuho_o1'
    print(f"Using Table: {tbl}")
    

    query = text(f"""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = '{tbl}' 
      AND (column_name LIKE '%odds%' OR column_name LIKE '%fukusho%')
    ORDER BY ordinal_position
    """)
    
    print("\nExecuting Schema Query...")
    try:
        with loader.engine.connect() as conn:
            result = conn.execute(query)
            columns = [row[0] for row in result.fetchall()]
            
        print(f"Columns in {tbl}:")
        for i in range(0, len(columns), 5):
            print(columns[i:i+5])
            
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    inspect_payouts()

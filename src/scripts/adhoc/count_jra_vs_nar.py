
import sys
import os
import pandas as pd
from sqlalchemy import text

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.inference.loader import InferenceDataLoader

def main():
    loader = InferenceDataLoader()
    
    # Query JRA (01-10) vs NAR (Others)
    # Note: keibajo_code is string '01', '02'...
    
    query = text("""
        SELECT 
            CASE 
                WHEN keibajo_code BETWEEN '01' AND '10' THEN 'JRA'
                ELSE 'NAR (Local)'
            END as race_type,
            COUNT(*) as race_count,
            MIN(kaisai_nen) as start_year,
            MAX(kaisai_nen) as end_year
        FROM jvd_ra
        GROUP BY 
            CASE 
                WHEN keibajo_code BETWEEN '01' AND '10' THEN 'JRA'
                ELSE 'NAR (Local)'
            END
    """)
    
    print("Executing query...")
    with loader.engine.connect() as conn:
        result = conn.execute(query)
        print(f"{'Type':<15} | {'Count':<10} | {'Range':<20}")
        print("-" * 50)
        total = 0
        for row in result:
            rtype = row[0]
            count = row[1]
            start = row[2]
            end = row[3]
            print(f"{rtype:<15} | {count:<10} | {start}-{end}")
            total += count
        print("-" * 50)
        print(f"{'Total':<15} | {total:<10}")

if __name__ == "__main__":
    main()

import sys
import pandas as pd
from sqlalchemy import text
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    query = text("""
        SELECT count(*) 
        FROM jvd_ra 
        WHERE kaisai_nen IN ('2024', '2025') 
          AND keibajo_code BETWEEN '01' AND '10' 
          AND kyoso_joken_code = '000'
    """)
    
    with loader.engine.connect() as conn:
        count = conn.execute(query).scalar()
        print(f"JRA Maiden Races in DB (2024-2025): {count}")

    # Check distribution of all Joken codes for JRA
    query_dist = text("""
        SELECT kyoso_joken_code, count(*) 
        FROM jvd_ra 
        WHERE kaisai_nen IN ('2024', '2025') 
          AND keibajo_code BETWEEN '01' AND '10'
        GROUP BY kyoso_joken_code
    """)
    with loader.engine.connect() as conn:
        df = pd.read_sql(query_dist, conn)
        print("\nJRA Joken Code Distribution in DB:")
        print(df.sort_values('count', ascending=False))

if __name__ == "__main__":
    main()

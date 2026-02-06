import sys
import pandas as pd
from sqlalchemy import text
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    # Query race names for code '703'
    query = text("""
        SELECT kaisai_nen, keibajo_code, race_bango, kyosomei_hondai, kyosomei_ryakusho_10, grade_code, kyoso_joken_code 
        FROM jvd_ra 
        WHERE kyoso_joken_code = '703' 
          AND kaisai_nen = '2024'
        LIMIT 10
    """)
    
    with loader.engine.connect() as conn:
        df = pd.read_sql(query, conn)
        print("Sample races for kyoso_joken_code '703':")
        print(df.to_string())
        
        # Also check '005' just to be sure
        query2 = text("""
            SELECT kaisai_nen, keibajo_code, race_bango, kyosomei_hondai, kyosomei_ryakusho_10, grade_code, kyoso_joken_code 
            FROM jvd_ra 
            WHERE kyoso_joken_code = '005' 
              AND kaisai_nen = '2024'
            LIMIT 5
        """)
        df2 = pd.read_sql(query2, conn)
        print("\nUsing code '005' (Should be 1-Win):")
        print(df2.to_string())

if __name__ == "__main__":
    main()

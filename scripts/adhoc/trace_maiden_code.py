import sys
import pandas as pd
from sqlalchemy import text
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    # 2024-01-06 Nakayama (Keibajo 06) Race 1 is usually Maiden
    query = text("""
        SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, race_bango, kyosomei_ryakusho_10, kyoso_joken_code 
        FROM jvd_ra 
        WHERE kaisai_nen = '2024' 
          AND kaisai_tsukihi = '0106' 
          AND keibajo_code = '06'
        ORDER BY race_bango
    """)
    
    with loader.engine.connect() as conn:
        df = pd.read_sql(query, conn)
        print("Races for 2024-01-06 Nakayama (06):")
        print(df.to_string())

if __name__ == "__main__":
    main()

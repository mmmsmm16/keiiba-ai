import os
import pandas as pd
from sqlalchemy import create_engine, text

# Setup DB
user = os.environ.get('POSTGRES_USER', 'postgres')
password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
port = os.environ.get('POSTGRES_PORT', '5433')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(db_url)

print("=== Inspecting NAR Odds Data Format ===\n")

with engine.connect() as conn:
    # Find ANY race in 2025 or 2024
    find_query = text("SELECT * FROM nvd_o1 WHERE kaisai_nen='2025' LIMIT 1")
    row = pd.read_sql(find_query, conn)
    
    if row.empty:
        print("No data in nvd_o1 for 2025. Trying 2024...")
        find_query = text("SELECT * FROM nvd_o1 WHERE kaisai_nen='2024' LIMIT 1")
        row = pd.read_sql(find_query, conn)
        
    if not row.empty:
        df = row
        r = df.iloc[0]
        real_rid = f"{r['kaisai_nen']}{r['keibajo_code']}{r['kaisai_kai']}{r['kaisai_nichime']}{r['race_bango']}"
        
        print(f"--- NVD_O1 Sample (Race: {real_rid}) ---")
        print(f"Venue: {r['keibajo_code']}, Starters: {r['shusso_tosu']}")
        
        print("\n> Tansho (Win) Odds String (first 200 chars):")
        print(r['odds_tansho'][:200] if r['odds_tansho'] else "None")
        
        print("\n> Fukusho (Place) Odds String (first 200 chars):")
        print(r['odds_fukusho'][:200] if r['odds_fukusho'] else "None")
        
        print("\n> Wakuren Odds String (first 200 chars):")
        print(r['odds_wakuren'][:200] if r['odds_wakuren'] else "None")
        
        # Check O2 for same race
        print(f"\n--- NVD_O2 (Umaren) for {real_rid} ---")
        q2 = text(f"""
            SELECT * FROM nvd_o2 
            WHERE kaisai_nen='{r['kaisai_nen']}' 
            AND keibajo_code='{r['keibajo_code']}' 
            AND kaisai_kai='{r['kaisai_kai']}' 
            AND kaisai_nichime='{r['kaisai_nichime']}' 
            AND race_bango='{r['race_bango']}'
        """)
        df2 = pd.read_sql(q2, conn)
        if not df2.empty:
            print("\n> Umaren Odds String (first 300 chars):")
            print(df2['odds_umaren'].iloc[0][:300] if df2['odds_umaren'].iloc[0] else "None")
        else:
            print("No O2 data for this race.")
            
    else:
        print("No data found in nvd_o1 for 2024/2025.")

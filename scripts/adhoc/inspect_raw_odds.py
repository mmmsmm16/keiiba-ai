
import pandas as pd
import os
from sqlalchemy import create_engine

def main():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(db_url)
    
    # Pick a sample race (2025)
    # 202501050611 (Nakayama 11R)
    # race_id construction: YYYY + Place(2) + Kai(2) + Day(2) + No(2)
    # "2025" + "06" (Nakayama) + "01" + "01" + "11"
    # Let's verify a race that exists.
    
    print("--- Searching for a sample race ---")
    q_race = "SELECT * FROM jvd_ra WHERE kaisai_nen='2025' AND keibajo_code IN ('01','02','03','04','05','06','07','08','09','10') LIMIT 1"
    race = pd.read_sql(q_race, engine)
    print(race.iloc[0])
    
    # Construct keys
    k_nen = race.iloc[0]['kaisai_nen']
    k_code = race.iloc[0]['keibajo_code']
    k_kai = race.iloc[0]['kaisai_kai']
    k_nichime = race.iloc[0]['kaisai_nichime']
    k_bango = race.iloc[0]['race_bango']
    
    print(f"\n--- Odds Records for Race {k_nen}-{k_code}-{k_kai}-{k_nichime}-{k_bango} ---")
    
    q_odds = f"""
        SELECT data_kubun, happyo_tsukihi_jifun, kaisai_nen, race_bango, odds_umaren
        FROM apd_sokuho_o2 
        WHERE kaisai_nen='{k_nen}' 
          AND keibajo_code='{k_code}' 
          AND kaisai_kai='{k_kai}' 
          AND kaisai_nichime='{k_nichime}'
          AND race_bango='{k_bango}'
        ORDER BY happyo_tsukihi_jifun ASC
    """
    odds = pd.read_sql(q_odds, engine)
    print(odds)
    
    if not odds.empty:
        print("\n--- Parsing Win Odds for last Record ---")
        last = odds.iloc[-1]
        print(f"Timestamp: {last['happyo_tsukihi_jifun']}, DataKubun: {last['data_kubun']}")
        # Show Start Time
        print(f"Start Time (Hasso): {race.iloc[0]['hasso_jikoku']}")

if __name__ == "__main__":
    main()

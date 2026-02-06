"""
Check if specific date's NAR race data exists in DB
"""
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from sqlalchemy import create_engine, text

# Use same connection method as NarDataLoader
user = os.environ.get('POSTGRES_USER', 'postgres')
password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
port = os.environ.get('POSTGRES_PORT', '5433')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(url)

# Check for 2025-12-18
year = '2025'
mmdd = '1218'
print(f"Checking NAR races for: {year}-{mmdd}")

VENUE_MAP = {
    '36': '大井', '37': '川崎', '38': '浦和', '39': '船橋',
    '42': '笠松', '43': '名古屋', '44': '金沢',
    '45': '園田', '46': '姫路', '47': '福山', '48': '高知', '50': '佐賀',
    '51': '荒尾', '52': '盛岡', '53': '水沢', '54': '門別',
    '55': '北見', '56': '旭川', '57': '帯広'
}

with engine.connect() as conn:
    # Check races count
    result = conn.execute(text(f"""
        SELECT COUNT(*) as cnt FROM nvd_ra 
        WHERE kaisai_nen = '{year}' AND kaisai_tsukihi = '{mmdd}'
    """))
    row = result.fetchone()
    print(f'Total races: {row[0]}')
    
    if row[0] == 0:
        print("\n[WARNING] No races found for this date!")
        print("Please ensure PC-KEIBA data has been loaded for 2025-12-18.")
        # Show latest available date
        result_latest = conn.execute(text("""
            SELECT kaisai_nen, kaisai_tsukihi, COUNT(*) as cnt 
            FROM nvd_ra 
            WHERE kaisai_nen = '2025'
            GROUP BY kaisai_nen, kaisai_tsukihi 
            ORDER BY kaisai_tsukihi DESC 
            LIMIT 5
        """))
        print("\nLatest available dates:")
        for r in result_latest.fetchall():
            print(f"  {r[0]}-{r[1]}: {r[2]} races")
    else:
        # Show venues and race count
        result2 = conn.execute(text(f"""
            SELECT keibajo_code, COUNT(*) as cnt FROM nvd_ra 
            WHERE kaisai_nen = '{year}' AND kaisai_tsukihi = '{mmdd}'
            GROUP BY keibajo_code
        """))
        
        for r in result2.fetchall():
            venue_name = VENUE_MAP.get(r[0], r[0])
            print(f'  {venue_name}: {r[1]} races')

        # Check race entries (shutsuba-hyo) - nvd_se table
        result3 = conn.execute(text(f"""
            SELECT COUNT(*) FROM nvd_se se
            JOIN nvd_ra r ON se.kaisai_nen = r.kaisai_nen 
                          AND se.keibajo_code = r.keibajo_code
                          AND se.kaisai_kai = r.kaisai_kai
                          AND se.kaisai_nichime = r.kaisai_nichime
                          AND se.race_bango = r.race_bango
            WHERE r.kaisai_nen = '{year}' AND r.kaisai_tsukihi = '{mmdd}'
        """))
        row3 = result3.fetchone()
        print(f'Total entries (horses): {row3[0]}')

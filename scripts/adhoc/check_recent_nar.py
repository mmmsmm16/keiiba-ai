"""
Check recent NAR race dates and venues in DB
"""
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from sqlalchemy import create_engine, text

user = os.environ.get('POSTGRES_USER', 'postgres')
password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
port = os.environ.get('POSTGRES_PORT', '5433')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(url)

VENUE_MAP = {
    '36': '大井', '37': '川崎', '38': '浦和', '39': '船橋',
    '42': '笠松', '43': '名古屋', '44': '金沢',
    '45': '園田', '46': '姫路', '47': '福山', '48': '高知', '50': '佐賀',
    '51': '荒尾', '52': '盛岡', '53': '水沢', '54': '門別',
    '55': '北見', '56': '旭川', '57': '帯広'
}

with engine.connect() as conn:
    print("=== Recent NAR Race Data (Dec 2025) ===\n")
    
    result = conn.execute(text("""
        SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, COUNT(*) as cnt 
        FROM nvd_ra 
        WHERE kaisai_nen = '2025' AND kaisai_tsukihi >= '1215'
        GROUP BY kaisai_nen, kaisai_tsukihi, keibajo_code
        ORDER BY kaisai_tsukihi, keibajo_code
    """))
    
    current_date = None
    for r in result.fetchall():
        date_str = f"{r[0]}-{r[1][:2]}-{r[1][2:]}"
        venue = VENUE_MAP.get(r[2], r[2])
        
        if current_date != date_str:
            if current_date is not None:
                print()
            print(f"[{date_str}]")
            current_date = date_str
        
        print(f"  {venue}: {r[3]} races")

    print("\n=== Summary ===")
    # Check if 12/18 specifically has no data
    result2 = conn.execute(text("""
        SELECT COUNT(*) FROM nvd_ra 
        WHERE kaisai_nen = '2025' AND kaisai_tsukihi = '1218'
    """))
    cnt = result2.fetchone()[0]
    
    if cnt == 0:
        print("12/18: NO DATA - NAR races may not be scheduled or data not yet loaded")
    else:
        print(f"12/18: {cnt} races available")

from sqlalchemy import create_engine, text
import os
import pandas as pd

user = os.environ.get('POSTGRES_USER', 'user')
password = os.environ.get('POSTGRES_PASSWORD', 'password')
host = os.environ.get('POSTGRES_HOST', 'db')
port = os.environ.get('POSTGRES_PORT', '5432')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
db_url = f'postgresql://{user}:{password}@{host}:{port}/{dbname}'
engine = create_engine(db_url)

with engine.connect() as conn:
    print("--- DB Check ---")
    
    # Check count of JRA races in 2025
    res = conn.execute(text("""
        SELECT count(*) 
        FROM jvd_ra 
        WHERE kaisai_nen = '2025' 
          AND jyo_cd IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
    """))
    count = res.fetchone()[0]
    print(f'2025 JRA Races: {count}')

    # Check latest date for JRA
    res = conn.execute(text("""
        SELECT MAX(TO_DATE(kaisai_nen || kaisai_tsukihi, 'YYYYMMDD')) 
        FROM jvd_ra 
        WHERE jyo_cd IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
    """))
    val = res.fetchone()[0]
    print(f'Latest JRA Date: {val}')
    
    # Check count for NAR (Local) in 2025 just for comparison
    res = conn.execute(text("""
        SELECT count(*) 
        FROM jvd_ra 
        WHERE kaisai_nen = '2025' 
          AND jyo_cd NOT IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
    """))
    nar_count = res.fetchone()[0]
    print(f'2025 NAR Races: {nar_count}')
    
    # Check specific date 2025-01-05
    target = '20250105'
    res = conn.execute(text(f"""
        SELECT MIN(TO_DATE(kaisai_nen || kaisai_tsukihi, 'YYYYMMDD')) 
        FROM jvd_ra 
        WHERE TO_DATE(kaisai_nen || kaisai_tsukihi, 'YYYYMMDD') > TO_DATE('{target}', 'YYYYMMDD')
          AND jyo_cd IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
    """))
    next_date = res.fetchone()[0]
    print(f'Next JRA Date after 2025-01-05: {next_date}')

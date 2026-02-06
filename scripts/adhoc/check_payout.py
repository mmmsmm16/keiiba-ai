"""Check payout data from jvd_hr for 2025-11-30"""
import os
from sqlalchemy import text, create_engine
import pandas as pd

PGSQL_HOST = os.environ.get('PGSQL_HOST', 'db')
PGSQL_PORT = os.environ.get('PGSQL_PORT', '5432')
PGSQL_DB = os.environ.get('PGSQL_DB', 'keiba_ai')
PGSQL_USER = os.environ.get('PGSQL_USER', 'keiba_ai')
PGSQL_PASSWORD = os.environ.get('PGSQL_PASSWORD', 'keiba_ai')
db_url = f"postgresql://{PGSQL_USER}:{PGSQL_PASSWORD}@{PGSQL_HOST}:{PGSQL_PORT}/{PGSQL_DB}"
engine = create_engine(db_url)

# 11/30の払戻データを確認
query = text("""
    SELECT 
        CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
        haraimodoshi_umaren_1a, haraimodoshi_umaren_1b, 
        haraimodoshi_umaren_2a, haraimodoshi_umaren_2b, 
        haraimodoshi_umaren_3a, haraimodoshi_umaren_3b
    FROM jvd_hr 
    WHERE kaisai_nen='2025' 
    AND race_bango='10'
    LIMIT 20
""")

df = pd.read_sql(query, engine)
print("Umaren payout sample:")
print(df.to_string())

# 08 10R の場合を確認 (race_id: 202508040810)
query2 = text("""
    SELECT 
        CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
        haraimodoshi_umaren_1a, haraimodoshi_umaren_1b
    FROM jvd_hr 
    WHERE CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) = '202508040810'
""")
df2 = pd.read_sql(query2, engine)
print("\nRace 202508040810 (08 10R) umaren payout:")
print(df2.to_string())

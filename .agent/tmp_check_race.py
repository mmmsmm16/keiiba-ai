import pandas as pd
from src.preprocessing.loader import JraVanDataLoader
loader = JraVanDataLoader()
query = """
SELECT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
       kaisai_nen,
       kaisai_tsukihi
FROM jvd_ra
ORDER BY kaisai_nen DESC, kaisai_tsukihi DESC, race_bango DESC
LIMIT 5
"""
df = pd.read_sql(query, loader.engine)
print(df.to_string(index=False))

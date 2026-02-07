import sys
sys.path.append('/workspace')

import pandas as pd
from src.preprocessing.loader import JraVanDataLoader as Loader

loader = Loader()

def run(sql):
    return pd.read_sql(sql, loader.engine)

q_ra = """
SELECT kaisai_nen, kaisai_tsukihi,
       COUNT(*) AS ra_rows,
       COUNT(DISTINCT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango)) AS ra_races
FROM jvd_ra
WHERE kaisai_nen='2026' AND kaisai_tsukihi IN ('0207','0208')
GROUP BY 1,2
ORDER BY 1,2
"""

q_o1 = """
SELECT kaisai_nen, kaisai_tsukihi,
       COUNT(*) AS o1_rows,
       COUNT(DISTINCT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango)) AS o1_races
FROM jvd_o1
WHERE kaisai_nen='2026' AND kaisai_tsukihi IN ('0207','0208')
GROUP BY 1,2
ORDER BY 1,2
"""

q_apd = """
SELECT kaisai_nen, kaisai_tsukihi,
       COUNT(*) AS apd_rows,
       COUNT(DISTINCT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango)) AS apd_races
FROM apd_sokuho_o1
WHERE kaisai_nen='2026' AND kaisai_tsukihi IN ('0207','0208')
GROUP BY 1,2
ORDER BY 1,2
"""

q_max = """
SELECT
  (SELECT MAX(kaisai_nen || kaisai_tsukihi) FROM jvd_ra) AS ra_max,
  (SELECT MAX(kaisai_nen || kaisai_tsukihi) FROM jvd_o1) AS o1_max,
  (SELECT MAX(kaisai_nen || kaisai_tsukihi) FROM apd_sokuho_o1) AS apd_max
"""

ra = run(q_ra)
o1 = run(q_o1)
apd = run(q_apd)
mx = run(q_max)

print('[jvd_ra]')
print(ra.to_string(index=False) if not ra.empty else '(empty)')
print('\n[jvd_o1]')
print(o1.to_string(index=False) if not o1.empty else '(empty)')
print('\n[apd_sokuho_o1]')
print(apd.to_string(index=False) if not apd.empty else '(empty)')
print('\n[max_date]')
print(mx.to_string(index=False))

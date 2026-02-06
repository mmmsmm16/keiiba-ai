import sys
import os
import pandas as pd
from sqlalchemy import text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    target_date = "20260131"
    year = target_date[:4]
    mmdd = target_date[4:]
    
    # Check if any rank is > 0 across the whole day
    with loader.engine.connect() as conn:
        q = text(f"SELECT count(*) as count FROM jvd_se WHERE kaisai_nen=:year AND kaisai_tsukihi=:mmdd AND kakutei_chakujun > '00'")
        res = conn.execute(q, {"year": year, "mmdd": mmdd}).fetchone()
        print(f"Number of horses with kakutei_chakujun > '00' on {target_date}: {res[0]}")
        
        # Check total horses on that day
        q_total = text(f"SELECT count(*) as count FROM jvd_se WHERE kaisai_nen=:year AND kaisai_tsukihi=:mmdd")
        res_total = conn.execute(q_total, {"year": year, "mmdd": mmdd}).fetchone()
        print(f"Total horses on {target_date}: {res_total[0]}")

        # Check a few races specifically
        q_races = text(f"SELECT race_bango, count(*) FROM jvd_se WHERE kaisai_nen=:year AND kaisai_tsukihi=:mmdd AND kakutei_chakujun > '00' GROUP BY race_bango ORDER BY race_bango")
        res_races = conn.execute(q_races, {"year": year, "mmdd": mmdd}).fetchall()
        print(f"\nRaces with results: {res_races}")

if __name__ == "__main__":
    main()

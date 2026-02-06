import sys
import os
import pandas as pd
from sqlalchemy import text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    
    # Check 2/1 (Today)
    target_date = "20260201"
    year = target_date[:4]
    mmdd = target_date[4:]
    with loader.engine.connect() as conn:
        # Check if entries exist
        q_count = text("SELECT count(*) FROM jvd_se WHERE kaisai_nen=:year AND kaisai_tsukihi=:mmdd")
        res_count = conn.execute(q_count, {"year": year, "mmdd": mmdd}).fetchone()
        print(f"Total horses for {target_date}: {res_count[0]}")
        
        # Check if results exist
        q_res = text("SELECT count(*) FROM jvd_se WHERE kaisai_nen=:year AND kaisai_tsukihi=:mmdd AND kakutei_chakujun > '00'")
        res_res = conn.execute(q_res, {"year": year, "mmdd": mmdd}).fetchone()
        print(f"Horses with results for {target_date}: {res_res[0]}")

if __name__ == "__main__":
    main()

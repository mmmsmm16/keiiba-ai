import sys
import os
import pandas as pd
from sqlalchemy import text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    
    with loader.engine.connect() as conn:
        print("--- Horse Data Info for 1/31 (Yesterday) ---")
        q1 = text("SELECT kaisai_nen, kaisai_tsukihi, race_bango, umaban, data_kubun, kakutei_chakujun FROM jvd_se WHERE kaisai_nen='2026' AND kaisai_tsukihi='0131' LIMIT 5")
        df1 = pd.read_sql(q1, conn)
        print(df1)
        
        print("\n--- Horse Data Info for 2/1 (Today) ---")
        q2 = text("SELECT kaisai_nen, kaisai_tsukihi, race_bango, umaban, data_kubun, kakutei_chakujun FROM jvd_se WHERE kaisai_nen='2026' AND kaisai_tsukihi='0201' LIMIT 5")
        df2 = pd.read_sql(q2, conn)
        print(df2)

if __name__ == "__main__":
    main()

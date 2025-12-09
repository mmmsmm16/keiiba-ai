import os
import sys
import pandas as pd
from sqlalchemy import create_engine

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from inference.loader import InferenceDataLoader

def main():
    loader = InferenceDataLoader()
    target_date = '20251207'
    
    print(f"Checking data for {target_date}...")
    
    # 1. Check jvd_ra (Race List)
    query_ra = f"SELECT count(*) FROM jvd_ra WHERE (kaisai_nen || kaisai_tsukihi) = '{target_date}'"
    try:
        count_ra = pd.read_sql(query_ra, loader.engine).iloc[0, 0]
        print(f"jvd_ra count: {count_ra}")
    except Exception as e:
        print(f"jvd_ra error: {e}")

    # 2. Check jvd_ur (Entries)
    query_ur = f"SELECT count(*) FROM jvd_ur WHERE (kaisai_nen || kaisai_tsukihi) = '{target_date}'"
    try:
        count_ur = pd.read_sql(query_ur, loader.engine).iloc[0, 0]
        print(f"jvd_ur count: {count_ur}")
    except Exception as e:
        print(f"jvd_ur error: {e}")

    # 3. Check jvd_se (Results)
    query_se = f"SELECT count(*) FROM jvd_se WHERE (kaisai_nen || kaisai_tsukihi) = '{target_date}'"
    try:
        count_se = pd.read_sql(query_se, loader.engine).iloc[0, 0]
        print(f"jvd_se count: {count_se}")
    except Exception as e:
        print(f"jvd_se error: {e}")

if __name__ == "__main__":
    main()

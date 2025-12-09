import os
import sys
import pandas as pd
from sqlalchemy import create_engine

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from inference.loader import InferenceDataLoader

def main():
    loader = InferenceDataLoader()
    
    print("Checking latest dates in DB...")
    
    # Check Race List
    query_ra = "SELECT MAX(kaisai_nen || kaisai_tsukihi) FROM jvd_ra"
    try:
        max_ra = pd.read_sql(query_ra, loader.engine).iloc[0, 0]
        print(f"Latest Race (jvd_ra): {max_ra}")
    except Exception as e:
        print(f"jvd_ra error: {e}")

    # Check Results
    query_se = "SELECT MAX(kaisai_nen || kaisai_tsukihi) FROM jvd_se"
    try:
        max_se = pd.read_sql(query_se, loader.engine).iloc[0, 0]
        print(f"Latest Result (jvd_se): {max_se}")
    except Exception as e:
        print(f"jvd_se error: {e}")

if __name__ == "__main__":
    main()

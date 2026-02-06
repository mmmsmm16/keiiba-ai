import os
import sys
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.nar.loader import NarDataLoader

def check_nar_data():
    print("Checking NAR data availability...")
    loader = NarDataLoader()
    
    # Check Record Counts
    tables = ['nvd_ra', 'nvd_race_shosai', 'nvd_o1']
    
    for tbl in tables:
        try:
            real_tbl = loader._get_table_name([tbl])
            query = text(f"SELECT count(*) FROM {real_tbl}")
            with loader.engine.connect() as conn:
                count = conn.execute(query).scalar()
            print(f"Table '{real_tbl}': {count} records")
        except Exception as e:
            print(f"Table '{tbl}': Error checking ({e})")

    # Check Latest Date
    try:
        query = text(f"SELECT MAX(kaisai_nen) as y, MAX(kaisai_tsukihi) as md FROM {loader._get_table_name(['nvd_ra'])}")
        with loader.engine.connect() as conn:
            res = conn.execute(query).mappings().first()
            print(f"Latest Race Date: {res['y']}-{res['md']}")
    except Exception:
        pass

if __name__ == "__main__":
    check_nar_data()

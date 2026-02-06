
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def check_date_range():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)
    
    try:
        with engine.connect() as conn:
            # Check range of dates in apd_sokuho_o1
            # Columns: kaisai_nen (YYYY), kaisai_tsukihi (MMDD)
            query = text("""
                SELECT 
                    MIN(CONCAT(kaisai_nen, kaisai_tsukihi)) as min_date,
                    MAX(CONCAT(kaisai_nen, kaisai_tsukihi)) as max_date,
                    COUNT(*) as count
                FROM apd_sokuho_o1
                WHERE kaisai_nen = '2025'
            """)
            df = pd.read_sql(query, conn)
            print("2025 Data availability in apd_sokuho_o1:")
            print(df)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_date_range()


import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def fetch_umaren_sample():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    try:
        engine = create_engine(connection_str)
        with engine.connect() as conn:
            # Fetch most recent record from o2
            query = text("SELECT * FROM apd_sokuho_o2 ORDER BY data_sakusei_nengappi DESC LIMIT 1")
            df = pd.read_sql(query, conn)
            
            if not df.empty:
                print("Sample Data (Latest O2 - Umaren):")
                row = df.iloc[0]
                for col in df.columns:
                    val = row[col]
                    # Truncate long strings for display
                    if isinstance(val, str) and len(val) > 100:
                        print(f"{col}: {val[:100]}... (Total len: {len(val)})")
                    else:
                        print(f"{col}: {val}")
            else:
                print("No data found in apd_sokuho_o2")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_umaren_sample()

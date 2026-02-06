
import os
import sys
from sqlalchemy import create_engine, inspect

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def inspect_table():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    try:
        engine = create_engine(connection_str)
        inspector = inspect(engine)
        
        table_name = 'apd_sokuho_o1'
        print(f"Inspecting {table_name}...")
        columns = inspector.get_columns(table_name)
        for col in columns:
            print(f"- {col['name']} ({col['type']})")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_table()

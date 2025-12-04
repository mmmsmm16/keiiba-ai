import os
import psycopg2
import sys
from sqlalchemy import create_engine, text

def check_db_connection():
    """
    Checks connection to the PostgreSQL database defined in docker-compose.yml.
    """
    db_user = os.environ.get('POSTGRES_USER', 'user')
    db_password = os.environ.get('POSTGRES_PASSWORD', 'password')
    db_name = os.environ.get('POSTGRES_DB', 'keiba')
    db_host = os.environ.get('POSTGRES_HOST', 'db') # 'db' is the service name in docker-compose
    db_port = os.environ.get('POSTGRES_PORT', '5432')

    print(f"Connecting to database {db_name} at {db_host}:{db_port} as {db_user}...")

    try:
        # Construct the connection string
        # If running outside docker but wanting to connect to localhost mapped port
        if db_host == 'db':
             # If this script runs outside docker (e.g. on host), we might need localhost
             # But inside the container 'db' resolves.
             # We'll assume this runs inside the container or user sets POSTGRES_HOST=localhost
             pass

        connection_str = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_str)

        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("Successfully connected to the database!")

            # Check if tables exist
            result = connection.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]
            print(f"Found tables: {tables}")

            expected_tables = {'races', 'horses', 'results', 'payouts'}
            if expected_tables.issubset(set(tables)):
                print("All expected tables are present.")
            else:
                print(f"Missing tables: {expected_tables - set(tables)}")

    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_db_connection()

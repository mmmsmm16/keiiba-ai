
import os
from sqlalchemy import create_engine, inspect

def main():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)

    inspector = inspect(engine)
    cols = [c['name'] for c in inspector.get_columns('jvd_o1')]
    
    with open('cols_o1.txt', 'w') as f:
        for c in cols:
            f.write(c + '\n')
    
    print("Written to cols_o1.txt")

if __name__ == "__main__":
    main()

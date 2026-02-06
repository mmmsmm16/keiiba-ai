
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
    
    with open('cols_complex.txt', 'w') as f:
        for t in ['jvd_o3', 'jvd_o5', 'jvd_o6']:
            f.write(f"=== {t} ===\n")
            try:
                cols = [c['name'] for c in inspector.get_columns(t)]
                for c in cols:
                    f.write(c + "\n")
            except:
                f.write("Error\n")
            f.write("\n")

if __name__ == "__main__":
    main()

from sqlalchemy import create_engine, inspect
import os

def main():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    # Use localhost if running outside container, or service name if inside
    # Assuming running via docker-compose exec app ...
    
    connection_str = f"postgresql://{user}:{password}@{host}:5432/{dbname}"
    print(f"Connecting to {connection_str.replace(password, '***')}...")
    
    engine = create_engine(connection_str)
    inspector = inspect(engine)
    
    with engine.connect() as conn:
        from sqlalchemy import text
        # jvd_hr usually uses same PK as jvd_ra (kaisai keys + race_bango)
        # Columns might meet PC-KEIBA standard names
        query = text("SELECT * FROM jvd_hr WHERE kaisai_nen='2024' LIMIT 1")
        row = conn.execute(query).fetchone()
        if row:
            keys = row._mapping.keys()
            # print(keys) # dump all keys
            # Filter interesting keys
            for k in keys:
                if 'haraimodoshi' in k and ('umaren' in k or 'sanren' in k) and ('1a' in k or '1b' in k):
                    print(f"{k}: {row._mapping[k]}")
        else:
            print("No data found for 2024.")

if __name__ == '__main__':
    main()

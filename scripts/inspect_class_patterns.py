from sqlalchemy import create_engine
import pandas as pd
import os

user = os.environ.get('POSTGRES_USER', 'user')
password = os.environ.get('POSTGRES_PASSWORD', 'password')
host = os.environ.get('POSTGRES_HOST', 'db')
port = os.environ.get('POSTGRES_PORT', '5432')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')

connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(connection_str)

query = """
SELECT 
    kyosei_meisho,
    kyosomei_hondai, 
    grade_code, 
    kyoso_shubetsu_code, 
    kyoso_joken_code 
FROM nvd_ra 
WHERE keibajo_code IN ('42','43','44','45') 
LIMIT 200
"""

try:
    df = pd.read_sql(query, engine)
    # Check if 'kyosei_meisho' (this is where class often is in PC-KEIBA Chiho result) exists
    # If not, use what's available
    print("Columns available:", df.columns.tolist())
    
    # Filter for titles containing A, B, C or specific patterns
    patterns = ['A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
    for p in patterns:
        subset = df[df['kyosomei_hondai'].str.contains(p, na=False)]
        if not subset.empty:
            print(f"\n--- Pattern: {p} ---")
            print(subset.head(3))
except Exception as e:
    print(f"Error: {e}")

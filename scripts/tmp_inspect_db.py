"""Analyze odds_tansho string structure"""
from sqlalchemy import create_engine
import pandas as pd
import os

user = os.environ.get('POSTGRES_USER', 'postgres')
password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
port = os.environ.get('POSTGRES_PORT', '5433')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

# Get one sample with full string
query = """
SELECT 
    odds_tansho,
    toroku_tosu
FROM apd_sokuho_o1
WHERE kaisai_nen = '2025' AND toroku_tosu IS NOT NULL
ORDER BY happyo_tsukihi_jifun DESC
LIMIT 1
"""
df = pd.read_sql(query, engine)
row = df.iloc[0]
odds_str = row['odds_tansho']
n_horses = int(row['toroku_tosu'])

print(f"Total String Length: {len(odds_str)}")
print(f"Num Horses: {n_horses}")
print(f"String: {odds_str}")

# Hypothesis: Fixed length per horse. Maybe 28 horses max?
# 224 / 28 = 8 chars?
# Let's try 8 chars splitting
# 01 0030 02
# 01: Horse Num
# 0030: Odds (3.0?)
# 02: Ninki order

print("\n=== Parsing Attempt (8 chars per block) ===")
for i in range(n_horses):
    start = i * 8
    end = start + 8
    block = odds_str[start:end]
    
    umaban = block[0:2]
    odds_val = block[2:6] # 4 digits odds?
    ninki = block[6:8]
    
    try:
        odds_float = int(odds_val) / 10.0
    except:
        odds_float = -1
        
    print(f"Horse {i+1}: Block='{block}' -> Uma={umaban}, Odds={odds_val} ({odds_float}), Ninki={ninki}")

from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')

q = "SELECT COUNT(DISTINCT CONCAT(kaisai_nen, keibajo_code, LPAD(kaisai_kai::text, 2, '0'), LPAD(kaisai_nichime::text, 2, '0'), LPAD(race_bango::text, 2, '0'))) as cnt FROM jvd_hr WHERE kaisai_nen = '2024'"
count = pd.read_sql(text(q), engine)
print(f"JVD_HR 2024 Races: {count['cnt'][0]}")

# Check overlaps
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df_2024 = df[df['date'].dt.year == 2024]
parquet_races = set(df_2024['race_id'].unique().astype(str))

# Get DB race_ids list
races_db_q = "SELECT DISTINCT CONCAT(kaisai_nen, keibajo_code, LPAD(kaisai_kai::text, 2, '0'), LPAD(kaisai_nichime::text, 2, '0'), LPAD(race_bango::text, 2, '0')) as race_id FROM jvd_hr WHERE kaisai_nen = '2024'"
db_races = pd.read_sql(text(races_db_q), engine)
db_races_set = set(db_races['race_id'].astype(str))

common = parquet_races.intersection(db_races_set)
print(f"Common Races: {len(common)}")
print(f"Example Common: {list(common)[:3]}")
print(f"Example Parquet Only: {list(parquet_races - db_races_set)[:3]}")
print(f"Example DB Only: {list(db_races_set - parquet_races)[:3]}")

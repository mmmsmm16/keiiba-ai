import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.nar.loader import NarDataLoader

def analyze_venues():
    loader = NarDataLoader()
    
    # Query to count races by venue code
    # Also tries to get venue name if possible? PC-KEIBA usually has master codes but not loaded here.
    # We will just dump codes and counts.
    
    query = """
    SELECT keibajo_code, count(*) as count, min(kaisai_nen) as min_year, max(kaisai_nen) as max_year
    FROM nvd_ra
    GROUP BY keibajo_code
    ORDER BY keibajo_code
    """
    
    try:
        df = pd.read_sql(query, loader.engine)
        print("NAR Venue Statistics:")
        print(df)
        
        # Expected South Kanto Codes (Need verification):
        # 42: Urawa
        # 43: Funabashi
        # 44: Ohi
        # 45: Kawasaki
        
        south_kanto_codes = ['42', '43', '44', '45']
        
        df['is_south_kanto'] = df['keibajo_code'].isin(south_kanto_codes)
        sk_df = df[df['is_south_kanto']]
        
        print("\nSouth Kanto Only:")
        print(sk_df)
        print(f"\nTotal South Kanto Races: {sk_df['count'].sum()}")
        print(f"Total Other NAR Races: {df[~df['is_south_kanto']]['count'].sum()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_venues()

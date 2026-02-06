
import os
import sys
import pandas as pd
from sqlalchemy import create_engine

# Adjust path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.preprocessing.loader import JraVanDataLoader

def test_loader_mining():
    loader = JraVanDataLoader()
    # Limit to small number for speed
    df = loader.load(limit=1000, history_start_date='2024-01-01')
    
    print("Columns loaded:", df.columns.tolist())
    
    mining_cols = ['mining_kubun', 'yoso_soha_time', 'yoso_gosa_plus', 'yoso_gosa_minus', 'yoso_juni']
    for c in mining_cols:
        if c in df.columns:
            print(f"✅ {c} exists.")
            print(df[c].head())
            print(f"Null count: {df[c].isnull().sum()} / {len(df)}")
        else:
            print(f"❌ {c} MISSING!")

if __name__ == "__main__":
    test_loader_mining()

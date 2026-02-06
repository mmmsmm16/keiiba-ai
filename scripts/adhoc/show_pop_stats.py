import pandas as pd
from sqlalchemy import create_engine
import os

def main():
    # Connect
    engine = create_engine('postgresql://postgres:postgres@host.docker.internal:5433/pckeiba')
    
    # Query: Win Rate by Popularity (JRA Only, 2024)
    query = """
    SELECT 
        tansho_ninkijun as popularity,
        COUNT(*) as total_races,
        SUM(CASE WHEN kakutei_chakujun = 1 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN kakutei_chakujun <= 3 THEN 1 ELSE 0 END) as top3
    FROM jvd_se
    WHERE kaisai_nen = '2024'
      AND keibajo_code BETWEEN '01' AND '10' -- JRA
      AND tansho_ninkijun > 0
    GROUP BY tansho_ninkijun
    ORDER BY tansho_ninkijun
    LIMIT 10
    """
    
    df = pd.read_sql(query, engine)
    
    df['win_rate'] = (df['wins'] / df['total_races'] * 100).round(1)
    df['top3_rate'] = (df['top3'] / df['total_races'] * 100).round(1)
    
    print("\n=== 人気順別 成績 (2024 JRA) ===")
    print(df[['popularity', 'win_rate', 'top3_rate', 'total_races']].to_markdown(index=False))

if __name__ == "__main__":
    main()


import sys
import os
import pandas as pd
from sqlalchemy import text

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.inference.loader import InferenceDataLoader

def main():
    loader = InferenceDataLoader()
    
    # Analyze Last 10 Years (2015-2025)
    # Group by Provider (JRA/NAR) and Surface (Turf/Dirt)
    
    query = text("""
        SELECT 
            CASE 
                WHEN keibajo_code BETWEEN '01' AND '10' THEN 'JRA'
                ELSE 'NAR'
            END as provider,
            CASE 
                WHEN track_code BETWEEN '10' AND '22' THEN 'Turf'
                WHEN track_code BETWEEN '23' AND '29' THEN 'Dirt'
                WHEN track_code BETWEEN '51' AND '59' THEN 'Jump'
                ELSE 'Other'
            END as surface,
            COUNT(*) as race_count
        FROM jvd_ra
        WHERE kaisai_nen >= '2015'
        GROUP BY 
            CASE 
                WHEN keibajo_code BETWEEN '01' AND '10' THEN 'JRA'
                ELSE 'NAR'
            END,
            CASE 
                WHEN track_code BETWEEN '10' AND '22' THEN 'Turf'
                WHEN track_code BETWEEN '23' AND '29' THEN 'Dirt'
                WHEN track_code BETWEEN '51' AND '59' THEN 'Jump'
                ELSE 'Other'
            END
        ORDER BY provider, surface
    """)
    
    print("Executing detailed query (2015-2025)...")
    with loader.engine.connect() as conn:
        result = conn.execute(query)
        
        print(f"{'Provider':<10} | {'Surface':<10} | {'Count':<10} | {'% of Total'}")
        print("-" * 50)
        
        rows = list(result)
        total_races = sum(r[2] for r in rows)
        
        total_dirt = 0
        total_turf = 0
        
        for row in rows:
            prov = row[0]
            surf = row[1]
            cnt = row[2]
            pct = (cnt / total_races) * 100
            print(f"{prov:<10} | {surf:<10} | {cnt:<10} | {pct:.1f}%")
            
            if surf == 'Dirt': total_dirt += cnt
            if surf == 'Turf': total_turf += cnt
            
        print("-" * 50)
        print(f"Total Races: {total_races}")
        print(f"Total Dirt:  {total_dirt} ({total_dirt/total_races*100:.1f}%)")
        print(f"Total Turf:  {total_turf} ({total_turf/total_races*100:.1f}%)")

if __name__ == "__main__":
    main()

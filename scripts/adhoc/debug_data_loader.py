import os
import sys
import pandas as pd
import logging

# プロジェクトルートにパスを通す
sys.path.append(os.getcwd())

from src.preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_loader():
    print("=== Data Loader Debug ===")
    loader = JraVanDataLoader()
    
    # 直近のデータ（2024年12月以降）を取得
    # jra_only=Falseにして地方競馬（もしあれば）も確認するが、基本はJRA
    try:
        df = loader.load(limit=2000, history_start_date="2024-12-01")
        
        print(f"\nLoaded {len(df)} rows.")
        print("\n=== Columns ===")
        print(df.columns.tolist())
        
        print("\n=== Odds Check ===")
        odds_nan = df['odds'].isna().sum()
        odds_zero = (df['odds'] == 0).sum()
        print(f"Odds NaN: {odds_nan} ({odds_nan/len(df):.1%})")
        print(f"Odds Zero: {odds_zero}")
        print("Sample Odds:", df['odds'].head(10).tolist())

        # Odds=0 Analysis
        odds_zero_df = df[df['odds'] == 0]
        if len(odds_zero_df) > 0:
            print("\n=== Odds=0 Analysis ===")
            finished_but_zero_odds = odds_zero_df[odds_zero_df['rank'].notna()]
            print(f"Finished (Rank exists) but Odds=0: {len(finished_but_zero_odds)} rows")
            if len(finished_but_zero_odds) > 0:
                print(finished_but_zero_odds[['race_id', 'horse_name', 'rank', 'odds', 'popularity']].head())
            else:
                print("All Odds=0 rows are unfinished (Rank is NaN). This is normal.")
        
        print("\n=== Sire ID Check ===")
        sire_nan = df['sire_id'].isna().sum()
        print(f"Sire ID NaN: {sire_nan} ({sire_nan/len(df):.1%})")
        print("Sample Sire IDs:", df['sire_id'].head(10).tolist())
        
        print("\n=== Prize Check ===")
        if 'honshokin' in df.columns:
            prize_nan = df['honshokin'].isna().sum()
            prize_zero = (df['honshokin'] == 0).sum()
            print(f"Prize NaN: {prize_nan}")
            print(f"Prize Zero: {prize_zero}")
        else:
            print("WARNING: 'honshokin' column not found in DataFrame.")
            # 元のクエリに含まれているか確認したいが、DataFrameにはない
            
        # rawデータも少し見たいので、loader内部の実装を真似てSQLを投げる手もあるが、
        # まずはロードされた結果を見る。
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_loader()

import pandas as pd
import os

CACHE_DIR = "data/features"

def main():
    print("Loading base_attributes...")
    df_base = pd.read_parquet(os.path.join(CACHE_DIR, "base_attributes.parquet"))
    
    # Extract Year
    df_base['year'] = df_base['race_id'].astype(str).str[:4].astype(int)
    
    # 1. Check Rows per Race
    print("\n--- Rows per Race Stats ---")
    race_counts = df_base.groupby(['year', 'race_id']).size()
    print(race_counts.groupby('year').describe())
    
    # 2. Inspect Jockey Stats
    print("\nLoading temporal_jockey_stats...")
    df_jockey = pd.read_parquet(os.path.join(CACHE_DIR, "temporal_jockey_stats.parquet"))
    
    # Merge year and jockey_id
    df_jockey = df_jockey.merge(df_base[['race_id', 'horse_number', 'year', 'jockey_id']], on=['race_id', 'horse_number'], how='left')
    
    print("\n--- Jockey Stats Sample (2024) ---")
    sample_24 = df_jockey[df_jockey['year'] == 2024].sample(10)
    print(sample_24[['race_id', 'jockey_n_races_180d', 'jockey_win_rate_180d']])
    
    print("\n--- Jockey Stats Sample (2025) ---")
    sample_25 = df_jockey[df_jockey['year'] == 2025].sample(10)
    print(sample_25[['race_id', 'jockey_n_races_180d', 'jockey_win_rate_180d']])
    
    # 3. Check for specific Jockey Duplication
    # Pick a jockey and see their entries on a specific date
    print("\n--- Duplicate Check for a common jockey ---")
    # Exclude 00000
    valid_jockeys = df_base[df_base['jockey_id'] != 0]['jockey_id']
    if not valid_jockeys.empty:
        jockey_id = valid_jockeys.mode()[0]
        print(f"Checking Jockey: {jockey_id}")
        
        # Sort by race_id (date not available in base_attributes parquet directly?)
        # Actually base_attributes usually has 'date' if pipeline includes it.
        # But earlier error said KeyError 'date'.
        # Let's inspect columns of df_base again.
        print(f"Base Columns: {df_base.columns.tolist()}")
        
        # Merge date from somewhere? 
        # Or just use race_id as proxy for sorting
        subset = df_base[df_base['jockey_id'] == jockey_id].sort_values('race_id')
        print(f"Total entries: {len(subset)}")
    
    # 4. Statistical Comparison (Excluding Outliers)
    print("\n--- Stats (Excluding Jockey 0/00000) ---")
    # Merge year to df_jockey
    # df_jockey is already merged above
    
    # Filter 0 or '00000'
    mask_valid = (df_jockey['jockey_id'] != 0) & (df_jockey['jockey_id'] != '00000') & (df_jockey['jockey_id'] != 50000) # Common outliers
    
    jockey_valid = df_jockey[mask_valid]
    
    print("2024 Summary:")
    print(jockey_valid[jockey_valid['year']==2024]['jockey_n_races_180d'].describe())
    
    print("2025 Summary:")
    print(jockey_valid[jockey_valid['year']==2025]['jockey_n_races_180d'].describe())

if __name__ == "__main__":
    main()

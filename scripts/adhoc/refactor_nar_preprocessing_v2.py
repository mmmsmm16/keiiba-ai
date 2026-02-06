import os

# Source is the clean original file
source_path = "src/preprocessing/run_preprocessing.py"
target_path = "src/nar/run_preprocessing.py"

# Read source (assuming UTF-8 as it works in container usually)
with open(source_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replacements
content = content.replace("src.preprocessing", "src.nar")
content = content.replace("JraVanDataLoader", "NarDataLoader")
content = content.replace("JRA-VAN", "NAR/PC-KEIBA")

# Replace Japanese help text to avoid encoding issues in all environments
content = content.replace("help='前回のStep 5完了時点から再開します'", "help='Resume from Step 5 checkpoint'")
content = content.replace("help='JRA (01-10) のレースのみに限定します'", "help='Filter JRA races only (Legacy)'") # Probably unused in NAR but safe to replace
content = content.replace("help='この日付以降のデータのみをロードします'", "help='Load data after this date'")
content = content.replace("help='チェックポイントファイルが存在する場合に削除して最初から実行します'", "help='Force restart (delete checkpoint)'")

# Output path customization
if 'DATA_DIR = "data"' in content:
    content = content.replace('DATA_DIR = "data"', 'DATA_DIR = "data/nar"')
    
if 'preprocessed_data_v11.parquet' in content:
    content = content.replace('preprocessed_data_v11.parquet', 'preprocessed_data_south_kanto.parquet')

# Wrapper Function Injection (Same as before)
wrapper_func = """
def filter_south_kanto(df):
    if df is None or df.empty:
        return df
    
    logger.info("Filtering for South Kanto (Minami Kanto) tracks...")
    SOUTH_KANTO_CODES = [42, 43, 44, 45]
    original_len = len(df)
    try:
        df = df[df['venue'].astype(str).isin([str(c) for c in SOUTH_KANTO_CODES])]
    except Exception as e:
        logger.warning(f"Venue filtering warning: {e}")
        
    logger.info(f"Venue Filter: {original_len} -> {len(df)} records")
    return df
"""

if "def main():" in content:
    content = content.replace("def main():", wrapper_func + "\n\ndef main():")

content = content.replace("loader.load(", "filter_south_kanto(loader.load(")

# Write to target with explicit UTF-8
with open(target_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Regenerated {target_path} from {source_path}")

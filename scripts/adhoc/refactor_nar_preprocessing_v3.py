import os

# Source is the clean original file
source_path = "src/preprocessing/run_preprocessing.py"
target_path = "src/nar/run_preprocessing.py"

# Read source (UTF-8)
with open(source_path, 'r', encoding='utf-8') as f:
    content = f.read()

# --- Feature 1: Fix Imports ---
content = content.replace("src.preprocessing", "src.nar")
content = content.replace("from preprocessing", "from src.nar")
content = content.replace("import preprocessing", "import src.nar as preprocessing") # Just in case
content = content.replace("JraVanDataLoader", "NarDataLoader")
content = content.replace("JRA-VAN", "NAR/PC-KEIBA")

# --- Feature 2: Fix Encoding Issues (Help Text) ---
# Replace Japanese help text to avoid encoding issues in all environments
content = content.replace("help='前回のStep 5完了時点から再開します'", "help='Resume from Step 5 checkpoint'")
content = content.replace("help='JRA (01-10) のレースのみに限定します'", "help='Filter JRA races only (Legacy)'")
content = content.replace("help='この日付以降のデータのみをロードします'", "help='Load data after this date'")
content = content.replace("help='チェックポイントファイルが存在する場合に削除して最初から実行します'", "help='Force restart (delete checkpoint)'")

# --- Feature 3: Fix Paths ---
if 'DATA_DIR = "data"' in content:
    content = content.replace('DATA_DIR = "data"', 'DATA_DIR = "data/nar"')
    
if 'preprocessed_data_v11.parquet' in content:
    content = content.replace('preprocessed_data_v11.parquet', 'preprocessed_data_south_kanto.parquet')

# --- Feature 4: Fix Arguments ---
content = content.replace("jra_only=args.jra_only,", "")
content = content.replace("jra_only=args.jra_only", "")
content = content.replace("jra_only=False,", "") # If hardcoded

# --- Feature 4: No complex injection needed (Loader handles filtering) ---

# Write to target with explicit UTF-8
with open(target_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Regenerated {target_path} (Simple Version)")

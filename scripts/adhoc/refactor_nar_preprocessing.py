import os

file_path = "src/nar/run_preprocessing.py"

with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Fix encoding artifacts if any (Powershell might have messed up BOM)
content = content.lstrip('\ufeff')

# Replacements
content = content.replace("src.preprocessing", "src.nar")
content = content.replace("JraVanDataLoader", "NarDataLoader")
# JRA specific imports in comments or strings
content = content.replace("JRA-VAN", "NAR/PC-KEIBA")

# Output path customization
if 'DATA_DIR = "data"' in content:
    content = content.replace('DATA_DIR = "data"', 'DATA_DIR = "data/nar"')
    
# Assuming default output filename
if 'preprocessed_data_v11.parquet' in content:
    content = content.replace('preprocessed_data_v11.parquet', 'preprocessed_data_south_kanto.parquet')

# Inject Region Filtering
# Look for where 'df = loader.load(...)' happens
# Insert filtering logic after that.
# We'll use a simple string injection.
load_call = "df = loader.load("
injection = """
        # [NAR Extension] Region Filtering
        # Default to South Kanto (42, 43, 44, 45) if not specified or just enforced for now
        # South Kanto: 42(Urawa), 43(Funabashi), 44(Ohi), 45(Kawasaki)
        SOUTH_KANTO_CODES = [42, 43, 44, 45]
        if df is not None and not df.empty:
            logger.info("Filtering for South Kanto (Minami Kanto) tracks...")
            original_len = len(df)
            # Try numeric filtering
            try:
                df['venue'] = pd.to_numeric(df['venue'], errors='coerce')
                df = df[df['venue'].isin(SOUTH_KANTO_CODES)]
            except:
                logger.warning("Venue filtering failed due to type mismatch")
            
            logger.info(f"Venue Filter: {original_len} -> {len(df)} records")

"""
# Note: loader.load is called in multiple places (incremental vs normal). 
# Safer to inject in NarDataLoader.load? No, better in run_preprocessing to control scope.
# But regex injection is risky.
# Let's append the filtering logic RIGHT AFTER any assignment to `df` that comes from `loader.load`.
# Or better, just override the `load` call pattern.

# Let's replace the load block in the main execution path.
# There are two main calls: line ~123 (incremental check) and line ~159 (full load).

# Strategy: Replace "df = loader.load(" with a wrapper call or inject code blocks.
# Actually, since we copied the file, we can be more aggressive.

# Main load
content = content.replace(
    "df = loader.load(", 
    "df = loader.load( # Modified for NAR\n"
)

# We will inject the filter logic manually after the `df = loader.load` lines by searching for the closing parenthesis
# This is getting complicated for string manipulation.
# Alternative: Add a function `filter_south_kanto(df)` and call it.

wrapper_func = """
def filter_south_kanto(df):
    if df is None or df.empty:
        return df
    
    logger.info("Filtering for South Kanto (Minami Kanto) tracks...")
    SOUTH_KANTO_CODES = [42, 43, 44, 45]
    original_len = len(df)
    # Ensure venue is numeric
    try:
        # Venue might be string '42' or int 42
        df = df[df['venue'].astype(str).isin([str(c) for c in SOUTH_KANTO_CODES])]
    except Exception as e:
        logger.warning(f"Venue filtering warning: {e}")
        
    logger.info(f"Venue Filter: {original_len} -> {len(df)} records")
    return df
"""

# Insert wrapper function definition before main()
if "def main():" in content:
    content = content.replace("def main():", wrapper_func + "\n\ndef main():")

# Call the wrapper.
# We need to find where df is assigned.
# "df = loader.load(" -> "df = filter_south_kanto(loader.load("
content = content.replace("loader.load(", "filter_south_kanto(loader.load(")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Modified src/nar/run_preprocessing.py")

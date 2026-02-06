import os
import glob

def fix_imports():
    files = glob.glob("src/nar/*.py")
    for file_path in files:
        if file_path.endswith("loader.py"):
            continue # Skip definition file
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        original_content = content
        
        # Replacements
        content = content.replace("JraVanDataLoader", "NarDataLoader")
        content = content.replace("from src.preprocessing", "from src.nar")
        content = content.replace("from preprocessing", "from src.nar")
        # Relative imports are fine if they exist, but .loader needs NarDataLoader
        
        if content != original_content:
            print(f"Fixing {file_path}...")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

if __name__ == "__main__":
    fix_imports()

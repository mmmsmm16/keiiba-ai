import sys, os
sys.path.append('src')
from inference.loader import InferenceDataLoader

loader = InferenceDataLoader()
df = loader.load(target_date='20251130')

if len(df) > 0:
    sample = df.iloc[0]
    print(f'Title: {sample.get("title", "N/A")}')
    print(f'Distance: {sample.get("distance", "N/A")}')
    print(f'Surface: {sample.get("surface", "N/A")}')
    print(f'State: {sample.get("state", "N/A")}')
    print(f'Weather: {sample.get("weather", "N/A")}')
    print(f'\nAll columns: {", ".join(df.columns.tolist())}')

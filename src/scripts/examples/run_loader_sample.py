import sys
import os
import logging

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from preprocessing.loader import RawDataLoader

logging.basicConfig(level=logging.INFO)

def main():
    try:
        loader = RawDataLoader()
        # limit=100 でテスト
        df = loader.load(limit=100)
        print("--- ロードされたデータ (先頭5件) ---")
        print(df.head())
        print(f"カラム一覧: {df.columns.tolist()}")
    except Exception as e:
        print(f"ロードテスト失敗 (DB未接続の可能性あり): {e}")

if __name__ == "__main__":
    main()

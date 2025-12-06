import pandas as pd
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='予測結果閲覧ツール')
    parser.add_argument('file', type=str, help='予測結果CSVファイルへのパス')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ファイルが見つかりません: {args.file}")
        sys.exit(1)

    try:
        df = pd.read_csv(args.file)
        
        # 整形表示
        # レースごとに表示
        races = df['race_id'].unique()
        
        print(f"=== {args.file} ===")
        print(f"対象レース数: {len(races)}")
        
        for rid in races:
            race_df = df[df['race_id'] == rid].sort_values('pred_rank')
            
            # メタデータ取得
            meta = race_df.iloc[0]
            print(f"\nRace ID: {rid} | {meta['venue']} {meta.get('race_number', '?')}R | {meta['date']}")
            
            # 上位5頭を表示
            cols = ['pred_rank', 'horse_number', 'horse_name', 'score', 'jockey_id']
            # カラムが存在するかチェック
            show_cols = [c for c in cols if c in race_df.columns]
            
            print(race_df[show_cols].head(5).to_string(index=False))
            print("-" * 40)

    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()

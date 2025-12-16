import pandas as pd
import pickle

# データ読み込み
df = pd.read_parquet('/workspace/data/processed/preprocessed_data_v10_leakfix.parquet')

# 欠損値情報の収集
missing_info = []

for col in df.columns:
    null_count = df[col].isna().sum()
    null_pct = (null_count / len(df)) * 100
    dtype = df[col].dtype
    
    if null_count > 0:
        missing_info.append({
            'column': col,
            'missing_count': null_count,
            'missing_pct': null_pct,
            'dtype': str(dtype),
            'non_null_count': len(df) - null_count
        })

# DataFrameに変換してソート
missing_df = pd.DataFrame(missing_info)
missing_df = missing_df.sort_values('missing_pct', ascending=False)

# 結果を出力
print("=" * 80)
print("欠損値情報 (欠損率が0%より大きいカラムのみ)")
print("=" * 80)
print(f"\n総カラム数: {len(df.columns)}")
print(f"欠損があるカラム数: {len(missing_df)}")
print(f"欠損なしカラム数: {len(df.columns) - len(missing_df)}")
print("\n")

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
    
    # CSV出力
    missing_df.to_csv('/workspace/missing_value_report.csv', index=False)
    print("\n\n✅ 詳細レポートを保存: /workspace/missing_value_report.csv")
else:
    print("全カラムに欠損値はありません。")

# 統計情報
print("\n" + "=" * 80)
print("数値カラムの基本統計 (min/maxで異常値チェック)")
print("=" * 80)

numeric_cols = df.select_dtypes(include=['int', 'float']).columns
stats_list = []

for col in numeric_cols[:30]:  # 最初の30個
    stats_list.append({
        'column': col,
        'min': df[col].min(),
        'max': df[col].max(),
        'mean': df[col].mean(),
        'std': df[col].std()
    })

stats_df = pd.DataFrame(stats_list)
print(stats_df.to_string(index=False))

import pickle
import pandas as pd

# LGBM学習用データセットを読み込み
with open('/workspace/data/processed/lgbm_datasets_v10_leakfix.pkl', 'rb') as f:
    datasets = pickle.load(f)

features = datasets['train']['X'].columns.tolist()

# 特徴量をカテゴリ別に分類
categories = {
    '基本情報 (レースメタデータ)': [],
    '過去走特徴 (Lag/Rolling)': [],
    'カテゴリ統計 (基本)': [],
    'コンテキスト統計 (組み合わせ)': [],
    '血統特徴': [],
    '展開・ペース特徴': [],
    '不利検出特徴': [],
    '相対的特徴': [],
    'リアルタイム特徴': [],
    '埋め込み特徴 (Embedding)': [],
    '経験値特徴': [],
    'レースレベル特徴': [],
    'その他': []
}

# 分類ロジック
for feat in features:
    if feat in ['race_number', 'distance', 'frame_number', 'horse_number', 'age', 'impost', 
                'weight_diff', 'sex_num', 'weather_num', 'surface_num', 'state_num', 
                'year', 'month', 'day', 'weekday', 'class_level', 'n_horses']:
        categories['基本情報 (レースメタデータ)'].append(feat)
    elif feat.startswith('lag1_') or feat.startswith('mean_') or feat.startswith('total_') or feat.startswith('wins_') or feat.startswith('win_rate_'):
        categories['過去走特徴 (Lag/Rolling)'].append(feat)
    elif any(x in feat for x in ['jockey_id_n_races', 'jockey_id_win_rate', 'jockey_id_top3_rate',
                                   'trainer_id_n_races', 'trainer_id_win_rate', 'trainer_id_top3_rate',
                                   'sire_id_n_races', 'sire_id_win_rate', 'sire_id_top3_rate',
                                   'class_level_n_races', 'class_level_win_rate', 'class_level_top3_rate']) and 'course' not in feat and 'dist' not in feat and 'surface' not in feat and 'recent' not in feat:
        categories['カテゴリ統計 (基本)'].append(feat)
    elif any(x in feat for x in ['_course_', '_dist_', '_surface_', '_trainer_', 'trainer_jockey']):
        categories['コンテキスト統計 (組み合わせ)'].append(feat)
    elif any(x in feat for x in ['sire_avg', 'sire_win_rate', 'sire_roi', 'sire_count', 'bms_']):
        categories['血統特徴'].append(feat)
    elif any(x in feat for x in ['nige', 'pace', 'race_avg', 'race_nige', 'interval', 'rest_score', 'momentum']):
        categories['展開・ペース特徴'].append(feat)
    elif any(x in feat for x in ['slow_start', 'pace_disadv', 'wide_run', 'track_bias', 'outer_frame', 'disadvantage']):
        categories['不利検出特徴'].append(feat)
    elif any(x in feat for x in ['_deviation', '_relative', '_race_rank']):
        categories['相対的特徴'].append(feat)
    elif feat.startswith('trend_'):
        categories['リアルタイム特徴'].append(feat)
    elif '_emb_' in feat:
        categories['埋め込み特徴 (Embedding)'].append(feat)
    elif any(x in feat for x in ['course_experience', 'course_best', 'distance_experience', 'distance_best',
                                   'first_', 'jockey_change', 'is_career_high']):
        categories['経験値特徴'].append(feat)
    elif any(x in feat for x in ['race_member', 'relative_strength']):
        categories['レースレベル特徴'].append(feat)
    else:
        categories['その他'].append(feat)

# Markdown形式でドキュメント生成
output = []
output.append("# 競馬AI: 学習データ特徴量一覧 (v10_leakfix)")
output.append("")
output.append(f"**総特徴量数**: {len(features)}")
output.append("")
output.append("---")
output.append("")

for cat_name, cat_features in categories.items():
    if len(cat_features) > 0:
        output.append(f"## {cat_name} ({len(cat_features)}個)")
        output.append("")
        for i, feat in enumerate(cat_features, 1):
            output.append(f"{i}. `{feat}`")
        output.append("")

# ファイルに保存
with open('/workspace/feature_list_v10_leakfix.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print("✅ 特徴量リストを生成しました: /workspace/feature_list_v10_leakfix.md")
print(f"総特徴量数: {len(features)}")
print("\nカテゴリ別内訳:")
for cat_name, cat_features in categories.items():
    if len(cat_features) > 0:
        print(f"  - {cat_name}: {len(cat_features)}個")

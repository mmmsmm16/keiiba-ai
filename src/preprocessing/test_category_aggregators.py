import pandas as pd
import pytest
from preprocessing.category_aggregators import CategoryAggregator

class TestCategoryAggregator:
    def test_aggregate_leakage_within_same_race(self):
        """
        同一レースに同一カテゴリ（例：調教師）の馬が複数出走する場合、
        処理順序によって「カンニング（未来情報の参照）」が発生しないかテストする。
        """
        # データ作成: 2つのレース
        # Race 1: Trainer T1 の馬が2頭 (H1, H2) 出走。H1が1着。
        # Race 2: Trainer T1 の馬が1頭 (H3) 出走。
        data = {
            'date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02']),
            'race_id': ['R1', 'R1', 'R2'],
            'horse_id': ['H1', 'H2', 'H3'],
            'trainer_id': ['T1', 'T1', 'T1'],
            'jockey_id': ['J1', 'J2', 'J3'], # ダミー
            'sire_id': ['S1', 'S2', 'S3'],   # ダミー
            'rank': [1, 2, 5] # H1が1着
        }
        df = pd.DataFrame(data)

        # 集計実行
        agg = CategoryAggregator()
        df_res = agg.aggregate(df)

        # 検証1: Race 1 の時点では、T1の過去成績は0のはず（初出走と仮定）
        # H1もH2も、このレースの結果を知ってはならない。
        r1_h1 = df_res[(df_res['race_id'] == 'R1') & (df_res['horse_id'] == 'H1')].iloc[0]
        r1_h2 = df_res[(df_res['race_id'] == 'R1') & (df_res['horse_id'] == 'H2')].iloc[0]

        # 期待値: n_races=0, win_rate=0
        # リークがある場合、H2はH1の結果（1着）を見てしまうため n_races=1, win_rate=1.0 になる可能性がある
        assert r1_h1['trainer_id_n_races'] == 0, "H1 should have 0 history"
        assert r1_h2['trainer_id_n_races'] == 0, "H2 should have 0 history (Leakage Check)"
        assert r1_h2['trainer_id_win_rate'] == 0.0, "H2 should have 0% win rate (Leakage Check)"

        # 検証2: Race 2 の時点では、Race 1の結果（2走中1勝）が反映されているべき
        r2_h3 = df_res[df_res['race_id'] == 'R2'].iloc[0]
        assert r2_h3['trainer_id_n_races'] == 2, "H3 should see 2 past races from T1"
        assert r2_h3['trainer_id_win_rate'] == 0.5, "H3 should see 50% win rate (1/2)"

    def test_aggregate_handles_missing_columns(self):
        """必須カラムが欠けていてもエラーにならないか（Warningでスキップするか）"""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01']),
            'race_id': ['R1'],
            'rank': [1]
            # trainer_id, jockey_id 等なし
        })
        agg = CategoryAggregator()
        # エラーが出なければOK
        df_res = agg.aggregate(df)
        assert 'trainer_id_n_races' not in df_res.columns

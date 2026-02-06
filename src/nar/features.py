import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class NarFeatureGenerator:
    """
    地方競馬（NAR）の特徴量生成クラス。
    Just-In-Time (JIT) 方式で、指定されたレース時点での過去実績を集計します。
    """
    
    def __init__(self, history_windows: list = None):
        """
        Args:
            history_windows: 集計する過去走の窓サイズリスト（例: [1, 3, 5]）
        """
        self.history_windows = history_windows or [1, 3, 5]

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        与えられた生データ DataFrame に対して特徴量を付与します。
        """
        logger.info(f"特徴量生成を開始: {len(df)} レコード")
        
        # 日付を datetime 型に変換
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # 1. 基本的なデータ型の変換とソート
        # 重要: 日付と馬IDでソートを確実にし、馬ごとの時系列を崩さない
        df = df.sort_values(['horse_id', 'date', 'race_id']).copy()
        
        # 2. 馬ごとの過去走集計 (JIT)
        df = self._add_horse_history_features(df)
        
        # 3. スピード指数の計算 (JIT)
        df = self._add_speed_index_features(df)
        
        # 4. 騎手・調教師の統計量
        df = self._add_human_stats_features(df)

        # 5. [NEW] 馬の属性・休養などの状態
        df = self._add_horse_state_features(df)

        # 6. [NEW] 人馬の相性（Chemistry）
        df = self._add_chemistry_features(df)

        # 7. [NEW] レース適性（Aptitude）
        df = self._add_aptitude_features(df)

        # 8. [NEW] 調教師の勢い（Momentum）- 直近30日間の成績
        df = self._add_trainer_momentum_features(df)

        # 9. [NEW] 改善ロジックに基づく特徴量（斤量差、不利フラグ、減衰履歴）
        df = self._add_refined_logic_features(df)

        # 10. [NEW] クラス特徴量（昇級・降級）
        df = self._add_class_features(df)
        
        # 11. [NEW] レース内相対指標 (Relative Field Strength)
        df = self._add_relative_field_features(df)

        # 12. [NEW] 状況適性 (Situational Aptitude)
        df = self._add_situational_aptitude_features(df)

        # 13. [NEW] 馬場バイアス (Real-time Track Bias)
        df = self._add_realtime_track_bias_features(df)

        # 14. [NEW] 血統特徴量 (Pedigree - Sire Stats)
        df = self._add_pedigree_features(df)

        logger.info("特徴量生成が完了しました。")
        return df

    def _add_trainer_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        調教師の直近30日間の成績を付与します。
        """
        logger.info("調教師の直近勢い特徴量を付与中...")
        
        # 日付順にソート（必須）
        sorted_df = df.sort_values(['date', 'race_id'])
        
        # 効率化のため、一意の (trainer_id, date, race_id) に対して成績を集計
        race_results = sorted_df[['trainer_id', 'date', 'race_id', 'rank']].drop_duplicates(['trainer_id', 'date', 'race_id'])
        
        # 調教師ごとにローリング集計
        # 日付でソート
        race_results = race_results.sort_values('date')
        
        # グループ化して apply
        # group_keys=True にして trainer_id をインデックスに残す
        def calc_rolling_stats(group):
            # rolling のために date を index にするが、重複がありうる
            # 重複がある場合、rolling('30D') は動作するが、結果のassignでindex alignment問題が起きる可能性がある
            # ここではシンプルに、dateをindexにして計算し、valuesを取得する
            g = group.set_index('date').sort_index()
            wins = (g['rank'] == 1).rolling('30D', closed='left').sum()
            runs = (g['rank']).rolling('30D', closed='left').count()
            
            # 結果を元の行と紐付けるために、元のindexを使用するか、
            # あるいは race_id を維持する必要がある
            res = pd.DataFrame({
                'trainer_30d_win_rate': (wins.values / runs.values),
                'race_id': g['race_id'].values,
                'trainer_id': group['trainer_id'].iloc[0] # 念のため
            })
            return res.fillna(0) # 0割りなどでNaNなら0に
            
        # apply の結果は index がリセットされるか、multi-index になる
        momentum_stats = race_results.groupby('trainer_id', group_keys=False).apply(calc_rolling_stats)
        
        df = df.merge(
            momentum_stats[['trainer_id', 'race_id', 'trainer_30d_win_rate']],
            on=['trainer_id', 'race_id'],
            how='left'
        )
        
        return df

    def _add_horse_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        馬齢、性別、休養期間、馬体重増減などの状態に関する特徴量を付与します。
        """
        logger.info("馬の状態・属性特徴量を付与中...")
        
        # 1. 性別・馬齢の処理
        # NarDataLoader で 'sex' と 'age' が構築されている前提
        if 'gender' not in df.columns and 'sex' in df.columns:
            df['gender'] = df['sex']
        
        df['age'] = pd.to_numeric(df.get('age', 0), errors='coerce')
        
        grouped = df.groupby('horse_id')
        
        # 2. 休養期間 (前走からの日数)
        df['days_since_prev_race'] = grouped['date'].diff().dt.days
        
        # 3. 馬体重増減
        # weight は数値型に変換しておく
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        df['weight_diff'] = grouped['weight'].diff()
        
        return df

    def _add_chemistry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        騎手と馬の相性（コンビ実績）を付与します。
        """
        logger.info("人馬の相性特徴量を付与中...")
        
        # 馬×騎手のペアでグループ化
        # 日付順に累積を計算
        pair_grouped = df.sort_values(['date', 'race_id']).groupby(['horse_id', 'jockey_id'])
        
        # コンビでの累積3着以内数
        cum_pair_places = pair_grouped['rank'].transform(lambda x: (x <= 3).shift(1).cumsum())
        # コンビでの累積騎乗回数
        cum_pair_runs = pair_grouped.cumcount()
        
        df['horse_jockey_place_rate'] = cum_pair_places / cum_pair_runs.replace(0, np.nan)
        # 継続騎乗フラグ (前走と騎手が同じか)
        df['is_consecutive_jockey'] = (df['jockey_id'] == df.groupby('horse_id')['jockey_id'].shift(1)).astype(int)
        
        return df

    def _add_aptitude_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        距離変更やコース実績などの適性に関する特徴量を付与します。
        """
        logger.info("適性特徴量を付与中...")
        
        grouped = df.groupby('horse_id')
        
        # 1. 距離変更 (今回 - 前走)
        df['distance_diff'] = grouped['distance'].diff()
        
        # 2. コース（競馬場）実績
        # 馬×競馬場での累積成績
        venue_grouped = df.sort_values(['date', 'race_id']).groupby(['horse_id', 'venue'])
        cum_venue_places = venue_grouped['rank'].transform(lambda x: (x <= 3).shift(1).cumsum())
        cum_venue_runs = venue_grouped.cumcount()
        
        df['horse_venue_place_rate'] = cum_venue_places / cum_venue_runs.replace(0, np.nan)
        
        return df

    def _add_horse_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        各馬の過去n戦の成績を集計します。
        """
        logger.info(f"馬の過去走実績 ({self.history_windows}戦) を集計中...")
        
        # 集計対象のカラム
        target_cols = ['rank', 'popularity', 'odds']
        
        # 馬ごとにグループ化（ソート済み前提）
        grouped = df.groupby('horse_id')
        
        for n in self.history_windows:
            for col in target_cols:
                new_col_name = f'horse_prev{n}_{col}_avg'
                # shift(1) で「今走」を含まないようにし、rolling(n) で集計
                # min_periods=1 とすることで、過去走がn戦に満たなくても存在する分だけで計算
                df[new_col_name] = grouped[col].transform(
                    lambda x: x.shift(1).rolling(window=n, min_periods=1).mean()
                )
        
        # 出走回数もカウント
        df['horse_run_count'] = grouped.cumcount()
        
        return df

    def _add_speed_index_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        スピード指数を計算し、その過去成績を付与します。
        基準タイムは、そのレース時点までの同一 (競馬場, 距離, 馬場状態) の中央値を使用します。
        """
        logger.info("スピード指数を計算中...")
        
        # 1. 基準タイムの計算 (expanding median)
        # リーク防止のため、今走を含まない過去データから算出
        # (会場, 距離, 馬場状態) ごとにグループ化
        # まずは日付でソートされていることを前提にする
        group_keys = ['venue', 'distance', 'state']
        
        # 補助的に (会場, 距離) だけのグループも作成（馬場状態が初出の場合のフォールバック）
        df['baseline_vds'] = df.groupby(group_keys)['time'].transform(
            lambda x: x.shift(1).expanding().median()
        )
        df['baseline_vd'] = df.groupby(['venue', 'distance'])['time'].transform(
            lambda x: x.shift(1).expanding().median()
        )
        
        # フォールバック適用
        df['baseline_final'] = df['baseline_vds'].fillna(df['baseline_vd'])
        
        # 2. 指数計算: (基準 - 走破タイム) * 係数 + 80
        # 係数は暫定で 1.0 (1秒 = 1ポイント) とします
        df['speed_index'] = (df['baseline_final'] - df['time']) * 1.0 + 80
        
        # 欠損値（過去に同一条件がない場合など）は 80 (平均的) で埋める
        df['speed_index'] = df['speed_index'].fillna(80)
        
        # 3. 馬ごとの過去走スピード指数を集計
        grouped = df.groupby('horse_id')
        for n in self.history_windows:
            new_col_name = f'horse_prev{n}_si_avg'
            df[new_col_name] = grouped['speed_index'].transform(
                lambda x: x.shift(1).rolling(window=n, min_periods=1).mean()
            )
            
        # 4. 不要な中間カラムを削除
        df.drop(columns=['baseline_vds', 'baseline_vd', 'baseline_final'], inplace=True)
        
        return df

    def _add_human_stats_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        騎手・調教師の統計量を付与します。
        """
        logger.info("人間系の統計量を付与中...")
        
        for human_id in ['jockey_id', 'trainer_id']:
            prefix = 'jockey' if human_id == 'jockey_id' else 'trainer'
            
            # 日付順にソートして累積を計算（リーク防止）
            sorted_df = df.sort_values(['date', 'race_id'])
            grouped = sorted_df.groupby(human_id)
            
            # 1着数・3着以内数の累積
            cum_wins = grouped['rank'].transform(lambda x: (x == 1).shift(1).cumsum())
            cum_places = grouped['rank'].transform(lambda x: (x <= 3).shift(1).cumsum())
            
            # 累積走数
            cum_runs = grouped.cumcount()
            
            # 累積率の算出
            win_rate = cum_wins / cum_runs.replace(0, np.nan)
            place_rate = cum_places / cum_runs.replace(0, np.nan)

            df[f'{prefix}_win_rate'] = win_rate.astype(float)
            df[f'{prefix}_place_rate'] = place_rate.astype(float)

        return df

    def _add_refined_logic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        斤量差、不利（故障・出遅れ等）フラグ、過去走の減衰重み付け特徴量を付与します。
        """
        logger.info("改善ロジックに基づく特徴量を付与中...")

        grouped = df.groupby('horse_id')

        # 1. 斤量差 (今回 - 前走)
        df['impost_diff'] = grouped['impost'].diff()

        # 2. 前走の異常・不利フラグ (再教育や失格など、正常終了以外を 1 とする)
        # accident_code: 0=なし, 1=取消, 2=除外, 3=中止, 4=失格, 5=再教育 等
        # ここでは「出走したが不利があった」ではなく「走りに異常があった」履歴として抽出
        if 'accident_code' in df.columns:
            # 正常(0)以外かつ、欠損でない場合を異常とする
            df['is_accident_tmp'] = (pd.to_numeric(df['accident_code'], errors='coerce').fillna(0) > 0).astype(int)
            df['was_accident_prev1'] = grouped['is_accident_tmp'].shift(1).fillna(0)
            df.drop(columns=['is_accident_tmp'], inplace=True)

        # 3. 過去走の減衰重み付け (1走前=1.0, 2走前=0.5, 3走前=0.25 ...)
        # 直近2走を重視し、それ以上を減衰させる
        def weighted_momentum(x, weights):
            # x は馬ごとの系列、weights は重みのリスト
            # shift(1) して今回を含まないようにする
            s = x.shift(1)
            res = 0.0
            total_w = 0.0
            for i, w in enumerate(weights):
                val = s.shift(i)
                # NaN でない場合のみ加算
                valid_mask = val.notna()
                res += val.fillna(0) * w
                total_w += valid_mask.astype(float) * w
            return res / total_w.replace(0, np.nan)

        # スピード指数と着順に対して適用
        decay_weights = [1.0, 0.5, 0.25, 0.125, 0.06] # 1〜5走前
        df['weighted_si_momentum'] = grouped['speed_index'].transform(lambda x: weighted_momentum(x, decay_weights))
        df['weighted_rank_momentum'] = grouped['rank'].transform(lambda x: weighted_momentum(x, decay_weights))

        return df

    def _add_class_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        レースのクラスを解析し、前走からの昇級・降級フラグを付与します。
        """
        logger.info("クラス特徴量（昇級・降級）を付与中...")

        def parse_class_rank(name):
            if pd.isna(name): return 0
            name = str(name)
            # 全角を半角に変換（簡易版）
            name = name.translate(str.maketrans('ＡＢＣ１２３４５６７８９０', 'ABC1234567890'))
            
            if 'A1' in name: return 10
            if 'A2' in name: return 9
            if 'B1' in name: return 8
            if 'B2' in name: return 7
            if 'B3' in name: return 6
            if 'C1' in name: return 5
            if 'C2' in name: return 4
            if 'C3' in name: return 3
            if '3歳' in name: return 2
            if '2歳' in name: return 1
            return 0

        # クラスランクの付与
        df['class_rank'] = df['class_name'].apply(parse_class_rank)

        # 前走のクラスランク
        grouped = df.groupby('horse_id')
        df['prev_class_rank'] = grouped['class_rank'].shift(1).fillna(df['class_rank'])
        
        # クラス差（昇級・降級）
        df['class_diff'] = df['class_rank'] - df['prev_class_rank']
        df['is_promoted'] = (df['class_diff'] > 0).astype(int)
        df['is_demoted'] = (df['class_diff'] < 0).astype(int)

        return df

    def _add_relative_field_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        レース内での各馬の相対的な立ち位置（偏差値、順位）を付与します。
        """
        logger.info("レース内相対特徴量を付与中...")
        
        # 適用対象のカラム
        target_cols = ['weighted_si_momentum', 'weighted_rank_momentum', 'class_rank']
        
        # レースごとの統計量を用いて正規化
        def calc_relative(group):
            for col in target_cols:
                if col in group.columns:
                    # 順位 (高いほど1位)
                    group[f'{col}_race_rank'] = group[col].rank(ascending=False, method='min')
                    # 偏差値的なもの (平均との差)
                    mean_val = group[col].mean()
                    std_val = group[col].std()
                    group[f'{col}_diff_from_avg'] = group[col] - mean_val
                    if not pd.isna(std_val) and std_val > 0:
                        group[f'{col}_zscore'] = (group[col] - mean_val) / std_val
                    else:
                        group[f'{col}_zscore'] = 0
            return group

        df = df.groupby('race_id', group_keys=False).apply(calc_relative)
        return df

    def _add_situational_aptitude_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        馬場状態や季節などの特定の状況下での適性を付与します。
        """
        logger.info("状況適性特徴量を付与中...")

        # 1. 馬場状態別実績 (馬 × 馬場状態)
        df['state_group'] = df['state'].fillna('不明')
        
        # 馬 × 馬場状態ペアでの累積
        state_grouped = df.sort_values(['date', 'race_id']).groupby(['horse_id', 'state_group'])
        cum_state_places = state_grouped['rank'].transform(lambda x: (x <= 3).shift(1).cumsum())
        cum_state_runs = state_grouped.cumcount()
        
        df['horse_state_place_rate'] = cum_state_places / cum_state_runs.replace(0, np.nan)
        
        # 2. 季節特徴量
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].map(lambda x: 1 if x in [12, 1, 2] else (2 if x in [3, 4, 5] else (3 if x in [6, 7, 8] else 4)))
        
        # 3. ナイター判定
        df['is_night_race'] = df['title'].str.contains('ナイター').fillna(False).astype(int)
        
        # 人間系の勢いバイアス
        if 'trainer_30d_win_rate' in df.columns and 'trainer_win_rate' in df.columns:
            df['trainer_momentum_bias'] = df['trainer_30d_win_rate'] - df['trainer_win_rate']

        return df

    def _add_realtime_track_bias_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        馬場バイアス（内枠有利、前有利など）をリアルタイム（当日・本レース前まで）で集計します。
        """
        logger.info("馬場バイアス特徴量（リアルタイム）を付与中...")

        # 1. 必要なカラムだけ抽出して、レースごとの結果サマリを作成
        # レース順にソート（必須）
        temp_df = df[['date', 'venue', 'race_id', 'horse_number', 'frame_number', 'rank', 'pass_1']].copy()
        temp_df = temp_df.drop_duplicates(['race_id', 'horse_number'])
        
        # 内枠(1-3枠)フラグ, 外枠(6-8枠)フラグ
        temp_df['is_inner'] = temp_df['frame_number'].isin([1, 2, 3]).astype(int)
        temp_df['is_outer'] = temp_df['frame_number'].isin([6, 7, 8]).astype(int)
        
        # 逃げ・先行フラグ (pass_1が 1 or 2 or 3)
        # pass_1 は '1-1-1' のような文字列だったり '1' だったりする
        def is_front_runner(val):
            if pd.isna(val): return 0
            val = str(val).split('-')[0] # 最初の通過順
            try:
                rank = int(val)
                return 1 if rank <= 3 else 0
            except:
                return 0
        temp_df['is_front'] = temp_df['pass_1'].apply(is_front_runner)

        # レースごとの「勝った馬の属性」を集計
        # race_id ごとに、「内枠が勝ったか」「前が勝ったか」を判定
        # rank=1 の行だけ抽出
        winners = temp_df[temp_df['rank'] == 1].copy()
        
        # 同着ありの場合、race_id が重複するので groupby して集計 (maxをとれば 1つでも内枠/前がいれば1になる)
        winners_agg = winners.groupby('race_id')[['is_inner', 'is_outer', 'is_front']].max()
        
        # 全レースIDを持つベースを作成 (勝者なしレースも考慮)
        race_summary = temp_df[['date', 'venue', 'race_id']].drop_duplicates('race_id').set_index('race_id')
        
        race_summary['inner_won'] = winners_agg['is_inner']
        race_summary['outer_won'] = winners_agg['is_outer']
        race_summary['front_won'] = winners_agg['is_front']
        
        # NaN (勝者データなし) は 0埋め
        race_summary = race_summary.fillna(0)
        
        # 日付・昇順でソート
        race_summary = race_summary.sort_values(['date', 'venue', 'race_id'])
        
        # 2. 当日・同会場内での累積を計算 (Shiftしてリーク防止)
        # グループ: (date, venue)
        grouped = race_summary.groupby(['date', 'venue'])
        
        # 当日のここまでのレース数 (自分を含まない)
        race_summary['daily_race_count'] = grouped.cumcount()
        
        # 内枠が勝った回数の累積 (Shift 1)
        race_summary['cum_inner_wins'] = grouped['inner_won'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
        race_summary['track_bias_inner_win_rate'] = race_summary['cum_inner_wins'] / race_summary['daily_race_count'].replace(0, np.nan)
        
        # 外枠が勝った回数
        race_summary['cum_outer_wins'] = grouped['outer_won'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
        race_summary['track_bias_outer_win_rate'] = race_summary['cum_outer_wins'] / race_summary['daily_race_count'].replace(0, np.nan)

        # 前（逃げ先行）が勝った回数
        race_summary['cum_front_wins'] = grouped['front_won'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
        race_summary['track_bias_front_win_rate'] = race_summary['cum_front_wins'] / race_summary['daily_race_count'].replace(0, np.nan)
        
        # 3. 元のデータフレームにマージ
        target_cols = ['track_bias_inner_win_rate', 'track_bias_outer_win_rate', 'track_bias_front_win_rate']
        race_summary = race_summary.reset_index() # race_id を列に戻す
        
        df = df.merge(race_summary[['race_id'] + target_cols], on='race_id', how='left')
        
        # 1レース目は NaN になるので 0.0 (または global mean) で埋める
        # ここでは「バイアスなし」とみなして 0.0 にする
        df[target_cols] = df[target_cols].fillna(0.0)
        
        return df

    def _add_pedigree_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        血統特徴量（種牡馬: sire_id）のターゲットエンコーディングを行います。
        リーク防止のため、Expanding Mean (過去の累積成績) を使用します。
        """
        logger.info("血統特徴量 (Sire) を付与中...")
        
        if 'sire_id' not in df.columns:
            return df
            
        # 1. 必要なカラムを抽出してソート
        df.sort_values(['date', 'race_id'], inplace=True)
        
        # 2. Sire ごとの累積成績を計算
        # group_keys=False is important to keep the original index or alignment
        # expanding().mean() works on numerical, so we convert rank conditions to int
        
        # rank 1 win
        # rank 3 place
        # To avoid slow groupby().expanding(), we can use transform with shift
        
        grouped = df.groupby('sire_id')
        
        # 勝数累積
        cum_wins = grouped['rank'].transform(lambda x: (x == 1).shift(1).cumsum())
        # 複勝数累積
        cum_places = grouped['rank'].transform(lambda x: (x <= 3).shift(1).cumsum())
        # 出走数累積
        cum_runs = grouped.cumcount()
        
        # 勝率・複勝率
        # 最低5戦くらいしていないと信頼性がない -> smooth target encoding?
        # ここでは単純に expanding mean とするが、分母が小さい場合は0かglobal meanにするのが定石
        # 今回は 0.0 で埋め、特徴量重要度で判断させる
        
        df['sire_win_rate'] = (cum_wins / cum_runs.replace(0, np.nan)).fillna(0.0)
        df['sire_place_rate'] = (cum_places / cum_runs.replace(0, np.nan)).fillna(0.0)
        
        return df

if __name__ == "__main__":
    # 簡易テスト
    logging.basicConfig(level=logging.INFO)
    gen = NarFeatureGenerator()
    print("NarFeatureGenerator initialized.")

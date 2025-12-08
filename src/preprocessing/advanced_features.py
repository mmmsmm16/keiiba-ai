import pandas as pd
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    レース展開や血統など、より高度なドメイン知識に基づく特徴量を生成するクラス。
    """
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        高度な特徴量を追加します。

        Args:
            df (pd.DataFrame): 前処理済みのデータ。

        Returns:
            pd.DataFrame: 特徴量が追加されたデータ。
        """
        logger.info("高度特徴量の生成を開始...")

        # 1. 逃げ判定 (Is Nige)
        # passing_rank (通過順) は "1-1-1" や "10-10-9" のような文字列
        # 最初の数字が 1 なら「逃げた」と判定
        def is_nige(s):
            if not isinstance(s, str): return 0
            try:
                # 数字以外の文字が含まれる場合もあるので注意
                first_pos = s.split('-')[0]
                # "1(2)" のようなケースもあるかも？数値変換できるか試す
                if first_pos.isdigit():
                    return 1 if int(first_pos) == 1 else 0
                return 0
            except:
                return 0

        # passing_rankカラムが存在し、有効なデータがあるかチェック
        # 推論時はpassing_rankがNULLなので、その場合は0で初期化
        if 'passing_rank' in df.columns and df['passing_rank'].notna().any():
            # 一時的なカラムとして作成
            df['is_nige_temp'] = df['passing_rank'].apply(is_nige)
        else:
            # passing_rankがない場合（推論時など）は0で初期化
            logger.info("passing_rankが利用できないため、is_nige_tempを0で初期化します")
            df['is_nige_temp'] = 0


        # 2. 馬ごとの過去の逃げ率 (Nige Rate)
        # 時系列順にソートして集計
        df = df.sort_values(['horse_id', 'date'])
        grouped_horse = df.groupby('horse_id')

        # shift(1)してexpanding meanを計算 (リーク防止)
        df['nige_rate'] = grouped_horse['is_nige_temp'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)

        # ----------------------------------------------------------------
        # 3. 間隔 (Interval) と体重変化 (Weight Change)
        # ----------------------------------------------------------------
        # dateはdatetime型であることを前提
        df['prev_date'] = grouped_horse['date'].shift(1)
        df['interval'] = (df['date'] - df['prev_date']).dt.days
        df['interval'] = df['interval'].fillna(0) # 初出走は0
        
        # 長期休養明けフラグ (180日以上)
        df['is_long_break'] = (df['interval'] > 180).astype(int)

        # 体重増減 (batai_taiju カラムがある場合)
        if 'batai_taiju' in df.columns:
            # batai_taiju が文字列の場合もあるので数値変換 ('480(0)' -> 480 or clean_batai_taiju exists?)
            # Usually cleansed in DataCleanser. Assuming float/int.
            df['prev_weight'] = grouped_horse['batai_taiju'].shift(1)
            df['weight_diff'] = df['batai_taiju'] - df['prev_weight']
            df['weight_diff'] = df['weight_diff'].fillna(0)
            
            # 大幅増減フラグ (+- 10kg)
            df['is_weight_changed_huge'] = (df['weight_diff'].abs() > 10).astype(int)

        # ----------------------------------------------------------------
        # 4. 騎手の近走勢い (Jockey Recent Momentum)
        # ----------------------------------------------------------------
        # 時系列ソート (全体)
        df = df.sort_values(['date', 'race_id'])
        
        # 騎手ごとの直近100レース勝率
        # (注: これは未来のデータを含まないよう shift(1) してから rolling する必要がある)
        if 'jockey_id' in df.columns:
            # 勝利フラグ (1着なら1)
            df['is_win'] = (df['rank'] == 1).astype(int)
            
            # グループ化して shift(1) -> rolling(100) -> mean
            # 処理が重くなる可能性があるので注意。
            # transformで実装
            df['jockey_recent_win_rate'] = df.groupby('jockey_id')['is_win'].transform(
                lambda x: x.shift(1).rolling(100, min_periods=10).mean()
            ).fillna(0)
            
            df.drop(columns=['is_win'], inplace=True)

        # ----------------------------------------------------------------
        # 5. レース展開・レベル予測 (Race Context)
        # ----------------------------------------------------------------
        grouped_race = df.groupby('race_id')

        # メンバーの平均逃げ率
        df['race_avg_nige_rate'] = grouped_race['nige_rate'].transform('mean')

        # 逃げ馬候補の数 (逃げ率 0.5 以上の馬の数)
        df['race_nige_horse_count'] = grouped_race['nige_rate'].transform(lambda x: (x >= 0.5).sum())
        
        # 逃げ馬割合 (Nige Bias) - メンバー中の逃げ馬の比率
        # count() はグループサイズ
        df['race_nige_bias'] = df['race_nige_horse_count'] / grouped_race['race_id'].transform('count')

        # ペース予測 (Pace Category)
        # 平均逃げ率が高い -> H (High)
        # 低い -> S (Slow)
        # しきい値は要調整だが、一旦分位点などで分けるか、固定値で。
        # 0.2以下: S, 0.2-0.4: M, 0.4以上: H
        def categorize_pace(x):
            if x < 0.2: return 0 # Slow
            elif x < 0.4: return 1 # Middle
            else: return 2 # High
        
        df['race_pace_cat'] = df['race_avg_nige_rate'].apply(categorize_pace)

        # メンバーの平均獲得賞金 (本賞金) -> レースレベルの代理変数
        # HistoryAggregatorで total_prize (過去の獲得賞金累計) が計算されていることを前提
        if 'total_prize' in df.columns:
             df['race_avg_prize'] = grouped_race['total_prize'].transform('mean')
        
        # メンバーの平均年齢 (Member Age Level) - 若馬戦か古馬戦かなど
        if 'age' in df.columns:
            # 必ず数値型に変換
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['race_avg_age'] = grouped_race['age'].transform('mean')

        # ================================================================
        # 6. 新規特徴量 (v6 Feature Engineering)
        # ================================================================
        logger.info("v6 新規特徴量を生成中...")
        
        # 6.1 出走頭数 (n_horses) - レースの難易度指標
        df['n_horses'] = grouped_race['race_id'].transform('count')
        
        # 6.2 枠番ゾーン (frame_zone) - 内枠/中枠/外枠の分類
        # frame_number: 1-8
        if 'frame_number' in df.columns:
            df['frame_number'] = pd.to_numeric(df['frame_number'], errors='coerce')
            def frame_to_zone(f):
                if pd.isna(f): return 1
                if f <= 2: return 0  # 内枠
                elif f <= 6: return 1  # 中枠
                else: return 2  # 外枠
            df['frame_zone'] = df['frame_number'].apply(frame_to_zone)
        
        # 6.3 直近3走の平均着順 (recent_3_avg_rank)
        # lag1_rank, lag2_rank, lag3_rank が存在する前提
        lag_cols = ['lag1_rank', 'lag2_rank', 'lag3_rank']
        existing_lag_cols = [c for c in lag_cols if c in df.columns]
        if len(existing_lag_cols) >= 2:
            df['recent_3_avg_rank'] = df[existing_lag_cols].mean(axis=1, skipna=True)
            df['recent_3_avg_rank'] = df['recent_3_avg_rank'].fillna(9.0)  # デフォルト: 中間値
            
            # 6.4 直近3走の勝率 (recent_3_win_rate)
            def calc_recent_wins(row):
                wins = 0
                count = 0
                for col in existing_lag_cols:
                    if pd.notna(row[col]):
                        count += 1
                        if row[col] == 1:
                            wins += 1
                return wins / count if count > 0 else 0
            df['recent_3_win_rate'] = df.apply(calc_recent_wins, axis=1)
        
        # 6.5 騎手×距離カテゴリ別勝率 (距離適性)
        # 距離カテゴリ: 短距離(<1400), マイル(1400-1800), 中距離(1800-2200), 長距離(2200+)
        if 'distance' in df.columns and 'jockey_id' in df.columns:
            df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
            def distance_category(d):
                if pd.isna(d): return 1
                if d < 1400: return 0  # 短距離
                elif d < 1800: return 1  # マイル
                elif d < 2200: return 2  # 中距離
                else: return 3  # 長距離
            df['distance_category'] = df['distance'].apply(distance_category)
            
            # 騎手×距離カテゴリ別の過去勝率 (時系列リーク防止: shift+expanding)
            df = df.sort_values(['jockey_id', 'distance_category', 'date'])
            df['is_win_temp'] = (df['rank'] == 1).astype(int)
            df['jockey_distance_winrate'] = df.groupby(['jockey_id', 'distance_category'])['is_win_temp'].transform(
                lambda x: x.shift(1).expanding().mean()
            ).fillna(0)
            df.drop(columns=['is_win_temp'], inplace=True)
        
        # 6.6 枠番×芝/ダート別勝率
        if 'frame_number' in df.columns and 'surface_num' in df.columns:
            df['is_win_temp'] = (df['rank'] == 1).astype(int)
            df = df.sort_values(['frame_number', 'surface_num', 'date'])
            df['frame_surface_winrate'] = df.groupby(['frame_number', 'surface_num'])['is_win_temp'].transform(
                lambda x: x.shift(1).expanding().mean()
            ).fillna(0.05)  # デフォルト: 5%
            df.drop(columns=['is_win_temp'], inplace=True)
        
        # 6.7 相対的人気順位 (レース内での人気の相対位置)
        # lag1_popularity を使用 (前走の人気)
        if 'lag1_popularity' in df.columns:
            df['relative_popularity_rank'] = grouped_race['lag1_popularity'].rank(method='min', ascending=True)
            df['relative_popularity_rank'] = df['relative_popularity_rank'].fillna(df['n_horses'] / 2)
        
        # 6.8 馬の連対率 (過去の1-2着率)
        if 'mean_rank_all' in df.columns:
            # mean_rank_all が低いほど連対率が高いと推定
            df['estimated_place_rate'] = 1 / (df['mean_rank_all'] + 1)
            df['estimated_place_rate'] = df['estimated_place_rate'].fillna(0.1)
             
        # 不要な一時カラムを削除
        top_drop_cols = ['is_nige_temp', 'prev_date', 'prev_weight']
        df.drop(columns=[c for c in top_drop_cols if c in df.columns], inplace=True)

        logger.info("高度特徴量の生成完了")
        return df

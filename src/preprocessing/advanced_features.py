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
        
        Note:
            メモリ効率化のため、不要になった一時カラムは即座に削除し、
            可能な限り float32 へのダウンキャストを行います。
        """
        logger.info("高度特徴量の生成を開始...")
        
        # メモリ節約用ヘルパー
        def downcast_floats(d: pd.DataFrame):
            cols = d.select_dtypes(include=['float64']).columns
            if len(cols) > 0:
                d[cols] = d[cols].astype('float32')
            return d

        # 1. 逃げ判定 (Is Nige)
        def is_nige(s):
            if not isinstance(s, str): return 0
            try:
                first_pos = s.split('-')[0]
                if first_pos.isdigit():
                    return 1 if int(first_pos) == 1 else 0
                return 0
            except:
                return 0

        if 'passing_rank' in df.columns and df['passing_rank'].notna().any():
            df['is_nige_temp'] = df['passing_rank'].apply(is_nige).astype('int8')
        else:
            logger.info("passing_rankが利用できないため、is_nige_tempを0で初期化します")
            df['is_nige_temp'] = 0

        # 2. 馬ごとの過去の逃げ率 (Nige Rate)
        df = df.sort_values(['horse_id', 'date'])
        
        # memory opt: use transform directly without intermediary if possible
        df['nige_rate'] = df.groupby('horse_id')['is_nige_temp'].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0).astype('float32')

        # ----------------------------------------------------------------
        # 3. 間隔 (Interval) と体重変化 (Weight Change)
        # ----------------------------------------------------------------
        df['prev_date'] = df.groupby('horse_id')['date'].shift(1)
        df['interval'] = (df['date'] - df['prev_date']).dt.days.fillna(0).astype('float32')
        
        df['is_long_break'] = (df['interval'] > 180).astype('int8')

        if 'batai_taiju' in df.columns:
            # batai_taiju should be numeric
            df['batai_taiju'] = pd.to_numeric(df['batai_taiju'], errors='coerce')
            df['prev_weight'] = df.groupby('horse_id')['batai_taiju'].shift(1)
            df['weight_diff'] = (df['batai_taiju'] - df['prev_weight']).fillna(0).astype('float32')
            
            df['is_weight_changed_huge'] = (df['weight_diff'].abs() > 10).astype('int8')
            
            # Clean up temp
            del df['prev_weight']

        # Clean up temp
        del df['prev_date']

        # ----------------------------------------------------------------
        # 4. 騎手の近走勢い (Jockey Recent Momentum)
        # ----------------------------------------------------------------
        # 時系列ソート (全体)
        df = df.sort_values(['date', 'race_id'])
        
        if 'jockey_id' in df.columns:
            df['is_win'] = (df['rank'] == 1).astype('int8')
            
            # Rolling mean optimization
            df['jockey_recent_win_rate'] = df.groupby('jockey_id')['is_win'].transform(
                lambda x: x.shift(1).rolling(100, min_periods=10).mean()
            ).fillna(0).astype('float32')
            
            del df['is_win']

        # ----------------------------------------------------------------
        # 4.5 調教師の近走勢い (Trainer Recent Momentum) - v7新規
        # ----------------------------------------------------------------
        if 'trainer_id' in df.columns:
            df['is_win_temp'] = (df['rank'] == 1).astype('int8')
            
            df['trainer_recent_win_rate'] = df.groupby('trainer_id')['is_win_temp'].transform(
                lambda x: x.shift(1).rolling(100, min_periods=10).mean()
            ).fillna(0).astype('float32')
            
            del df['is_win_temp']

        # ----------------------------------------------------------------
        # 4.6 モメンタム: 直近5走の着順トレンド (Momentum Slope) - v7新規
        # ----------------------------------------------------------------
        logger.info("v7: モメンタム(momentum_slope)を計算中...")
        lag_rank_cols = ['lag1_rank', 'lag2_rank', 'lag3_rank', 'lag4_rank', 'lag5_rank']
        existing_lag_cols = [c for c in lag_rank_cols if c in df.columns]
        
        if len(existing_lag_cols) >= 3:
            # ベクトル化した線形回帰の傾き計算
            # y = [lag5, lag4, lag3, lag2, lag1] (古い順)
            # x = [0, 1, 2, 3, 4]
            # 傾きが負 = 着順が良くなっている（上昇トレンド）
            
            lag_df = df[existing_lag_cols].astype('float32')
            n = len(existing_lag_cols)
            
            # Simple linear regression slope: Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
            x = pd.Series(range(n))
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()
            
            # 加重平均（古い方から新しい方への変化）
            # lag5が最も古い(x=0)、lag1が最新(x=n-1)
            y_mean = lag_df.mean(axis=1)
            
            # 各列に対してx-x̄を乗算して合計
            numerator = 0
            for i, col in enumerate(existing_lag_cols):  # lag1 is index 0 but newest
                # lag1_rank is most recent, so assign highest x value
                x_val = n - 1 - i  # lag1 -> 4, lag5 -> 0
                numerator += (x_val - x_mean) * (lag_df[col] - y_mean)
            
            df['momentum_slope'] = (numerator / x_var).fillna(0).astype('float32')
            
            # 符号反転: 傾きが正 = 最近になるほど着順が悪化 → 負の傾きが良いトレンド
            # そのままでOK (正の値 = 悪化トレンド、負の値 = 上昇トレンド)
            del lag_df
        else:
            df['momentum_slope'] = 0.0

        # ----------------------------------------------------------------
        # 4.7 休養スコア (Rest Score) - v7新規
        # ----------------------------------------------------------------
        logger.info("v7: 休養スコア(rest_score)を計算中...")
        if 'interval' in df.columns:
            def calc_rest_score(interval):
                if pd.isna(interval) or interval <= 0:
                    return 0.8  # 初出走
                elif 14 <= interval <= 28:
                    return 1.0  # 最適休養
                elif interval < 14:
                    return max(0.5, 0.8 - (14 - interval) * 0.02)  # 詰まりすぎ
                elif interval <= 56:
                    return 0.9  # 中程度の休養
                else:
                    return max(0.3, 0.7 - (interval - 56) / 200)  # 長期休養
            
            df['rest_score'] = df['interval'].apply(calc_rest_score).astype('float32')
        else:
            df['rest_score'] = 0.8

        # ----------------------------------------------------------------
        # 5. レース展開・レベル予測 (Race Context)
        # ----------------------------------------------------------------
        # groupby オブジェクトを再利用
        grouped_race = df.groupby('race_id')

        # メンバーの平均逃げ率
        df['race_avg_nige_rate'] = grouped_race['nige_rate'].transform('mean').astype('float32')

        # 逃げ馬候補の数
        # transform で行うとメモリ食う可能性があるので、mapでやる手もあるが、一旦そのまま型指定
        df['race_nige_horse_count'] = grouped_race['nige_rate'].transform(lambda x: (x >= 0.5).sum()).astype('int16')
        
        # 逃げ馬割合
        race_counts = grouped_race['race_id'].transform('count')
        df['race_nige_bias'] = (df['race_nige_horse_count'] / race_counts).astype('float32')

        # ペース予測
        def categorize_pace(x):
            if x < 0.2: return 0
            elif x < 0.4: return 1
            else: return 2
        
        df['race_pace_cat'] = df['race_avg_nige_rate'].apply(categorize_pace).astype('int8')

        # メンバーの平均獲得賞金
        if 'total_prize' in df.columns:
             df['race_avg_prize'] = grouped_race['total_prize'].transform('mean').astype('float32')
        
        # メンバーの平均年齢
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['race_avg_age'] = grouped_race['age'].transform('mean').astype('float32')

        # ================================================================
        # 6. 新規特徴量 (v6 Feature Engineering)
        # ================================================================
        logger.info("v6 新規特徴量を生成中...")
        
        # 6.1 出走頭数
        logger.info("v6.1: N Horses")
        df['n_horses'] = race_counts.astype('int16') # reuse race_counts
        
        # 6.2 枠番ゾーン
        logger.info("v6.2: Frame Zone")
        if 'frame_number' in df.columns:
            df['frame_number'] = pd.to_numeric(df['frame_number'], errors='coerce')
            def frame_to_zone(f):
                if pd.isna(f): return 1
                if f <= 2: return 0  # 内枠
                elif f <= 6: return 1  # 中枠
                else: return 2  # 外枠
            df['frame_zone'] = df['frame_number'].apply(frame_to_zone).astype('int8')
        
        # 6.3 直近3走の平均着順 & 勝率
        logger.info("v6.3: Recent 3 Runs")
        lag_cols = ['lag1_rank', 'lag2_rank', 'lag3_rank']
        existing_lag_cols = [c for c in lag_cols if c in df.columns]
        
        if len(existing_lag_cols) >= 2:
            # Vectorized operations are faster and lighter than apply
            lag_df = df[existing_lag_cols].astype('float32')
            
            # Mean rank
            df['recent_3_avg_rank'] = lag_df.mean(axis=1, skipna=True).fillna(9.0).astype('float32')
            
            # Win rate logic vectorized
            wins = (lag_df == 1.0).sum(axis=1)
            counts = lag_df.notna().sum(axis=1)
            
            df['recent_3_win_rate'] = (wins / counts).fillna(0).astype('float32')
            
            del lag_df

        # メモリ最適化ヘルパー: Expanding Mean を CumSum/CumCount で高速計算
        def vectorized_expanding_mean(d, group_cols, target_col, output_col):
            # Sort is required for correct historical accumulation
            d = d.sort_values(group_cols + ['date'])
            
            # Groupby CumSum/CumCount is vectorized and much faster/lighter than transform(expanding)
            g = d.groupby(group_cols)[target_col]
            
            cumsum = g.cumsum()
            cumcount = g.cumcount() # 0-indexed count of previous items
            
            # Shift Logic: We want history EXCLUDING current.
            # current cumulative sum includes current value.
            # prev_cumsum = cumsum - current_value
            # prev_count = cumcount (because currently row N is the (N+1)th item, so there are N items before it)
            
            prev_cumsum = cumsum - d[target_col]
            prev_count = cumcount
            
            # Avoid division by zero
            # 0 count -> 0 rate
            result = prev_cumsum / prev_count.replace(0, 1) # replace 0 with 1 to avoid Inf, then fillna(0) handles the real 0s
            
            # Where prev_count was 0, result is 0/1 = 0. Correct.
            d[output_col] = result.fillna(0).astype('float32')
            return d

        # 6.5 騎手×距離カテゴリ別勝率
        logger.info("v6.5: Jockey x Distance Winrate")
        if 'distance' in df.columns and 'jockey_id' in df.columns:
            df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
            def distance_category(d):
                if pd.isna(d): return 1
                if d < 1400: return 0
                elif d < 1800: return 1
                elif d < 2200: return 2
                else: return 3
            df['distance_category'] = df['distance'].apply(distance_category).astype('int8')
            
            df['is_win_temp'] = (df['rank'] == 1).astype('float32') # Use float for calculations
            
            # Use optimized vectorization
            df = vectorized_expanding_mean(
                df, 
                ['jockey_id', 'distance_category'], 
                'is_win_temp', 
                'jockey_distance_winrate'
            )
            
            del df['is_win_temp']
        
        # 6.6 枠番×芝/ダート別勝率
        logger.info("v6.6: Frame x Surface Winrate")
        if 'frame_number' in df.columns:
             if 'surface_num' not in df.columns and 'surface' in df.columns:
                 surface_map = {'芝': 1, 'ダート': 2, '障害': 3, 'Unknown': 0}
                 df['surface_num'] = df['surface'].map(surface_map).fillna(0).astype('int8')

             if 'surface_num' in df.columns:
                df['is_win_temp'] = (df['rank'] == 1).astype('float32')
                
                # Use optimized vectorization
                # Note: Default fillna was 0.05 in previous version, here 0. 
                # If we want 0.05 for unknowns, we can fillna(0.05)
                # But strict calculation gives 0 for first race.
                df = vectorized_expanding_mean(
                    df, 
                    ['frame_number', 'surface_num'], 
                    'is_win_temp', 
                    'frame_surface_winrate'
                )
                
                # Apply default logic for 0 counts if needed? 
                # Original logic: .fillna(0.05). 
                # Vectorized logic returns 0 for first run.
                # Let's align with original intent -> replace 0.0 with 0.05 ONLY if count was 0?
                # A bit complex to do vectorized. Keep 0 for now or global replace?
                # Actually, 0 win rate is valid if they lost. Unknown (first run) is the issue.
                # Let's keep 0 for simplicity and memory.
                
                del df['is_win_temp']
        
        # 6.7 相対的人気順位
        logger.info("v6.7: Relative Popularity")
        if 'lag1_popularity' in df.columns:
            grouped_race_pop = df.groupby('race_id')['lag1_popularity']
            df['relative_popularity_rank'] = grouped_race_pop.rank(method='min', ascending=True).fillna(df['n_horses'] / 2).astype('float32')
        
        # 6.8 連対率推定
        logger.info("v6.8: Estimated Place Rate")
        if 'mean_rank_all' in df.columns:
            df['estimated_place_rate'] = (1 / (df['mean_rank_all'] + 1)).fillna(0.1).astype('float32')

        # ================================================================
        # 7. v7 新規特徴量
        # ================================================================
        
        # 7.1 前走上がり3F順位 (Lag1 Last 3F Rank)
        logger.info("v7.1: 前走上がり3F順位")
        if 'last_3f' in df.columns:
            # まず現在のレースでの上がり3F順位を計算
            df['last_3f_numeric'] = pd.to_numeric(df['last_3f'], errors='coerce')
            df['last_3f_race_rank'] = df.groupby('race_id')['last_3f_numeric'].rank(
                ascending=True,  # タイムが短いほど良い
                method='min',
                na_option='keep'
            )
            
            # 前走の上がり3F順位を取得
            df = df.sort_values(['horse_id', 'date'])
            df['lag1_last_3f_rank'] = df.groupby('horse_id')['last_3f_race_rank'].shift(1)
            df['lag1_last_3f_rank'] = df['lag1_last_3f_rank'].fillna(8.0).astype('float32')
            
            # 一時カラム削除
            if 'last_3f_numeric' in df.columns:
                del df['last_3f_numeric']
            if 'last_3f_race_rank' in df.columns:
                del df['last_3f_race_rank']
        
        # 7.2 クラスギャップ (Class Gap) - 昇級/降級
        logger.info("v7.2: クラスギャップ")
        if 'class_level' in df.columns:
            
            # 前走のクラスレベルを取得
            df = df.sort_values(['horse_id', 'date'])
            df['lag1_class_level'] = df.groupby('horse_id')['class_level'].shift(1)
            
            # クラスギャップ = 今回 - 前回 (正=昇級, 負=降級)
            df['class_gap'] = (df['class_level'] - df['lag1_class_level']).fillna(0).astype('int8')
            
            # 昇級初戦フラグ
            df['is_class_up'] = (df['class_gap'] > 0).astype('int8')
            
            # 一時カラム削除
            if 'lag1_class_level' in df.columns:
                del df['lag1_class_level']
        if 'is_nige_temp' in df.columns: del df['is_nige_temp']
        
        # Final Downcast check
        df = downcast_floats(df)

        logger.info("高度特徴量の生成完了")
        return df

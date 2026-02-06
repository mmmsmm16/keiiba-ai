import pandas as pd
import numpy as np
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
        
        # ソート（時系列順）- 多くの特徴量がこれを前提とする
        df = df.sort_values(['horse_id', 'date'])
        
        # 新しい特徴量を格納する辞書
        new_cols = {}
        
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

        # passing_rank is needed for calculation
        is_nige_temp = None
        if 'passing_rank' in df.columns and df['passing_rank'].notna().any():
            is_nige_temp = df['passing_rank'].apply(is_nige).astype('int8')
        else:
            logger.info("passing_rankが利用できないため、is_nige_tempを0で初期化します")
            is_nige_temp = pd.Series(0, index=df.index, dtype='int8')

        # 2. 馬ごとの過去の逃げ率 (Nige Rate)
        # memory opt: use transform directly without intermediary if possible
        # Need to use is_nige_temp Series for groupby.
        # df is sorted by horse_id, date.
        new_cols['nige_rate'] = is_nige_temp.groupby(df['horse_id']).transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0).astype('float32')

        # [v18] ハナ争い激化予測 (Nige Intensity)
        # 前走1角で1-2番手だった馬の今走頭数
        def get_top2_pos(s):
            if not isinstance(s, str) or '-' not in s: return 0
            try:
                p1 = int(s.split('-')[0])
                return 1 if p1 <= 2 else 0
            except: return 0

        is_front_runner = df['passing_rank'].apply(get_top2_pos).astype('int8')
        lag1_is_front = is_front_runner.groupby(df['horse_id']).shift(1).fillna(0)
        new_cols['nige_intensity'] = lag1_is_front.groupby(df['race_id']).transform('sum').astype('int16')

        # ----------------------------------------------------------------
        # 3. 間隔 (Interval) と体重変化 (Weight Change)
        # ----------------------------------------------------------------
        prev_date = df.groupby('horse_id')['date'].shift(1)
        interval = (df['date'] - prev_date).dt.days.fillna(0).astype('float32')
        new_cols['interval'] = interval
        new_cols['is_long_break'] = (interval > 180).astype('int8')

        if 'batai_taiju' in df.columns:
            # batai_taiju should be numeric
            weight_numeric = pd.to_numeric(df['batai_taiju'], errors='coerce')
            # Assuming we can modify df for temp cleaning or use series
            # Let's use clean series
            prev_weight = weight_numeric.groupby(df['horse_id']).shift(1)
            weight_diff = (weight_numeric - prev_weight).fillna(0).astype('float32')
            new_cols['weight_diff'] = weight_diff
            new_cols['is_weight_changed_huge'] = (weight_diff.abs() > 10).astype('int8')

        # ----------------------------------------------------------------
        # 4. 騎手の近走勢い (Jockey Recent Momentum)
        # ----------------------------------------------------------------
        # 時系列ソート (全体) - GroupBy transform doesn't strictly need global sort if using rolling on time?
        # Rolling(min_periods) on grouped object works on the order of group.
        # If we want chronological rolling, we need to sort by date.
        # df is sorted by horse_id, date.
        # For Jockey, we need separation by Jockey, then sorted by Date.
        # Re-sorting df is expensive.
        # Can we sort just the indices?
        # Or just accept that we need to sort for these features.
        
        # To avoid re-sorting large DF multiple times:
        # We can perform all horse-based ops first (already sorted).
        # Then sort by date for Jockey/Trainer.
        
        # Horse-based features are 3, 4.6(Momentum), 4.7(Rest), 7.1, 7.2, 5(Shift part)
        # Jockey/Trainer are 4, 4.5
        # Race-based are 5, 8, 
        
        # Let's do Horse-based first. (1, 2, 3, 4.6, 4.7, 7.1, 7.2)
        
        # 4.6 モメンタム: 直近5走の着順トレンド (Momentum Slope)
        logger.info("v7: モメンタム(momentum_slope)を計算中...")
        lag_rank_cols = ['lag1_rank', 'lag2_rank', 'lag3_rank', 'lag4_rank', 'lag5_rank']
        existing_lag_cols = [c for c in lag_rank_cols if c in df.columns]
        
        if len(existing_lag_cols) >= 3:
            lag_df = df[existing_lag_cols].astype('float32')
            n = len(existing_lag_cols)
            x = pd.Series(range(n))
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()
            y_mean = lag_df.mean(axis=1)
            numerator = 0
            for i, col in enumerate(existing_lag_cols):
                x_val = n - 1 - i
                numerator += (x_val - x_mean) * (lag_df[col] - y_mean)
            new_cols['momentum_slope'] = (numerator / x_var).fillna(0).astype('float32')
        else:
            new_cols['momentum_slope'] = 0.0

        # 4.7 休養スコア
        logger.info("v7: 休養スコア(rest_score)を計算中...")
        def calc_rest_score(iv):
            if pd.isna(iv) or iv <= 0: return 0.8
            elif 14 <= iv <= 28: return 1.0
            elif iv < 14: return max(0.5, 0.8 - (14 - iv) * 0.02)
            elif iv <= 56: return 0.9
            else: return max(0.3, 0.7 - (iv - 56) / 200)
            
        new_cols['rest_score'] = interval.apply(calc_rest_score).astype('float32')

        # 7.1 前走上がり3F順位
        logger.info("v7.1: 前走上がり3F順位")
        if 'last_3f' in df.columns:
            l3f_num = pd.to_numeric(df['last_3f'], errors='coerce')
            # Rank within race
            l3f_rank = l3f_num.groupby(df['race_id']).rank(ascending=True, method='min', na_option='keep')
            # Lag 1 by horse (df sorted by horse)
            lag1_l3f = l3f_rank.groupby(df['horse_id']).shift(1)
            new_cols['lag1_last_3f_rank'] = lag1_l3f.fillna(8.0).astype('float32')

        # 7.2 クラスギャップ
        logger.info("v7.2: クラスギャップ")
        if 'class_level' in df.columns:
            lag1_class = df.groupby('horse_id')['class_level'].shift(1)
            class_gap = (df['class_level'] - lag1_class).fillna(0).astype('int8')
            new_cols['class_gap'] = class_gap
            new_cols['is_class_up'] = (class_gap > 0).astype('int8')
            
        # --- Now switch sort to Date for Jockey/Trainer Momentum ---
        # But wait, df index must match new_cols index.
        # If we sort df, we lose alignment with new_cols if new_cols is just a dict of Series (which align by Index).
        # We must NOT change df index without reindexing new_cols or keeping alignment.
        # Best way: Use temporary dataframe for sorting, calculate, then reindex back to df.index.
        
        # Helper for J/T momentum
        def calc_momentum(target_id_col, is_win_series):
            # Create a localized DF to sort
            tmp = pd.DataFrame({'id': df[target_id_col], 'is_win': is_win_series, 'date': df['date']})
            tmp = tmp.sort_values(['date', 'id']) # Sort by date
            
            # Groupby ID and roll
            return tmp.groupby('id')['is_win'].transform(
                lambda x: x.shift(1).rolling(100, min_periods=10).mean()
            ).sort_index() # Restore original index order
            
        if 'jockey_id' in df.columns:
            is_win = (df['rank'] == 1).astype('int8')
            new_cols['jockey_recent_win_rate'] = calc_momentum('jockey_id', is_win).fillna(0).astype('float32')
            
        if 'trainer_id' in df.columns:
            is_win = (df['rank'] == 1).astype('int8')
            new_cols['trainer_recent_win_rate'] = calc_momentum('trainer_id', is_win).fillna(0).astype('float32')

        # ----------------------------------------------------------------
        # 5. レース展開・レベル予測 (Race Context)
        # ----------------------------------------------------------------
        # Need 'nige_rate' which is in new_cols, not df yet!
        # Cannot use df.groupby('race_id')['nige_rate'].
        # Must construct temporary series or dataframe.
        
        # Construct temp DF with necessary columns for Race aggregation
        # We need race_id (from df) and nige_rate (from new_cols)
        # And total_prize, age (from df)
        
        # To avoid concatenating everything now, just use the series aligned with df
        # nige_rate_series = new_cols['nige_rate'] (Series with same index)
        
        grouped_race = df['race_id']
        nige_rate_series = new_cols['nige_rate']
        
        # race_avg_nige_rate
        # We need to group `nige_rate_series` by `df['race_id']`
        # pd.Series.groupby(other_series) works
        new_cols['race_avg_nige_rate'] = nige_rate_series.groupby(grouped_race).transform('mean').astype('float32')
        
        # race_nige_horse_count
        new_cols['race_nige_horse_count'] = nige_rate_series.groupby(grouped_race).transform(lambda x: (x >= 0.5).sum()).astype('int16')
        
        # race_nige_bias
        race_counts = grouped_race.groupby(grouped_race).transform('count')
        new_cols['race_nige_bias'] = (new_cols['race_nige_horse_count'] / race_counts).astype('float32')
        
        # race_pace_cat
        def categorize_pace(x):
            if x < 0.2: return 0
            elif x < 0.4: return 1
            else: return 2
        new_cols['race_pace_cat'] = new_cols['race_avg_nige_rate'].apply(categorize_pace).astype('int8')

        # v11 N4
        logger.info("v11 N4: 展開予測特徴量を生成中...")
        new_cols['nige_candidate_count'] = nige_rate_series.groupby(grouped_race).transform(lambda x: (x > 0.3).sum()).astype('int8')
        new_cols['senkou_ratio'] = nige_rate_series.groupby(grouped_race).transform(lambda x: (x > 0.1).mean()).astype('float32')

        if 'total_prize' in df.columns:
             new_cols['race_avg_prize'] = df['total_prize'].groupby(grouped_race).transform('mean').astype('float32')
        
        if 'age' in df.columns:
            age_num = pd.to_numeric(df['age'], errors='coerce')
            new_cols['race_avg_age'] = age_num.groupby(grouped_race).transform('mean').astype('float32')

        # ----------------------------------------------------------------
        # 6. 新規特徴量 (v6 Feature Engineering)
        # ----------------------------------------------------------------
        logger.info("v6 新規特徴量を生成中...")
        new_cols['n_horses'] = race_counts.astype('int16')
        
        # 6.2 Frame Zone
        logger.info("v6.2: Frame Zone")
        if 'frame_number' in df.columns:
            f_num = pd.to_numeric(df['frame_number'], errors='coerce')
            def frame_to_zone(f):
                if pd.isna(f): return 1
                if f <= 2: return 0
                elif f <= 6: return 1
                else: return 2
            new_cols['frame_zone'] = f_num.apply(frame_to_zone).astype('int8')
        
        # 6.3 Recent 3 avg rank
        logger.info("v6.3: Recent 3 Runs")
        if len(existing_lag_cols) >= 3:
            lag3_cols = ['lag1_rank', 'lag2_rank', 'lag3_rank']
            existing_l3 = [c for c in lag3_cols if c in df.columns]
            if len(existing_l3) >= 2:
                lag_df = df[existing_l3].astype('float32')
                new_cols['recent_3_avg_rank'] = lag_df.mean(axis=1, skipna=True).fillna(9.0).astype('float32')
                wins = (lag_df == 1.0).sum(axis=1)
                counts = lag_df.notna().sum(axis=1)
                new_cols['recent_3_win_rate'] = (wins / counts).fillna(0).astype('float32')

        # Vectorized Expanding Mean Helper
        def vectorized_expanding_mean(d, group_cols, target_col):
            # Sort is required
            sort_cols = group_cols + ['date']
            d = d.sort_values(sort_cols)
            g = d.groupby(group_cols)[target_col]
            cumsum = g.cumsum()
            cumcount = g.cumcount()
            prev_cumsum = cumsum - d[target_col]
            result = prev_cumsum / cumcount.replace(0, 1)
            # Restore index
            return result.fillna(0).astype('float32').sort_index()

        # 6.5 Jockey x Distance Winrate
        logger.info("v6.5: Jockey x Distance Winrate")
        if 'distance' in df.columns and 'jockey_id' in df.columns:
            dist = pd.to_numeric(df['distance'], errors='coerce')
            def distance_category(d):
                if pd.isna(d): return 1
                if d < 1400: return 0
                elif d < 1800: return 1
                elif d < 2200: return 2
                else: return 3
            dist_cat = dist.apply(distance_category).astype('int8')
            # Temporarily add to df for helper? Or modify helper to accept series.
            # Helper sorts df, so it needs df.
            # Let's create temp df for helper.
            is_win = (df['rank'] == 1).astype('float32')
            tmp = pd.DataFrame({
                'jockey_id': df['jockey_id'],
                'distance_category': dist_cat,
                'is_win': is_win,
                'date': df['date']
            })
            new_cols['jockey_distance_winrate'] = vectorized_expanding_mean(
                tmp, ['jockey_id', 'distance_category'], 'is_win'
            )
        
        # 6.6 Frame x Surface Winrate
        logger.info("v6.6: Frame x Surface Winrate")
        if 'frame_number' in df.columns and 'surface' in df.columns:
             # Logic simplified
             surface_map = {'芝': 1, 'ダート': 2, '障害': 3, 'Unknown': 0}
             surface_num = df['surface'].map(surface_map).fillna(0).astype('int8')
             is_win = (df['rank'] == 1).astype('float32')
             tmp = pd.DataFrame({
                'frame_number': df['frame_number'],
                'surface_num': surface_num,
                'is_win': is_win,
                'date': df['date']
             })
             new_cols['frame_surface_winrate'] = vectorized_expanding_mean(
                 tmp, ['frame_number', 'surface_num'], 'is_win'
             )

        # 6.7 Relative Popularity
        logger.info("v6.7: Relative Popularity")
        if 'lag1_popularity' in df.columns:
            new_cols['relative_popularity_rank'] = df.groupby('race_id')['lag1_popularity'].rank(method='min', ascending=True).fillna(new_cols['n_horses'] / 2).astype('float32')
        
        # 6.8 Estimated Place Rate
        logger.info("v6.8: Estimated Place Rate")
        if 'mean_rank_all' in df.columns:
            new_cols['estimated_place_rate'] = (1 / (df['mean_rank_all'] + 1)).fillna(0.1).astype('float32')

        # 6.9 Jockey x Course Winrate (Pure Performance)
        logger.info("v12: Jockey x Course Winrate")
        if 'jockey_id' in df.columns and 'course_id' in df.columns:
            is_win = (df['rank'] == 1).astype('float32')
            tmp = pd.DataFrame({
                'jockey_id': df['jockey_id'],
                'course_id': df['course_id'],
                'is_win': is_win,
                'date': df['date']
            })
            new_cols['jockey_course_winrate'] = vectorized_expanding_mean(
                tmp, ['jockey_id', 'course_id'], 'is_win'
            )

        # 6.10 Trainer x Jockey Winrate (Pure Performance)
        logger.info("v12: Trainer x Jockey Winrate")
        if 'trainer_id' in df.columns and 'jockey_id' in df.columns:
            is_win = (df['rank'] == 1).astype('float32')
            tmp = pd.DataFrame({
                'trainer_id': df['trainer_id'],
                'jockey_id': df['jockey_id'],
                'is_win': is_win,
                'date': df['date']
            })
            new_cols['trainer_jockey_winrate'] = vectorized_expanding_mean(
                tmp, ['trainer_id', 'jockey_id'], 'is_win'
            )

        # 6.11 Sire x Course Winrate (Pure Performance)
        logger.info("v12: Sire x Course Winrate")
        if 'sire_id' in df.columns and 'course_id' in df.columns:
            is_win = (df['rank'] == 1).astype('float32')
            tmp = pd.DataFrame({
                'sire_id': df['sire_id'],
                'course_id': df['course_id'],
                'is_win': is_win,
                'date': df['date']
            })
            new_cols['sire_course_winrate'] = vectorized_expanding_mean(
                tmp, ['sire_id', 'course_id'], 'is_win'
            )

        # ----------------------------------------------------------------
        # P4: Running Style Continuous Features (脚質連続値)
        # ----------------------------------------------------------------
        logger.info("P4: Running Style Continuous Features を生成中...")
        
        # passing_rank（通過順位）から脚質を連続値化
        # データにある場合のみ生成
        if 'n_horses' in new_cols:
            n_horses_s = new_cols['n_horses']
        elif 'n_horses' in df.columns:
            n_horses_s = df['n_horses']
        else:
            n_horses_s = df.groupby('race_id')['rank'].transform('count')
        
        # passing_1（1角通過順位）と passing_4（4角通過順位）をチェック
        has_passing = 'passing_1' in df.columns and 'passing_4' in df.columns
        
        if has_passing:
            # 通過順位を正規化（0=先頭, 1=最後尾）
            passing_1 = pd.to_numeric(df['passing_1'], errors='coerce')
            passing_4 = pd.to_numeric(df['passing_4'], errors='coerce')
            
            early_pos_pct = (passing_1 - 1) / (n_horses_s - 1).clip(lower=1)
            late_pos_pct = (passing_4 - 1) / (n_horses_s - 1).clip(lower=1)
            
            # 位置変化（正=追い込み型、負=逃げ残り型）
            pos_gain = early_pos_pct - late_pos_pct  # 前から下がった=正
            
            # Lag1 (前走の値)
            new_cols['lag1_early_pos_pct'] = df.groupby('horse_id')[early_pos_pct].shift(1).fillna(0.5).astype('float32')
            new_cols['lag1_late_pos_pct'] = df.groupby('horse_id')[late_pos_pct].shift(1).fillna(0.5).astype('float32')
            new_cols['lag1_pos_gain'] = df.groupby('horse_id')[pos_gain].shift(1).fillna(0.0).astype('float32')
            
            # 過去3走平均のpos_gain
            new_cols['avg_pos_gain_3'] = df.groupby('horse_id')[pos_gain].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            ).fillna(0.0).astype('float32')
            
            logger.info(f"  lag1_early_pos_pct 平均: {new_cols['lag1_early_pos_pct'].mean():.2f}")
            logger.info(f"  lag1_pos_gain 平均: {new_cols['lag1_pos_gain'].mean():.3f}")
        else:
            logger.warning("P4: passing_1/passing_4 がないため脚質連続値をスキップ")
        
        # Closing Index (末脚型指標) = last_3f_index - time_index
        # 既に Speed Index が計算されている場合のみ
        if 'lag1_last_3f_index' in df.columns and 'lag1_time_index' in df.columns:
            new_cols['closing_index'] = (df['lag1_last_3f_index'] - df['lag1_time_index']).fillna(0.0).astype('float32')
            logger.info(f"P4: closing_index 生成完了 (平均: {new_cols['closing_index'].mean():.2f})")

        # 8. [v11 Phase B] 追加特徴量
        logger.info("v11 Phase B: 追加特徴量を生成中...")
        if 'weight' in df.columns:
            w_num = pd.to_numeric(df['weight'], errors='coerce')
            new_cols['weight_is_missing'] = w_num.isna().astype('int8')
            new_cols['weight'] = w_num.fillna(w_num.median() if not w_num.isna().all() else 470.0).astype('float32')

        # B3 total_prize check (logging only)
        if 'total_prize' in df.columns:
             non_zero_rate = (df['total_prize'] != 0).mean()
             if non_zero_rate < 0.01:
                 logger.warning(f"B3: total_prize low non-zero rate: {non_zero_rate:.2%}")
        
        # ----------------------------------------------------------------
        # 9. [v11 N2] Speed Index (Time & Last 3F) with Hierarchical Fallback
        # ----------------------------------------------------------------
        logger.info("v11 N2: Speed Index calculating (Robust)...")
        
        speed_index_cols_required = ['time', 'course_id', 'distance', 'track_condition_code']
        if all(c in df.columns for c in speed_index_cols_required):
            # 1. 前処理 & カテゴリ生成
            time_num = pd.to_numeric(df['time'], errors='coerce')
            dist_num = pd.to_numeric(df['distance'], errors='coerce')
            track_cond_num = pd.to_numeric(df['track_condition_code'], errors='coerce').fillna(0).astype('int16')
            
            # Distance Category
            def get_dist_cat(d):
                if pd.isna(d): return 1
                if d < 1400: return 0
                elif d < 1800: return 1
                elif d < 2200: return 2
                else: return 3
            
            # Use distance_category from df if exists (e.g. from exp features?), but might not be there.
            # Safe to recompute.
            dist_cat = dist_num.apply(get_dist_cat).astype('int8')

            # Helper
            def get_expanding_stats(d_idx, group_cols, target_s, prefix):
                # Need to construct temp df with target and group cols
                # d_idx is the index of original df
                # group_cols are list of column names, need to fetch from df or passed data
                
                # Construct clean temp DF
                data = {c: df[c] if c in df.columns else None for c in group_cols if isinstance(c, str)} 
                # Note: some group_cols might be series passed in? No, always existing cols or we pass series.
                # Here group_cols are from 'df' except dist_cat which is local.
                
                # Better approach: Pass Series list for grouping
                # group_keys: list of Series
                
                # Re-implementation for isolated Usage
                # Sort criteria: group_cols + [date, race_id]
                # We need date/race_id from df
                
                tmp = pd.DataFrame({'date': df['date'], 'race_id': df['race_id'], 'target': target_s})
                
                # Add group keys to tmp
                keys = []
                for val in group_cols:
                    if isinstance(val, str):
                        tmp[val] = df[val]
                        keys.append(val)
                    else:
                        # Assumed Series
                        col_name = f'key_{len(keys)}'
                        tmp[col_name] = val
                        keys.append(col_name)
                
                sort_cols = keys + ['date', 'race_id']
                tmp = tmp.sort_values(sort_cols)
                
                g = tmp.groupby(keys)['target']
                
                prev_count = g.cumcount()
                cumsum = g.cumsum()
                prev_sum = cumsum - tmp['target'].fillna(0)
                
                tmp['target_sq'] = tmp['target'] ** 2
                cumsum_sq = tmp.groupby(keys)['target_sq'].cumsum()
                prev_sum_sq = cumsum_sq - tmp['target_sq'].fillna(0)
                del tmp['target_sq']
                
                denom = prev_count.replace(0, 1)
                stats_mean = prev_sum / denom
                stats_mean_sq = prev_sum_sq / denom
                stats_var = (stats_mean_sq - stats_mean**2).clip(lower=0)
                stats_std = np.sqrt(stats_var)
                
                # Return aligned to original index (fill missing with 0 for robustness)
                return (
                    stats_mean.reindex(d_idx, fill_value=0).astype('float32'),
                    stats_std.reindex(d_idx, fill_value=0).astype('float32'),
                    prev_count.reindex(d_idx, fill_value=0).astype('int32')
                )

            # Define Keys
            # L1: Course x Dist x Cond
            l1_keys = ['course_id', dist_num, track_cond_num]
            # L2: Course x DistCat x Cond
            l2_keys = ['course_id', dist_cat, track_cond_num]
            # L3: Course x Cond
            l3_keys = ['course_id', track_cond_num]

            # ---------------------------
            # Time Index Calc
            # ---------------------------
            # Get Stats
            l1_mean, l1_std, l1_count = get_expanding_stats(df.index, l1_keys, time_num, 'L1')
            l2_mean, l2_std, l2_count = get_expanding_stats(df.index, l2_keys, time_num, 'L2')
            l3_mean, l3_std, l3_count = get_expanding_stats(df.index, l3_keys, time_num, 'L3')
            
            min_samples = 30
            # Masks
            mask_l1 = (l1_count >= min_samples) & (l1_std > 0.001)
            mask_l2 = (~mask_l1) & (l2_count >= min_samples) & (l2_std > 0.001)
            mask_l3 = (~mask_l1) & (~mask_l2) & (l3_count >= min_samples) & (l3_std > 0.001)
            
            # Vectorized Choice
            # Default 50.0
            t_idx = pd.Series(50.0, index=df.index, dtype='float32')
            
            # Lower time is better -> Mean - Time
            def calc_score(mean_s, std_s, val_s):
                return 50.0 + 10.0 * (mean_s - val_s) / std_s
            
            # Apply
            # Using Series.where is nice but we have 3 layers.
            # Using update/loc
            if mask_l1.any():
                t_idx.loc[mask_l1] = calc_score(l1_mean, l1_std, time_num)[mask_l1]
            if mask_l2.any():
                t_idx.loc[mask_l2] = calc_score(l2_mean, l2_std, time_num)[mask_l2]
            if mask_l3.any():
                t_idx.loc[mask_l3] = calc_score(l3_mean, l3_std, time_num)[mask_l3]
                
            # Shift Logic: Lag1 (this is for prediction, so we use lagged index)
            # Group by Horse ID (df sorted by horse date for this)
            # Re-sort input data? No, df is sorted by horse_id at start of function.
            # But the Series t_idx is Aligned with df.
            # So t_idx follows df's (horse_id) order.
            new_cols['lag1_time_index'] = t_idx.groupby(df['horse_id']).shift(1).fillna(50.0).astype('float32')

            # ---------------------------
            # Last 3F Index Calc
            # ---------------------------
            if 'last_3f' in df.columns:
                l3f_num = pd.to_numeric(df['last_3f'], errors='coerce')
                
                l1_3f_mean, l1_3f_std, l1_3f_count = get_expanding_stats(df.index, l1_keys, l3f_num, 'L1_3f')
                l2_3f_mean, l2_3f_std, l2_3f_count = get_expanding_stats(df.index, l2_keys, l3f_num, 'L2_3f')
                l3_3f_mean, l3_3f_std, l3_3f_count = get_expanding_stats(df.index, l3_keys, l3f_num, 'L3_3f')
                
                l3f_idx = pd.Series(50.0, index=df.index, dtype='float32')
                
                m_l1 = (l1_3f_count >= min_samples) & (l1_3f_std > 0.001)
                m_l2 = (~m_l1) & (l2_3f_count >= min_samples) & (l2_3f_std > 0.001)
                m_l3 = (~m_l1) & (~m_l2) & (l3_3f_count >= min_samples) & (l3_3f_std > 0.001)
                
                def calc_score_3f(mean_s, std_s, val_s):
                    res = 50.0 + 10.0 * (mean_s - val_s) / std_s
                    return res.astype('float32')

                if m_l1.any():
                    l3f_idx.loc[m_l1] = calc_score_3f(l1_3f_mean, l1_3f_std, l3f_num)[m_l1]
                if m_l2.any():
                    l3f_idx.loc[m_l2] = calc_score_3f(l2_3f_mean, l2_3f_std, l3f_num)[m_l2]
                if m_l3.any():
                    l3f_idx.loc[m_l3] = calc_score_3f(l3_3f_mean, l3_3f_std, l3f_num)[m_l3]
                    
            new_cols['lag1_last_3f_index'] = l3f_idx.groupby(df['horse_id']).shift(1).fillna(50.0).astype('float32')

            # ---------------------------
            # [v18] First 3F Index Calc (Early Speed)
            # ---------------------------
            if 'first_3f' in df.columns:
                f3f_num = pd.to_numeric(df['first_3f'], errors='coerce')
                
                l1_f3f_mean, l1_f3f_std, l1_f3f_count = get_expanding_stats(df.index, l1_keys, f3f_num, 'L1_f3f')
                l2_f3f_mean, l2_f3f_std, l2_f3f_count = get_expanding_stats(df.index, l2_keys, f3f_num, 'L2_f3f')
                l3_f3f_mean, l3_f3f_std, l3_f3f_count = get_expanding_stats(df.index, l3_keys, f3f_num, 'L3_f3f')
                
                f3f_idx = pd.Series(50.0, index=df.index, dtype='float32')
                
                m_l1_f = (l1_f3f_count >= min_samples) & (l1_f3f_std > 0.001)
                m_l2_f = (~m_l1_f) & (l2_f3f_count >= min_samples) & (l2_f3f_std > 0.001)
                m_l3_f = (~m_l1_f) & (~m_l2_f) & (l3_f3f_count >= min_samples) & (l3_f3f_std > 0.001)
                
                if m_l1_f.any():
                    f3f_idx.loc[m_l1_f] = calc_score(l1_f3f_mean, l1_f3f_std, f3f_num)[m_l1_f]
                if m_l2_f.any():
                    f3f_idx.loc[m_l2_f] = calc_score(l2_f3f_mean, l2_f3f_std, f3f_num)[m_l2_f]
                if m_l3_f.any():
                    f3f_idx.loc[m_l3_f] = calc_score(l3_f3f_mean, l3_f3f_std, f3f_num)[m_l3_f]
                
                new_cols['lag1_first_3f_index'] = f3f_idx.groupby(df['horse_id']).shift(1).fillna(50.0).astype('float32')
                
                # レースメンバーのテン3F指数分散 (Early Speed Std)
                new_cols['early_speed_std'] = new_cols['lag1_first_3f_index'].groupby(df['race_id']).transform('std').fillna(0).astype('float32')
                new_cols['lag1_first_3f_index'] = 50.0
                new_cols['early_speed_std'] = 0.0

            logger.info("v11 N2: Speed Index generated (lag1_time_index, lag1_last_3f_index, lag1_first_3f_index).")
            
            # ---------------------------
            # [v19] Jockey Pace Interaction
            # ---------------------------
            # 騎手のPCI傾向 (lag1_pci based)
            if 'lag1_pci' in df.columns and 'jockey_id' in df.columns:
                # 騎手ごとの平均PCI (Expanding)
                new_cols['jockey_avg_pci'] = df.groupby('jockey_id')['lag1_pci'].transform(
                     lambda x: x.expanding().mean()
                ).fillna(50.0).astype('float32')
                
                # レースの想定ペース (race_avg_nige_rate) との交互作用
                # nige_rateが高い(逃げ馬多い) -> ペース速い -> PCI低い(消耗戦)ほど有利?
                # または、騎手がハイペース得意なら、ハイペースレースで有利。
                # 単純なInteraction: jockey_avg_pci * race_avg_nige_rate
                if 'race_avg_nige_rate' in new_cols:
                    new_cols['jockey_pci_interaction'] = (new_cols['jockey_avg_pci'] * new_cols['race_avg_nige_rate']).astype('float32')
        else:
            missing = [c for c in speed_index_cols_required if c not in df.columns]
            logger.warning(f"Speed Index calculation skipped. Missing cols: {missing}")
            new_cols['lag1_time_index'] = pd.Series(50.0, index=df.index, dtype='float32')
            new_cols['lag1_last_3f_index'] = pd.Series(50.0, index=df.index, dtype='float32')

        # Final Concat
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            # Prevent duplicate columns by dropping existing ones first
            cols_to_drop = new_df.columns.intersection(df.columns)
            if not cols_to_drop.empty:
                df = df.drop(columns=cols_to_drop)
            
            df = pd.concat([df, new_df], axis=1)

        # Final Downcast check
        df = downcast_floats(df)

        logger.info("高度特徴量の生成完了")
        return df



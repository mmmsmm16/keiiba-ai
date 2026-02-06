import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class JockeyTrainerProfileEngineer:
    """
    騎手・厩舎(調教師)に関するプロファイル特徴量を生成するクラス。
    条件別（コース、距離など）の成績を集計し、相性特徴量も生成します。
    
    Phase 18 Update:
    - 騎手xコース、調教師xコース、騎手x調教師
    - Strict Leak Prevention (Daily Aggregation -> Shift -> Merge)
    - Bayesian Smoothing
    """
    
    SMOOTHING_C = 30.0
    
    def __init__(self):
        # 外部データのロードは不要（df内のIDを使用）
        pass

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("騎手・厩舎プロファイル特徴量(Phase 18)の生成を開始...")
        
        # 必須カラム確認
        # jockey_id, trainer_id, course_id, date, rank
        required_cols = ['date', 'rank', 'jockey_id', 'trainer_id', 'course_id']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"以下の必須カラムが不足しているため、プロファイル集計をスキップします: {missing}")
            return df

        # 型変換
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # 数値化
        df['rank_numeric'] = pd.to_numeric(df['rank'], errors='coerce')
        df['is_win'] = (df['rank_numeric'] == 1).astype(int)
        df['is_top3'] = (df['rank_numeric'] <= 3).astype(int)
        
        # 欠損値埋め
        fill_vals = {'course_id': 'Unknown', 'jockey_id': 'Unknown', 'trainer_id': 'Unknown'}
        for col, val in fill_vals.items():
            if col in df.columns:
                if df[col].dtype.name == 'category':
                    df[col] = df[col].astype(str)
                df[col] = df[col].fillna(val)

        GLOBAL_WIN_RATE = 0.08
        GLOBAL_TOP3_RATE = 0.25

        # 集計ロジック (Strict Leak Prevention: Daily Aggregation)
        def calculate_smoothed_stats(group_cols, prefix):
            nonlocal df
            logger.info(f"Target Encoding: {prefix} (cols={group_cols})")
            
            agg_keys = group_cols + ['date']
            
            # 1. Daily Stats
            daily = df.groupby(agg_keys)[['rank_numeric', 'is_win', 'is_top3']].agg({
                'rank_numeric': 'count',
                'is_win': 'sum',
                'is_top3': 'sum'
            }).reset_index()
            daily.rename(columns={'rank_numeric': 'daily_count', 'is_win': 'daily_win', 'is_top3': 'daily_top3'}, inplace=True)
            
            # 2. Cumulative Stats (Up to Yesterday)
            daily = daily.sort_values(agg_keys)
            grouped = daily.groupby(group_cols)
            
            daily['cum_count'] = grouped['daily_count'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
            daily['cum_win'] = grouped['daily_win'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
            daily['cum_top3'] = grouped['daily_top3'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
            
            # 3. Smoothing
            C = self.SMOOTHING_C
            daily[f'{prefix}_win_rate'] = (daily['cum_win'] + C * GLOBAL_WIN_RATE) / (daily['cum_count'] + C)
            daily[f'{prefix}_top3_rate'] = (daily['cum_top3'] + C * GLOBAL_TOP3_RATE) / (daily['cum_count'] + C)
            daily[f'{prefix}_n_samples'] = daily['cum_count'].astype('int32')
            
            # Raw Counts
            daily[f'{prefix}_wins'] = daily['cum_win'].astype('int32')
            daily[f'{prefix}_top3'] = daily['cum_top3'].astype('int32')

            # Float32 cast
            cols = [f'{prefix}_win_rate', f'{prefix}_top3_rate']
            daily[cols] = daily[cols].astype('float32')
            
            # 4. Merge
            merge_cols = agg_keys + cols + [f'{prefix}_n_samples', f'{prefix}_wins', f'{prefix}_top3']
            df = df.merge(daily[merge_cols], on=agg_keys, how='left')

        # --- Features ---
        
        # 1. Jockey Overall (Backoff)
        calculate_smoothed_stats(['jockey_id'], 'jockey_overall')
        
        # 2. Trainer Overall (Backoff)
        calculate_smoothed_stats(['trainer_id'], 'trainer_overall')
        
        # 3. Jockey x Course
        calculate_smoothed_stats(['jockey_id', 'course_id'], 'jockey_course')
        
        # 4. Trainer x Course
        calculate_smoothed_stats(['trainer_id', 'course_id'], 'trainer_course')
        
        # 5. Jockey x Trainer (Compatibility)
        calculate_smoothed_stats(['jockey_id', 'trainer_id'], 'jockey_trainer')

        # Cleanup
        drop_cols = ['rank_numeric', 'is_win', 'is_top3']
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        logger.info("騎手・厩舎プロファイル特徴量の生成完了")
        return df

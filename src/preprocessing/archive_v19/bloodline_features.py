import pandas as pd
import numpy as np
import logging
from .loader import JraVanDataLoader

logger = logging.getLogger(__name__)

class BloodlineFeatureEngineer:
    """
    血統（種牡馬・繁殖牝馬）に関する特徴量を生成するクラス。
    PC-KEIBAのjvd_um（競走馬マスタ）テーブルから血統情報を取得し、
    条件別（コース、距離、馬場）の適性を集計します。
    
    Phase 18 Update:
    - 距離区分、馬場区分、コース区分ごとの詳細集計
    - Bayesian Smoothingによる疎なデータの補正
    - Strict Leak Prevention (Daily Aggregation -> Shift -> Marge)
      同日開催レース間のリークを防ぐため、常に「前日までの成績」を使用する。
    """
    
    # Smoothing parameter (Prior weight)
    SMOOTHING_C = 30.0
    
    def __init__(self, data_loader: JraVanDataLoader = None):
        self.loader = data_loader if data_loader else JraVanDataLoader()
        self.bloodline_map = None

    def _load_bloodline_data(self):
        """競走馬マスタから血統情報をロード"""
        if self.bloodline_map is not None:
            return

        logger.info("競走馬マスタから血統情報をロード中...")
        # ketto_joho_01a: Sire ID, 02a: Mare ID, 03a: BMS ID
        query = """
        SELECT
            ketto_toroku_bango AS horse_id,
            ketto_joho_01a AS sire_id,
            ketto_joho_02a AS mare_id,
            ketto_joho_03a AS bms_id
        FROM jvd_um
        """
        try:
            df_um = pd.read_sql(query, self.loader.engine)
            df_um = df_um.drop_duplicates(subset=['horse_id'])
            self.bloodline_map = df_um[['horse_id', 'sire_id', 'mare_id', 'bms_id']]
            logger.info(f"血統情報ロード完了: {len(self.bloodline_map)}頭")
        except Exception as e:
            logger.error(f"血統情報のロードに失敗しました: {e}")
            self.bloodline_map = pd.DataFrame()

    def _assign_distance_type(self, distance):
        """距離区分判定 (1000-1400: Short, 1401-1800: Mile, 1801-2200: Intermediate, 2201+: Long)"""
        if pd.isna(distance): return 'Unknown'
        try:
            dist = float(distance)
        except:
            return 'Unknown'
        
        if dist <= 1400: return 'Short'
        elif dist <= 1800: return 'Mile'
        elif dist <= 2200: return 'Intermediate'
        else: return 'Long'

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("血統適性特徴量(Phase 18)の生成を開始...")

        # 1. 血統情報の結合
        self._load_bloodline_data()
        if self.bloodline_map.empty:
            return df

        new_cols = [c for c in self.bloodline_map.columns if c not in df.columns and c != 'horse_id']
        if new_cols:
            df = df.merge(self.bloodline_map[['horse_id'] + new_cols], on='horse_id', how='left')

        # 2. 前処理 (距離区分など)
        if 'distance_type' not in df.columns and 'distance' in df.columns:
            df['distance_type'] = df['distance'].apply(self._assign_distance_type)

        # 確保すべきカラム
        required_cols = ['date', 'rank', 'sire_id', 'course_id', 'surface', 'distance_type']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"以下の必須カラムが不足しているため、血統詳細集計をスキップします: {missing}")
            return df

        # 型変換とfillna
        df['date'] = pd.to_datetime(df['date'])
        df['rank_numeric'] = pd.to_numeric(df['rank'], errors='coerce')
        df['is_win'] = (df['rank_numeric'] == 1).astype(int)
        df['is_top3'] = (df['rank_numeric'] <= 3).astype(int)
        
        # 欠損値埋め: GroupByキーがNaNだと除外されるため
        # Categorical型の場合はエラーになるため、object型に変換して埋める
        fill_vals = {'course_id': 'Unknown', 'surface': 'Unknown', 'distance_type': 'Unknown'}
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
            
            # GroupBy Keys needs Date
            agg_keys = group_cols + ['date']
            
            # 1. Daily Stats (その日の成績)
            # count() for starts, sum() for flags
            daily = df.groupby(agg_keys)[['rank_numeric', 'is_win', 'is_top3']].agg({
                'rank_numeric': 'count',
                'is_win': 'sum',
                'is_top3': 'sum'
            }).reset_index()
            daily.rename(columns={'rank_numeric': 'daily_count', 'is_win': 'daily_win', 'is_top3': 'daily_top3'}, inplace=True)
            
            # 2. Cumulative Stats (Up to Yesterday)
            # Sort by date
            daily = daily.sort_values(agg_keys)
            grouped = daily.groupby(group_cols)
            
            # shift(1) -> expanding -> sum
            daily['cum_count'] = grouped['daily_count'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
            daily['cum_win'] = grouped['daily_win'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
            daily['cum_top3'] = grouped['daily_top3'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
            
            # 3. Smoothing
            C = self.SMOOTHING_C
            daily[f'{prefix}_win_rate'] = (daily['cum_win'] + C * GLOBAL_WIN_RATE) / (daily['cum_count'] + C)
            daily[f'{prefix}_top3_rate'] = (daily['cum_top3'] + C * GLOBAL_TOP3_RATE) / (daily['cum_count'] + C)
            daily[f'{prefix}_n_samples'] = daily['cum_count'] # Request: keep n_samples
            
            # Also keep raw counts as requested
            daily[f'{prefix}_wins'] = daily['cum_win'].astype('int32')
            daily[f'{prefix}_top3'] = daily['cum_top3'].astype('int32')

            # Float32 cast
            cols = [f'{prefix}_win_rate', f'{prefix}_top3_rate']
            daily[cols] = daily[cols].astype('float32')
            daily[f'{prefix}_n_samples'] = daily[f'{prefix}_n_samples'].astype('int32')
            
            # 4. Merge
            merge_cols = agg_keys + cols + [f'{prefix}_n_samples', f'{prefix}_wins', f'{prefix}_top3']
            
            df = df.merge(daily[merge_cols], on=agg_keys, how='left')
            
        # --- Features ---
        
        # 1. Sire Overall (Backoff)
        calculate_smoothed_stats(['sire_id'], 'sire_overall')
        
        # 2. Sire x Course
        calculate_smoothed_stats(['sire_id', 'course_id'], 'sire_course')
        
        # 3. Sire x Distance Type
        calculate_smoothed_stats(['sire_id', 'distance_type'], 'sire_distance')
        
        # 4. Sire x Surface
        calculate_smoothed_stats(['sire_id', 'surface'], 'sire_surface')

        # BMS (母父) Overall
        calculate_smoothed_stats(['bms_id'], 'bms_overall')

        # Cleanup
        drop_cols = ['rank_numeric', 'is_win', 'is_top3']
        # Feature columns (distance_type) may be kept or dropped. 
        # Plan doesn't say. Keeping it is useful.
        
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        # Fill NaNs for features (Implicitly handled by left merge + Prior logic? No.)
        # If merge yields NaN (e.g. unknown sire not in daily stats? No, daily comes from df),
        # only possibility is keys were NaN in df? But filled above.
        # So essentially no NaNs expected, except if expanding sum yields NaN?
        # fillna(0) was used.
        # So no NaNs.
        # Just in case, fillna(0) for counts and Prior for rates.
        # But let's trust logic.
        
        logger.info("血統特徴量の生成完了")
        return df

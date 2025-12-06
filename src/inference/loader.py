import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import ProgrammingError
import logging

logger = logging.getLogger(__name__)

class InferenceDataLoader:
    """
    推論用：開催予定のレースデータ（出馬表）をロードするクラス。
    pckeibaデータベースの jvd_race_shosai (詳細) と jvd_uma_race (出馬表) を結合して取得します。
    """
    def __init__(self):
        user = os.environ.get('POSTGRES_USER', 'user')
        password = os.environ.get('POSTGRES_PASSWORD', 'password')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
        connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(connection_str)

    def _get_table_name(self, candidates: list[str]) -> str:
        """
        候補リストの中から、データベースに実際に存在するテーブル名を返します。
        """
        try:
            inspector = inspect(self.engine)
            existing_tables = set(inspector.get_table_names(schema='public'))
            
            for cand in candidates:
                if cand in existing_tables:
                    logger.info(f"テーブル検出: {cand}")
                    return cand
            
            logger.warning(f"推奨テーブルが見つかりません: {candidates}. デフォルトの {candidates[0]} を使用します。")
            return candidates[0]
        except Exception as e:
            logger.warning(f"テーブル存在確認中にエラーが発生しました: {e}. デフォルトの {candidates[0]} を使用します。")
            return candidates[0]

    def load(self, target_date: str = None, race_ids: list[str] = None) -> pd.DataFrame:
        """
        指定された日付またはレースIDリストに基づいて、推論用データをロードします。

        Args:
            target_date (str): 'YYYYMMDD' 形式の日付 strings.
            race_ids (list[str]): レースIDのリスト.

        Returns:
            pd.DataFrame: 推論用データフレーム (結果系カラムはNaN/Noneで埋められます)
        """
        # テーブル名の解決
        # 出馬表データは jvd_uma_race (または uma_race, jvd_ur) に格納されていると想定
        tbl_race = self._get_table_name(['jvd_race_shosai', 'race_shosai', 'jvd_ra'])
        tbl_entry = self._get_table_name(['jvd_uma_race', 'uma_race', 'jvd_ur'])
        tbl_uma = self._get_table_name(['jvd_uma_master', 'uma_master', 'jvd_um'])

        # スキーマ判定 (PC-KEIBAの短縮名対応)
        is_pckeiba_short = (tbl_race == 'jvd_ra' or tbl_race == 'ra')

        # カラム定義
        if is_pckeiba_short:
            logger.info("PC-KEIBA短縮名スキーマ (jvd_ra, jvd_ur) を検出しました。")
            col_title = "r.kyosomei_hondai"
            col_state = """CASE
                WHEN CAST(r.track_code AS INTEGER) BETWEEN 10 AND 22 THEN r.babajotai_code_shiba
                ELSE r.babajotai_code_dirt
            END"""
            # 出馬表テーブル(ur)のカラム想定
            col_horse_name = "ur.bamei"
            col_sex = "ur.seibetsu_code"
            # 出馬表には血統IDが含まれているか確認が必要だが、uma_masterと結合するならそちら優先でも良い
            # jvd_uma_raceにも ketto_toroku_bango はあるはず
            col_sire = "uma.ketto_joho_01a"
            col_mare = "uma.ketto_joho_02a"
            # 負担重量 (Futan Juryo) -> ハンデ等
            col_futan_juryo = "ur.futan_juryo"
        else:
            logger.info("標準スキーマ (jvd_race_shosai, jvd_uma_race) を使用します。")
            col_title = "r.race_mei_honbun"
            col_state = "r.baba_jotai_code"
            
            col_horse_name = "uma.bamei"
            col_sex = "uma.seibetsu_code"
            col_sire = "uma.fushu_ketto_toroku_bango"
            col_mare = "uma.boshu_ketto_toroku_bango"
            col_futan_juryo = "ur.futan_juryo" # 標準でもこの名前と想定

        # クエリ構築
        # 注意: 結果系カラム (rank, time, last_3f, odds, popularity) は未来の情報なので NULL にする
        # weight (馬体重) は当日発表なので、直前でなければ NULL になる可能性が高い
        
        query = f"""
        SELECT
            CONCAT(
                r.kaisai_nen,
                r.keibajo_code,
                r.kaisai_kai,
                r.kaisai_nichime,
                r.race_bango
            ) AS race_id,

            TO_DATE(r.kaisai_nen || r.kaisai_tsukihi, 'YYYYMMDD') AS date,

            r.keibajo_code AS venue,
            r.race_bango::integer AS race_number,
            r.kyori::integer AS distance,
            r.track_code AS surface,
            r.tenko_code AS weather,
            {col_state} AS state,
            {col_title} AS title,

            ur.ketto_toroku_bango AS horse_id,
            ur.kishu_code AS jockey_id,
            ur.chokyoshi_code AS trainer_id,
            ur.wakuban::integer AS frame_number,
            ur.umaban::integer AS horse_number,

            -- Future Target Columns (Set to NULL)
            NULL AS rank_str,
            NULL AS raw_time,
            NULL AS last_3f,
            NULL AS odds,
            NULL AS popularity,

            -- Weight (Bataiju) might be available if executed just before race
            -- If not, ur.bataiju might be NULL. 
            -- Note: jvd_uma_race normally has 'bataiju' column if updating from Sokuhou.
            ur.bataiju AS weight,
            ur.zogen_sa AS weight_diff_val,
            ur.zogen_fugo AS weight_diff_sign,

            ur.barei AS age,

            {col_horse_name} AS horse_name,
            {col_sex} AS sex,
            {col_sire} AS sire_id,
            {col_mare} AS mare_id,

            -- Passing Rank (NULL)
            NULL AS pass_1,
            NULL AS pass_2,
            NULL AS pass_3,
            NULL AS pass_4

        FROM {tbl_race} r
        JOIN {tbl_entry} ur
            ON r.kaisai_nen = ur.kaisai_nen
            AND r.keibajo_code = ur.keibajo_code
            AND r.kaisai_kai = ur.kaisai_kai
            AND r.kaisai_nichime = ur.kaisai_nichime
            AND r.race_bango = ur.race_bango
        LEFT JOIN {tbl_uma} uma
            ON ur.ketto_toroku_bango = uma.ketto_toroku_bango

        WHERE 1=1
        """

        # フィルタリング条件
        if target_date:
            query += f" AND (r.kaisai_nen || r.kaisai_tsukihi) = '{target_date}'"
        
        if race_ids:
            # race_ids list format: ['202401010101', ...]
            # Need to parse or assume format. 
            # For simplicity, if race_ids provided, assume user handles parsing or logic outside, 
            # OR implement filtering by parsing race_id to components.
            # PC-KEIBA DB usually indices by (nen, keibajo, kai, nichime, race_no)
            # Construction of ID in SQL: CONCAT(...)
            # So checking IN (...) on the CONCAT result is valid but slow.
            ids_str = ",".join([f"'{rid}'" for rid in race_ids])
            query += f"""
            AND CONCAT(
                r.kaisai_nen,
                r.keibajo_code,
                r.kaisai_kai,
                r.kaisai_nichime,
                r.race_bango
            ) IN ({ids_str})
            """

        query += " ORDER BY date, race_id, horse_number"

        logger.info("JRA-VAN推論用データをロード中...")
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"ロード件数: {len(df)} 件")

            if len(df) == 0:
                logger.warning("対象のレースデータが見つかりません。")
                return df

            # --- Python側での前処理 (Loader共通処理) ---
            
            # 型変換 (NULL許容)
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

            # OddsなどはNULLのまま

            def convert_weight_diff(row):
                try:
                    val = int(row['weight_diff_val'])
                    sign = row['weight_diff_sign']
                    if sign == '-': return -val
                    return val
                except:
                    return 0 # 推論時は変化なし(0)と仮定するか、NULLにするか。ここでは0
            df['weight_diff'] = df.apply(convert_weight_diff, axis=1)

            # 性別マッピング
            sex_map = {1: '牡', 2: '牝', 3: 'セ'}
            df['sex'] = pd.to_numeric(df['sex'], errors='coerce').map(sex_map).fillna('Unknown')

            # 天候マッピング
            weather_map = {1: '晴', 2: '曇', 3: '雨', 4: '小雨', 5: '雪', 6: '雪'}
            df['weather'] = pd.to_numeric(df['weather'], errors='coerce').map(weather_map).fillna('Unknown')

            # トラックコードマッピング
            def map_surface(code):
                try:
                    c = int(code)
                    if 10 <= c <= 22: return '芝'
                    if 23 <= c <= 29: return 'ダート'
                    if 51 <= c <= 59: return '障害'
                    return 'Unknown'
                except:
                    return 'Unknown'
            df['surface'] = df['surface'].apply(map_surface)

            # 馬場状態マッピング
            state_map = {1: '良', 2: '稍重', 3: '重', 4: '不良'}
            df['state'] = pd.to_numeric(df['state'], errors='coerce').map(state_map).fillna('Unknown')

            # Placeholder columns for compatibility with Preprocessing Pipeline
            # パイプラインが rank などを期待する場合、NaNで埋めておく
            df['rank'] = np.nan
            df['time'] = np.nan
            df['passing_rank'] = None 

            logger.info(f"推論用データロード完了: {len(df)} 件")
            return df

        except Exception as e:
            logger.error(f"推論用データロード中にエラーが発生しました: {e}")
            raise e

import numpy as np

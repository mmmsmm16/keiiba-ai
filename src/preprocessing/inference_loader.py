import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import ProgrammingError
import logging

logger = logging.getLogger(__name__)

class SokuhoDataLoader:
    """
    PC-KEIBAの速報データ（apd_sokuho_ra, apd_sokuho_se）から
    開催当日の出馬表データをロードするクラス。
    """
    def __init__(self):
        user = os.environ.get('POSTGRES_USER', 'user')
        password = os.environ.get('POSTGRES_PASSWORD', 'password')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
        connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(connection_str)

    def load(self, target_date: str = None) -> pd.DataFrame:
        """
        速報データをロードします。

        Args:
            target_date (str): 取得対象の日付（YYYYMMDD形式）。指定がない場合はテーブル内の全データを取得。

        Returns:
            pd.DataFrame: 前処理パイプラインへの入力形式に合わせたDataFrame
        """
        # PC-KEIBA 速報テーブル（固定）
        tbl_race = "apd_sokuho_ra"
        tbl_seiseki = "apd_sokuho_se"
        tbl_uma = "jvd_um" # 血統情報はマスタから取得

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

            -- State (Baba)
            CASE
                WHEN CAST(r.track_code AS INTEGER) BETWEEN 10 AND 22 THEN r.babajotai_code_shiba
                ELSE r.babajotai_code_dirt
            END AS state,

            r.kyosomei_hondai AS title,

            res.ketto_toroku_bango AS horse_id,
            res.kishu_code AS jockey_id,
            res.chokyoshi_code AS trainer_id,
            res.wakuban::integer AS frame_number,
            res.umaban::integer AS horse_number,

            -- Results (Future data -> NULL)
            NULL AS rank_str,
            NULL AS raw_time,
            NULL AS last_3f,

            -- Real-time info (Odds/Weight might be available)
            res.tansho_odds AS odds,
            res.tansho_ninkijun AS popularity,
            res.bataiju AS weight,
            res.zogen_sa AS weight_diff_val,
            res.zogen_fugo AS weight_diff_sign,

            res.barei AS age,

            res.bamei AS horse_name,
            res.seibetsu_code AS sex,

            -- Pedigree
            uma.ketto_joho_01a AS sire_id,
            uma.ketto_joho_02a AS mare_id,

            -- Passing Rank (Future -> NULL)
            NULL AS pass_1, NULL AS pass_2, NULL AS pass_3, NULL AS pass_4

        FROM {tbl_race} r
        JOIN {tbl_seiseki} res
            ON r.kaisai_nen = res.kaisai_nen
            AND r.keibajo_code = res.keibajo_code
            AND r.kaisai_kai = res.kaisai_kai
            AND r.kaisai_nichime = res.kaisai_nichime
            AND r.race_bango = res.race_bango
        LEFT JOIN {tbl_uma} uma
            ON res.ketto_toroku_bango = uma.ketto_toroku_bango
        """

        if target_date:
            query += f" WHERE r.kaisai_nen || r.kaisai_tsukihi = '{target_date}'"

        query += " ORDER BY date, race_id, horse_number"

        logger.info(f"速報データをロード中... (Date: {target_date})")
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"ロード件数: {len(df)} 件")

            if len(df) == 0:
                logger.warning("速報データが0件です。PC-KEIBAで「速報データの登録」が行われているか確認してください。")
                return df

            # --- Minimal Preprocessing ---
            # 基本的な型変換等は JraVanDataLoader と合わせる

            # Rank (None)
            df['rank'] = np.nan

            # Time (None)
            df['time'] = np.nan

            # Weight Diff
            def convert_weight_diff(row):
                try:
                    val = int(row['weight_diff_val'])
                    sign = row['weight_diff_sign']
                    if sign == '-': return -val
                    return val
                except:
                    return 0
            df['weight_diff'] = df.apply(convert_weight_diff, axis=1)

            # Passing Rank (None)
            df['passing_rank'] = None

            # Mapping (Sex, Weather, Surface, State)
            sex_map = {1: '牡', 2: '牝', 3: 'セ'}
            df['sex'] = pd.to_numeric(df['sex'], errors='coerce').map(sex_map).fillna('Unknown')

            weather_map = {1: '晴', 2: '曇', 3: '雨', 4: '小雨', 5: '雪', 6: '雪'}
            df['weather'] = pd.to_numeric(df['weather'], errors='coerce').map(weather_map).fillna('Unknown')

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

            state_map = {1: '良', 2: '稍重', 3: '重', 4: '不良'}
            df['state'] = pd.to_numeric(df['state'], errors='coerce').map(state_map).fillna('Unknown')

            return df

        except Exception as e:
            logger.error(f"速報データのロード中にエラーが発生しました: {e}")
            raise e

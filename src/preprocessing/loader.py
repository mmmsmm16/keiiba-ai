import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
import logging

logger = logging.getLogger(__name__)

class RawDataLoader:
    """
    (Deprecated) スクレイピングデータをロードするクラス。
    """
    def __init__(self):
        user = os.environ.get('POSTGRES_USER', 'user')
        password = os.environ.get('POSTGRES_PASSWORD', 'password')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'keiba')
        connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(connection_str)

    def load(self, limit: int = None) -> pd.DataFrame:
        # Legacy implementation...
        return pd.DataFrame()

class JraVanDataLoader:
    """
    PC-KEIBA Database (JRA-VAN) のデータをロードするクラス。
    """
    def __init__(self):
        user = os.environ.get('POSTGRES_USER', 'user')
        password = os.environ.get('POSTGRES_PASSWORD', 'password')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        # PC-KEIBAのDB名は pckeiba
        dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
        connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(connection_str)

    def load(self, limit: int = None) -> pd.DataFrame:
        """
        JRA-VANデータをロードし、学習用フォーマットに変換します。
        """
        # PC-KEIBA (Short Table Names) 対応
        # jvd_ra: Race Info, jvd_se: Seiseki, jvd_um: Uma Master
        query = """
        SELECT
            -- ID Construction (Netkeiba Format: YYYYJJMMDDRR)
            CONCAT(
                r.kaisai_nen,
                r.keibajo_code,
                r.kaisai_kai,
                r.kaisai_nichime,
                r.race_bango
            ) AS race_id,

            -- Date
            TO_DATE(r.kaisai_nen || r.kaisai_tsukihi, 'YYYYMMDD') AS date,

            -- Race Info
            r.keibajo_code AS venue,
            r.race_bango::integer AS race_number,
            r.kyori::integer AS distance,
            r.track_code AS surface,
            r.tenko_code AS weather,
            -- Baba State: Prioritize Shiba if Track is Turf (10-22), else Dirt
            CASE
                WHEN CAST(r.track_code AS INTEGER) BETWEEN 10 AND 22 THEN r.babajotai_code_shiba
                ELSE r.babajotai_code_dirt
            END AS state,
            r.kyosomei_hondai AS title,

            -- Results
            res.ketto_toroku_bango AS horse_id,
            res.kishu_code AS jockey_id,
            res.chokyoshi_code AS trainer_id,
            res.wakuban::integer AS frame_number,
            res.umaban::integer AS horse_number,

            -- Rank
            TRIM(res.kakutei_chakujun) AS rank_str,

            -- Time (1:35.5 -> 1355 or 955)
            res.soha_time AS raw_time,

            -- Other Results
            res.kohan_3f AS last_3f,
            res.tansho_odds AS odds,
            res.tansho_ninkijun AS popularity,
            res.bataiju AS weight,
            res.zogen_sa AS weight_diff_val,
            res.zogen_fugo AS weight_diff_sign,

            res.barei AS age,

            -- Horse Info
            res.bamei AS horse_name,
            res.seibetsu_code AS sex,
            uma.fushu_ketto_toroku_bango AS sire_id,
            uma.boshu_ketto_toroku_bango AS mare_id

        FROM jvd_ra r
        JOIN jvd_se res
            ON r.kaisai_nen = res.kaisai_nen
            AND r.keibajo_code = res.keibajo_code
            AND r.kaisai_kai = res.kaisai_kai
            AND r.kaisai_nichime = res.kaisai_nichime
            AND r.race_bango = res.race_bango
        LEFT JOIN jvd_um uma
            ON res.ketto_toroku_bango = uma.ketto_toroku_bango

        -- 障害レースを除外する場合: WHERE r.track_code NOT IN ('...')
        ORDER BY date, race_id
        """

        if limit:
            query += f" LIMIT {limit}"

        logger.info("JRA-VANデータをロード中...")
        try:
            df = pd.read_sql(query, self.engine)

            # --- Python側での前処理 ---

            # Rank: '1' -> 1, '中止' -> NaN
            df['rank'] = pd.to_numeric(df['rank_str'], errors='coerce')

            # Time: '1355' (1分35秒5) -> 95.5秒
            def convert_time(t):
                if pd.isna(t): return None
                try:
                    t_str = str(int(t)).zfill(4) # 1355
                    minutes = int(t_str[:-3])
                    seconds = int(t_str[-3:-1])
                    dec = int(t_str[-1])
                    return minutes * 60 + seconds + dec * 0.1
                except:
                    return None

            df['time'] = df['raw_time'].apply(convert_time)

            # Weight Diff: sign (+/-) + val
            # sign: '+' or '-' or ' '
            def convert_weight_diff(row):
                try:
                    val = int(row['weight_diff_val'])
                    sign = row['weight_diff_sign']
                    if sign == '-': return -val
                    return val
                except:
                    return 0
            df['weight_diff'] = df.apply(convert_weight_diff, axis=1)

            # Sex: Code mapping
            # JVD: 1=Male, 2=Female, 3=Gelding.
            sex_map = {1: '牡', 2: '牝', 3: 'セ'}
            df['sex'] = pd.to_numeric(df['sex'], errors='coerce').map(sex_map).fillna('Unknown')

            # Weather: 1:晴, 2:曇, 3:雨, 4:小雨, 5:雪, 6:小雪
            weather_map = {1: '晴', 2: '曇', 3: '雨', 4: '小雨', 5: '雪', 6: '雪'}
            df['weather'] = pd.to_numeric(df['weather'], errors='coerce').map(weather_map).fillna('Unknown')

            # Surface (Track):
            # 10-22: 芝, 23-29: ダート, 51-59: 障害
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

            # State (Baba): 1:良, 2:稍重, 3:重, 4:不良
            state_map = {1: '良', 2: '稍重', 3: '重', 4: '不良'}
            df['state'] = pd.to_numeric(df['state'], errors='coerce').map(state_map).fillna('Unknown')

            logger.info(f"ロード完了: {len(df)} 件")
            return df

        except ProgrammingError as e:
            logger.error("データロード中にエラーが発生しました: テーブルまたはカラムが見つかりません。")
            logger.error(f"詳細: {e}")
            logger.error("ヒント: PC-KEIBA Database の設定を確認してください。")
            logger.error("想定しているテーブル名: jvd_ra, jvd_se, jvd_um")
            raise e
        except Exception as e:
            logger.error(f"データロード中にエラーが発生しました: {e}")
            raise e

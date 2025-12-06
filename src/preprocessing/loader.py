import os
import pandas as pd
from sqlalchemy import create_engine, inspect
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

    def _get_table_name(self, candidates: list[str]) -> str:
        """
        候補リストの中から、データベースに実際に存在するテーブル名を返します。
        見つからない場合はリストの先頭を返します。
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

    def load(self, limit: int = None) -> pd.DataFrame:
        """
        JRA-VANデータをロードし、学習用フォーマットに変換します。
        """
        # テーブル名の解決 (jvd_プレフィックスの有無、短縮名に対応)
        tbl_race = self._get_table_name(['jvd_race_shosai', 'race_shosai', 'jvd_ra'])
        tbl_seiseki = self._get_table_name(['jvd_seiseki', 'seiseki', 'jvd_se'])
        tbl_uma = self._get_table_name(['jvd_uma_master', 'uma_master', 'jvd_um'])

        # スキーマ判定: PC-KEIBAの短縮名テーブル (jvd_ra 等) の場合はカラム名が異なる
        is_pckeiba_short = (tbl_race == 'jvd_ra' or tbl_race == 'ra')

        # 通過順 (Passing Rank) 用のカラム定義 (Schema agnostic approach if names match?)
        # PC-KEIBA typically uses corner_1, corner_2... similar to JRA-VAN Spec
        col_pass = ["res.corner_1", "res.corner_2", "res.corner_3", "res.corner_4"]

        if is_pckeiba_short:
            logger.info("PC-KEIBA短縮名スキーマ (jvd_ra) を検出しました。")
            col_title = "r.kyosomei_hondai"
            # 馬場状態: 芝(10-22)ならshiba, それ以外(ダート)ならdirtを優先
            col_state = """CASE
                WHEN CAST(r.track_code AS INTEGER) BETWEEN 10 AND 22 THEN r.babajotai_code_shiba
                ELSE r.babajotai_code_dirt
            END"""
            col_rank = "TRIM(res.kakutei_chakujun)"
            col_last3f = "res.kohan_3f"
            col_pop = "res.tansho_ninkijun"
            col_horse_name = "res.bamei" # 成績テーブルに馬名がある
            col_sex = "res.seibetsu_code" # 成績テーブルに性別がある
            col_sire = "uma.ketto_joho_01a"
            col_mare = "uma.ketto_joho_02a"
        else:
            logger.info("標準スキーマ (jvd_race_shosai) を使用します。")
            col_title = "r.race_mei_honbun"
            col_state = "r.baba_jotai_code"
            col_rank = "TRIM(res.kakutei_chakusun)"
            col_last3f = "res.agari_3f"
            col_pop = "res.ninki"
            col_horse_name = "uma.bamei"
            col_sex = "uma.seibetsu_code"
            # 標準カラム (存在しない場合はエラーになるため、必要に応じて ketto_joho に変更検討)
            col_sire = "uma.fushu_ketto_toroku_bango"
            col_mare = "uma.boshu_ketto_toroku_bango"

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

            res.ketto_toroku_bango AS horse_id,
            res.kishu_code AS jockey_id,
            res.chokyoshi_code AS trainer_id,
            res.wakuban::integer AS frame_number,
            res.umaban::integer AS horse_number,

            {col_rank} AS rank_str,
            res.soha_time AS raw_time,

            {col_last3f} AS last_3f,
            res.tansho_odds AS odds,
            {col_pop} AS popularity,
            res.bataiju AS weight,
            res.zogen_sa AS weight_diff_val,
            res.zogen_fugo AS weight_diff_sign,

            res.barei AS age,

            {col_horse_name} AS horse_name,
            {col_sex} AS sex,
            {col_sire} AS sire_id,
            {col_mare} AS mare_id,

            -- Passing Rank Columns
            {col_pass[0]} AS pass_1,
            {col_pass[1]} AS pass_2,
            {col_pass[2]} AS pass_3,
            {col_pass[3]} AS pass_4

        FROM {tbl_race} r
        JOIN {tbl_seiseki} res
            ON r.kaisai_nen = res.kaisai_nen
            AND r.keibajo_code = res.keibajo_code
            AND r.kaisai_kai = res.kaisai_kai
            AND r.kaisai_nichime = res.kaisai_nichime
            AND r.race_bango = res.race_bango
        LEFT JOIN {tbl_uma} uma
            ON res.ketto_toroku_bango = uma.ketto_toroku_bango

        ORDER BY date, race_id
        """

        if limit:
            query += f" LIMIT {limit}"

        logger.info("JRA-VANデータをロード中...")
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"ロード件数 (重複削除前): {len(df)} 件")

            # 【重要】重複データの削除
            # race_id と horse_number (馬番) の組み合わせはユニークであるはず
            before_len = len(df)
            df.drop_duplicates(subset=['race_id', 'horse_number'], inplace=True)
            after_len = len(df)
            if before_len != after_len:
                logger.warning(f"重複データを削除しました: {before_len} -> {after_len} 件 (削除: {before_len - after_len} 件)")

            if len(df) == 0:
                logger.warning("注意: 取得データが0件です。データベースが空か、結合条件に一致するデータがありません。")
                logger.warning("src/tools/diagnose_db.py を実行してテーブルの行数を確認してください。")
                return df

            # --- Python側での前処理 ---
            
            # 基本的な型変換
            df['rank'] = pd.to_numeric(df['rank_str'], errors='coerce')
            
            # 【修正】オッズを 1/10 にする (JRA-VAN仕様: 1.5倍 -> 15)
            # 文字列 '---' などは NaN になる
            df['odds'] = pd.to_numeric(df['odds'], errors='coerce') / 10.0
            
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

            def convert_time(t):
                if pd.isna(t): return None
                try:
                    t_str = str(int(t)).zfill(4)
                    minutes = int(t_str[:-3])
                    seconds = int(t_str[-3:-1])
                    dec = int(t_str[-1])
                    return minutes * 60 + seconds + dec * 0.1
                except:
                    return None
            df['time'] = df['raw_time'].apply(convert_time)

            def convert_weight_diff(row):
                try:
                    val = int(row['weight_diff_val'])
                    sign = row['weight_diff_sign']
                    if sign == '-': return -val
                    return val
                except:
                    return 0
            df['weight_diff'] = df.apply(convert_weight_diff, axis=1)

            # Passing Rank生成 ("1-1-1-1")
            pass_cols = ['pass_1', 'pass_2', 'pass_3', 'pass_4']
            def make_passing_rank(row):
                # 数値または文字列のコーナー順位を結合
                vals = []
                for c in pass_cols:
                    v = row.get(c)
                    if pd.notnull(v) and v != 0 and v != '0':
                        vals.append(str(v).replace('.0', '')) # 1.0 -> 1
                if not vals:
                    return None
                return "-".join(vals)
            df['passing_rank'] = df.apply(make_passing_rank, axis=1)

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

            logger.info(f"前処理完了: {len(df)} 件")
            return df

        except ProgrammingError as e:
            logger.error("データロード中にエラーが発生しました: テーブルまたはカラムが見つかりません。")
            logger.error(f"詳細: {e}")
            logger.error(f"判定スキーマ: {'PC-KEIBA Short' if is_pckeiba_short else 'Standard'}")
            logger.error(f"試行したテーブル名: {tbl_race}, {tbl_seiseki}, {tbl_uma}")
            raise e
        except Exception as e:
            logger.error(f"データロード中にエラーが発生しました: {e}")
            raise e
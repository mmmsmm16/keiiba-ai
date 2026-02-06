import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import ProgrammingError
import logging

logger = logging.getLogger(__name__)

class NarDataLoader:
    """
    PC-KEIBA Database (NAR/Chiho) のデータをロードするクラス。
    JraVanDataLoader をベースに、nvd_ プレフィックスのテーブルを使用するように調整。
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

    def load(self, limit: int = None, history_start_date: str = "2014-01-01", region: str = "south_kanto") -> pd.DataFrame:
        """
        NARデータをロードし、学習用フォーマットに変換します。
        
        Args:
            limit (int, optional): ロードする件数の上限
            history_start_date (str, optional): この日付以降のデータのみを読み込む (YYYY-MM-DD)
            region (str, optional): 'south_kanto' で南関東4場に限定。None なら全データ。
        """
        logger.info(f"NARデータ読み込み開始: history_start_date={history_start_date}, region={region}")
        
        # テーブル名の解決 (nvd_ra 等)
        # PC-KEIBA pattern for NAR is likely nvd_ra, nvd_se, nvd_um
        tbl_race = self._get_table_name(['nvd_ra', 'nvd_race_shosai'])
        tbl_seiseki = self._get_table_name(['nvd_se', 'nvd_seiseki'])
        tbl_uma = self._get_table_name(['nvd_um', 'nvd_uma_master'])

        # スキーマ判定: PC-KEIBAの短縮名テーブル (nvd_ra 等) の場合はカラム名が異なる前提
        # JRA版と同じカラム体系と仮定するが、プレフィックスが nvd_ になってもカラム名は共通のことが多い
        is_pckeiba_short = 'ra' in tbl_race # nvd_ra

        # 通過順 (Passing Rank) 用のカラム定義
        col_pass = ["res.corner_1", "res.corner_2", "res.corner_3", "res.corner_4"]

        if is_pckeiba_short:
            logger.info("PC-KEIBA短縮名スキーマ (nvd_ra) を検出しました。")
            col_title = "r.race_mei_honbun" # nvd_ra may use race_mei_honbun or kyosomei_hondai? Let's assume standard unless proven otherwise.
            # Warning: JRA (jvd_ra) uses kyosomei_hondai, but let's check if nvd_ra is same.
            # Usually PC-KEIBA standardizes column names.
            col_title = "r.kyosomei_hondai" 
            # クラス・条件等
            col_grade = "r.grade_code"
            col_kyoso_shubetsu = "r.kyoso_shubetsu_code"
            col_kyoso_joken = "r.kyoso_joken_code"
            # タイム詳細
            col_lap_time = "r.lap_time"
            col_zenhan_3f = "r.zenhan_3f"
            
            # 馬場状態: NARはほぼダートだが、コード体系は共通と仮定
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
            # Fallback (Unlikely to be hit for nvd)
            logger.info("標準スキーマ (nvd_race_shosai) を使用します。")
            col_title = "r.race_mei_honbun"
            col_state = "r.baba_jotai_code"
            col_rank = "TRIM(res.kakutei_chakusun)"
            col_last3f = "res.agari_3f"
            col_pop = "res.ninki"
            col_horse_name = "uma.bamei"
            col_sex = "uma.seibetsu_code"
            col_sire = "uma.fushu_ketto_toroku_bango"
            col_mare = "uma.boshu_ketto_toroku_bango"
            # クラス・条件等 (Fallback)
            col_grade = "r.grade_code"
            col_kyoso_shubetsu = "r.kyoso_shubetsu_code"
            col_kyoso_joken = "r.kyoso_joken_code"
            col_lap_time = "r.lap_time"
            col_zenhan_3f = "r.zenhan_3f"

        # フィルタリング条件の構築
        where_clauses = []
        
        if history_start_date:
            date_filter = history_start_date.replace('-', '')
            where_clauses.append(f"CONCAT(r.kaisai_nen, r.kaisai_tsukihi) >= '{date_filter}'")
        
        # NAR specific filters if needed (e.g. exclude Ban-ei if track code differs? Ban-ei is Obitu 30?)
        # For now, load everything in nvd_ra
        
        where_str = ""
        if where_clauses:
            where_str = "WHERE " + " AND ".join(where_clauses)

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
            {col_grade} AS grade_code,
            {col_kyoso_shubetsu} AS kyoso_shubetsu_code,
            {col_kyoso_joken} AS kyoso_joken_code,
            r.kyoso_joken_meisho AS class_name,
            {col_lap_time} AS lap_time,
            {col_zenhan_3f} AS zenhan_3f,

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

            -- 斤量
            res.futan_juryo AS impost,

            -- Passing Rank Columns
            {col_pass[0]} AS pass_1,
            {col_pass[1]} AS pass_2,
            {col_pass[2]} AS pass_3,
            {col_pass[3]} AS pass_4,

            -- 異常コード (Accident Code)
            res.ijo_kubun_code AS accident_code

        FROM {tbl_race} r
        JOIN {tbl_seiseki} res
            ON r.kaisai_nen = res.kaisai_nen
            AND r.keibajo_code = res.keibajo_code
            AND r.kaisai_kai = res.kaisai_kai
            AND r.kaisai_nichime = res.kaisai_nichime
            AND r.race_bango = res.race_bango
        LEFT JOIN {tbl_uma} uma
            ON res.ketto_toroku_bango = uma.ketto_toroku_bango

        {where_str}

        ORDER BY date, race_id
        """

        if limit:
            query += f" LIMIT {limit}"

        logger.info("PC-KEIBA (NAR) データをロード中...")
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"ロード件数 (重複削除前): {len(df)} 件")

            before_len = len(df)
            df.drop_duplicates(subset=['race_id', 'horse_number'], inplace=True)
            after_len = len(df)
            if before_len != after_len:
                logger.warning(f"重複データを削除しました: {before_len} -> {after_len} 件")

            if len(df) == 0:
                logger.warning("注意: 取得データが0件です。")
                return df

            # --- [NAR Extension] Filtering ---
            # region='south_kanto' の場合、南関東 (42,43,44,45) に限定
            if region == 'south_kanto':
                logger.info("南関東 (浦和・船橋・大井・川崎) のレースのみ抽出します。")
                south_kanto_codes = [42, 43, 44, 45]
                # venueカラムが数値か文字列か確認しながらフィルタ
                # PC-KEIBA returns numeric codes mostly
                try:
                    df['venue'] = pd.to_numeric(df['venue'], errors='coerce')
                    before_filter_len = len(df)
                    df = df[df['venue'].isin(south_kanto_codes)]
                    logger.info(f"フィルタ結果: {before_filter_len} -> {len(df)} 件")
                except Exception as e:
                    logger.warning(f"Region filtering failed: {e}")

            # --- Python側での前処理 ---
            # 共通処理としてJRAと同じロジックを適用
            
            df['rank'] = pd.to_numeric(df['rank_str'], errors='coerce')
            # NARのオッズも10倍値か？PC-KEIBA仕様ならおそらくYes。後でデータが入ったら要確認。
            df['odds'] = pd.to_numeric(df['odds'], errors='coerce') / 10.0
            
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df['impost'] = pd.to_numeric(df['impost'], errors='coerce') / 10.0

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

            # Passing Rank
            pass_cols = ['pass_1', 'pass_2', 'pass_3', 'pass_4']
            def make_passing_rank(row):
                vals = []
                for c in pass_cols:
                    v = row.get(c)
                    if pd.notnull(v) and v != 0 and v != '0':
                        vals.append(str(v).replace('.0', ''))
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

            # トラックコードマッピング (NAR)
            # NAR独自のコードがある可能性ありだが、とりあえずPC-KEIBA共通と仮定
            def map_surface(code):
                try:
                    c = int(code)
                    if 10 <= c <= 22: return '芝'
                    if 23 <= c <= 29: return 'ダート'
                    # ばんえいは？ -> 30? 要確認
                    if 51 <= c <= 59: return '障害'
                    return 'Unknown'
                except:
                    return 'Unknown'
            df['surface'] = df['surface'].apply(map_surface)

            # 馬場状態マッピング
            state_map = {1: '良', 2: '稍重', 3: '重', 4: '不良'}
            df['state'] = pd.to_numeric(df['state'], errors='coerce').map(state_map).fillna('Unknown')

            if 'date' in df.columns and len(df) > 0:
                date_min = df['date'].min()
                date_max = df['date'].max()
                n_races = df['race_id'].nunique()
                logger.info(f"データ読み込み期間: {date_min} ~ {date_max}")
                logger.info(f"レース数: {n_races:,}件, レコード数: {len(df):,}件")

            logger.info(f"前処理完了: {len(df)} 件")
            return df

        except ProgrammingError as e:
            logger.error("NARデータロード中にエラーが発生しました: テーブル不一致の可能性があります。")
            logger.error(f"詳細: {e}")
            raise e
        except Exception as e:
            logger.error(f"NARデータロード中にエラーが発生しました: {e}")
            raise e

    def load_payouts(self, date_str: str, race_ids: list = None) -> pd.DataFrame:
        """
        払戻金データ (nvd_hr) をロードします。
        """
        year = date_str[:4]
        mmdd = date_str[5:7] + date_str[8:10]
        
        query = f"""
        SELECT 
            CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
            haraimodoshi_tansho_1a AS payout_win_horse_1,
            haraimodoshi_tansho_1b AS payout_win_amount_1,
            haraimodoshi_fukusho_1a AS payout_place_horse_1,
            haraimodoshi_fukusho_1b AS payout_place_amount_1,
            haraimodoshi_fukusho_2a AS payout_place_horse_2,
            haraimodoshi_fukusho_2b AS payout_place_amount_2,
            haraimodoshi_fukusho_3a AS payout_place_horse_3,
            haraimodoshi_fukusho_3b AS payout_place_amount_3,
            haraimodoshi_umaren_1a AS payout_umaren_combo_1,
            haraimodoshi_umaren_1b AS payout_umaren_amount_1
        FROM nvd_hr
        WHERE kaisai_nen = '{year}' AND kaisai_tsukihi = '{mmdd}'
        """
        df = pd.read_sql(query, self.engine)
        if race_ids:
            df = df[df['race_id'].isin(race_ids)]
        return df

    def load_human_master(self) -> dict:
        """
        騎手 (nvd_ks) および 調教師 (nvd_ch) マスタをロードします。
        """
        try:
            jockeys = pd.read_sql("SELECT kishu_code, kishumei, tozai_shozoku_code FROM nvd_ks", self.engine)
            trainers = pd.read_sql("SELECT chokyoshi_code, chokyoshimei, tozai_shozoku_code FROM nvd_ch", self.engine)
        except:
            jockeys = pd.read_sql("SELECT kishu_code, kishumei FROM nvd_ks", self.engine)
            trainers = pd.read_sql("SELECT chokyoshi_code, chokyoshimei FROM nvd_ch", self.engine)
            
        return {
            'jockeys': jockeys,
            'trainers': trainers
        }

    def load_bloodline_master(self, horse_ids: list = None) -> pd.DataFrame:
        """
        血統情報を含む馬マスタ (nvd_um) をロードします。
        """
        query = "SELECT ketto_toroku_bango as horse_id, bamei, ketto_joho_01a, ketto_joho_02a, ketto_joho_03a FROM nvd_um"
        if horse_ids:
            id_str = ",".join([f"'{h}'" for h in horse_ids])
            query += f" WHERE ketto_toroku_bango IN ({id_str})"
        
        df = pd.read_sql(query, self.engine)
        return df

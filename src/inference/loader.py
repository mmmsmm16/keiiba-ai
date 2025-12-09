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
    def load_race_list(self, target_date: str) -> pd.DataFrame:
        """
        指定された日付の開催レース一覧を取得します。
        
        Args:
            target_date (str): 'YYYY-MM-DD' or 'YYYYMMDD'
        """
        tbl_race = self._get_table_name(['jvd_race_shosai', 'race_shosai', 'jvd_ra'])
        is_pckeiba_short = (tbl_race == 'jvd_ra' or tbl_race == 'ra')
        
        col_title = "r.kyosomei_hondai" if is_pckeiba_short else "r.race_mei_honbun"

        query = f"""
        SELECT
            CONCAT(r.kaisai_nen, r.keibajo_code, r.kaisai_kai, r.kaisai_nichime, r.race_bango) AS race_id,
            r.keibajo_code AS venue,
            r.race_bango::integer AS race_number,
            {col_title} AS title,
            r.hasso_jikoku AS start_time
        FROM {tbl_race} r
        WHERE (r.kaisai_nen || r.kaisai_tsukihi) = '{target_date.replace('-', '')}'
        ORDER BY r.keibajo_code, r.race_bango
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            logger.error(f"レース一覧取得エラー: {e}")
            return pd.DataFrame()

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
        # jvd_se (成績) も候補に追加 (過去レースの推論や、スキーマによってはエントリ含むため)
        tbl_race = self._get_table_name(['jvd_race_shosai', 'race_shosai', 'jvd_ra'])
        tbl_entry = self._get_table_name(['jvd_uma_race', 'uma_race', 'jvd_ur', 'jvd_se'])
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
        
        # jvd_se (成績) テーブルの解決
        tbl_se = self._get_table_name(['jvd_se', 'seiseki', 'jvd_ur']) # urと同じ場合もあるがse優先

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

            -- Fetch Results (Rank, Passing Rank) for Context Features
            -- se (Results) takes precedence over ur (Entry) if available
            COALESCE(se.kakutei_chakujun, NULL)::numeric AS rank,
            -- se.tsuka_jun AS passing_rank,
            NULL AS passing_rank,
            
            NULL AS rank_str, -- Deprecated/Unused downstream but kept for schema compat
            NULL AS raw_time,
            NULL AS last_3f,
            
            -- Prioirty: Realtime Odds > Result Odds > Entry Odds
            ur.tansho_odds AS odds,
            ur.tansho_ninkijun AS popularity,

            ur.bataiju AS weight,
            ur.zogen_sa AS weight_diff_val,
            ur.zogen_fugo AS weight_diff_sign,

            ur.barei AS age,

            {col_horse_name} AS horse_name,
            {col_sex} AS sex,
            {col_sire} AS sire_id,
            {col_mare} AS mare_id,

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
            
        -- Join Results table to get results of FINISHED races for context
        LEFT JOIN {tbl_se} se
            ON ur.kaisai_nen = se.kaisai_nen
            AND ur.keibajo_code = se.keibajo_code
            AND ur.kaisai_kai = se.kaisai_kai
            AND ur.kaisai_nichime = se.kaisai_nichime
            AND ur.race_bango = se.race_bango
            AND ur.umaban = se.umaban

        WHERE 1=1
        """


        # フィルタリング条件
        if target_date:
            # target_date might be YYYY-MM-DD matches typical string input. 
            # DB stores nen/tsukihi as strings without hyphens.
            flat_date = target_date.replace('-', '')
            query += f" AND (r.kaisai_nen || r.kaisai_tsukihi) = '{flat_date}'"
        
        if race_ids:
            # race_idが指定されている場合でも、リアルタイム特徴量のために「その日の全レース」が必要。
            # しかし、現在のスキーマのrace_id (YYYYJJNNRR) からは日付(MMDD)を直接抽出できない。
            # そのため、target_date が指定されている場合はそれを優先し、
            # target_date がない場合は race_ids からの推測を試みるが、スキーマ的に失敗する可能性が高い。
            
            if target_date:
                # target_dateですでにフィルタされているので、ここでの追加フィルタは不要
                # (race_idsによる絞り込みは行わず、返り値の全レースから呼び出し元が抽出する想定)
                pass
            else:
                 # race_ids for filtering without target_date
                 # Optimized: Use tuple matching instead of CONCAT for index usage
                 # race_id structure: YYYY(4) + Venue(2) + Kai(2) + Nichi(2) + Race(2)
                 val_list = []
                 for rid in race_ids:
                     if len(rid) == 12:
                         t = f"('{rid[0:4]}', '{rid[4:6]}', '{rid[6:8]}', '{rid[8:10]}', '{rid[10:12]}')"
                         val_list.append(t)
                 
                 if val_list:
                     in_clause = ",".join(val_list)
                     query += f"""
                     AND (r.kaisai_nen, r.keibajo_code, r.kaisai_kai, r.kaisai_nichime, r.race_bango) IN ({in_clause})
                     """
                 logger.info(f"Filtering by {len(race_ids)} race_ids.")



        query += " ORDER BY date, race_id, horse_number"

        logger.info("JRA-VAN推論用データをロード中...")
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"ロード件数: {len(df)} 件")

            if len(df) == 0:
                logger.warning("対象のレースデータが見つかりません。")
                return df

            # --- Real-time Odds Overlay ---
            try:
                # ターゲット日付がない場合、dfから取得 (複数の可能性あるが、推論用なら通常同一日)
                t_date = target_date
                if not t_date and not df.empty and 'date' in df.columns:
                    # date format in df is datetime or timestamp
                    t_date = df['date'].iloc[0].strftime('%Y%m%d')

                # race_ids が None なら df から抽出
                r_ids = race_ids
                if r_ids is None and not df.empty:
                    r_ids = df['race_id'].unique().tolist()

                rt_df = self._load_realtime_odds(t_date, r_ids)
                
                if not rt_df.empty:
                    logger.info(f"Real-time Odds (jvd_o1) detected: {len(rt_df)} records. Merging...")
                    # Merge on race_id, horse_number
                    # df may have 'odds', 'popularity' with None or Old values
                    # rt_df has 'rt_odds', 'rt_popularity'
                    
                    df = pd.merge(df, rt_df, on=['race_id', 'horse_number'], how='left')
                    
                    # Update (Coalesce)
                    # Use rt_odds if available
                    df['odds'] = df['rt_odds'].fillna(df['odds'])
                    df['popularity'] = df['rt_popularity'].fillna(df['popularity'])
                    
                    df.drop(columns=['rt_odds', 'rt_popularity'], inplace=True)
            except Exception as e:
                logger.warning(f"Failed to merge real-time odds: {e}")

            # --- Python側での前処理 (Loader共通処理) ---
            
            # 型変換 (NULL許容)
            # tansho_ninkijun (人気順) は整数なのでそのまま変換
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

            # Odds を数値に変換（PC-KEIBAフォーマット対応）
            # tansho_odds は '0037' = 3.7倍 のように10倍されているので10で割る
            # lag1_odds 生成のため、数値型である必要がある
            df['odds'] = pd.to_numeric(df['odds'], errors='coerce') / 10.0

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
            # パイプラインが rank などを期待する場合、NaNで埋めておく -> RealTimeFeatureのために残す
            # df['rank'] = np.nan
            # df['time'] = np.nan
            # df['passing_rank'] = None 

            logger.info(f"推論用データロード完了: {len(df)} 件")
            return df

        except Exception as e:
            logger.error(f"推論用データロード中にエラーが発生しました: {e}")
            raise e

    def _load_realtime_odds(self, target_date: str, race_ids: list[str] = None) -> pd.DataFrame:
        """
        jvd_o1 テーブルからリアルタイムオッズ（単勝・人気）を取得し、パースして返します。
        """
        # テーブル存在確認
        tbl = self._get_table_name(['jvd_o1', 'apd_sokuho_o1'])
        if not tbl:
            return pd.DataFrame()
            
        # クエリ: 最新のものを取得するために happyo_tsukihi_jifun でソート
        query = f"""
        SELECT
            CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
            happyo_tsukihi_jifun,
            odds_tansho
        FROM {{}}
        WHERE 1=1
        """.format(tbl)
        
        if target_date:
            flat_date = target_date.replace('-', '')
            query += f" AND (kaisai_nen || kaisai_tsukihi) = '{{}}'".format(flat_date)
        
        # race_ids フィルタがあれば適用 (高速化)
        if race_ids:
             ids_str = ",".join([f"'{{}}'".format(rid) for rid in race_ids])
             query += f"""
             AND CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) IN ({{}})
             """.format(ids_str)

        try:
            df = pd.read_sql(query, self.engine)
        except Exception as e:
            logger.warning(f"ODDS Load Error: {{}}".format(e))
            return pd.DataFrame()
        
        if df.empty:
            return pd.DataFrame()
            
        # レースごとに最新のレコードを取得
        # happyo_tsukihi_jifun が文字列時刻(YYYYMMDDHHMMSS or similar)と仮定
        if 'happyo_tsukihi_jifun' in df.columns:
            df = df.sort_values('happyo_tsukihi_jifun', ascending=False)
            df = df.drop_duplicates(subset=['race_id'], keep='first')
        
        # パース処理
        records = []
        for _, row in df.iterrows():
            race_id = row['race_id']
            raw_odds = row['odds_tansho']
            
            if not raw_odds or len(raw_odds) < 8:
                continue
                
            # 固定長パース: 8バイト x 28頭 = 224バイト
            # [馬番(2)][単勝(4)][人気(2)]
            for i in range(28):
                offset = i * 8
                if offset + 8 > len(raw_odds):
                    break
                
                chunk = raw_odds[offset : offset+8]
                umaban_str = chunk[0:2]
                odds_str = chunk[2:6]
                ninki_str = chunk[6:8]
                
                # Validation
                try:
                    umaban = int(umaban_str)
                    if umaban == 0: continue # Empty slot
                    
                    # Odds
                    # '0000' means invalid or none
                    if odds_str == '0000':
                        odds_val = None
                    else:
                        odds_val = int(odds_str) # Keep as Raw Integer for compatibility
                    
                    # Popularity
                    if ninki_str == '00':
                        pop_val = None
                    else:
                        pop_val = int(ninki_str)
                        
                    records.append({
                        'race_id': race_id,
                        'horse_number': umaban,
                        'rt_odds': odds_val,
                        'rt_popularity': pop_val
                    })
                except ValueError:
                    continue
                    
        return pd.DataFrame(records)

    def load_complex_odds(self, target_date: str, race_ids: list[str] = None) -> dict:
        """
        馬連(O2), ワイド(O3), 馬単(O4), 3連複(O5), 3連単(O6) のオッズを取得して辞書で返します。
        Review: Heavy operation. Use only when necessary.
        Returns: { 'race_id': { 'umaren': { '0102': 12.3, ... }, ... } }
        """
        # Tables to fetch
        targets = {
            'jvd_o2': {'type': 'umaren', 'len': 13, 'col': 'odds_umaren'},
            'jvd_o3': {'type': 'wide', 'len': 17, 'col': 'odds_wide'},
            'jvd_o4': {'type': 'umatan', 'len': 13, 'col': 'odds_umatan'},
            'jvd_o5': {'type': 'sanrenpuku', 'len': 15, 'col': 'odds_sanrenpuku'},
            'jvd_o6': {'type': 'sanrentan', 'len': 17, 'col': 'odds_sanrentan'}
        }
        
        # Result container
        # structure: { race_id: { umaren: {}, wide: {}, ... } }
        results = {}

        # 1. Determine Tables exist
        inspector = inspect(self.engine)
        existing_tables = set(inspector.get_table_names(schema='public'))

        flat_date = target_date.replace('-', '')
        ids_cond = ""
        if race_ids:
             ids_str = ",".join([f"'{rid}'" for rid in race_ids])
             ids_cond = f"AND CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) IN ({ids_str})"

        for tbl, info in targets.items():
            if tbl not in existing_tables:
                continue

            ticket_type = info['type']
            byte_len = info['len']
            col_name = info['col']
            
            # Query
            query = f"""
            SELECT
                CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
                happyo_tsukihi_jifun,
                {col_name}
            FROM {tbl}
            WHERE (kaisai_nen || kaisai_tsukihi) = '{flat_date}'
            {ids_cond}
            ORDER BY happyo_tsukihi_jifun DESC
            """
            
            try:
                df = pd.read_sql(query, self.engine)
                if df.empty:
                    continue
                    
                # Dedup (Latest)
                if 'happyo_tsukihi_jifun' in df.columns:
                     df = df.sort_values('happyo_tsukihi_jifun', ascending=False)
                     df = df.drop_duplicates(subset=['race_id'], keep='first')
                
                # Parse
                for _, row in df.iterrows():
                    rid = row['race_id']
                    raw = row[col_name]
                    if not raw: continue
                    
                    if rid not in results: results[rid] = {}
                    if ticket_type not in results[rid]: results[rid][ticket_type] = {}
                    
                    # Parsing
                    # Iterate chunks
                    total_len = len(raw)
                    count = total_len // byte_len
                    
                    for i in range(count):
                        offset = i * byte_len
                        chunk = raw[offset : offset+byte_len]
                        
                        try:
                            # Logic varies by type
                            if ticket_type in ['umaren', 'umatan']:
                                # H1(2)+H2(2)+Odds(6)+Pop(3) = 13
                                h1 = chunk[0:2]
                                h2 = chunk[2:4]
                                odds_str = chunk[4:10]
                                # pop = chunk[10:13]
                                
                                if h1 == '00' or h2 == '00': continue
                                if odds_str == '000000': continue
                                
                                odds = int(odds_str) / 10.0 # Typically 123 -> 12.3 (Wait, 6 digits? '001230'?)
                                # JRA-VAN O2 Odds is 6 bytes. 
                                # Spec usually 001230 = 123.0? Or 123.0?
                                # Let's assume standard 10x implied for now or check data magnitude.
                                # Sample O2: 011487 -> 1148.7? or 114.8?
                                # Usually Umaren ranges 5.0 to 1000.0.
                                # If 11487 -> 114.8 is reasonable. /100?
                                # Let's check O1 logic. O1 (4 digits) '0037' -> 3.7 (/10).
                                # O2 (6 digits) '011487'. If /10 -> 1148.7. If /100 -> 114.87?
                                # JRA odds are usually 0.1 step.
                                # '011487' -> 11487? No.
                                # Maybe /10 is correct. 1148.7 is possible for Umaren.
                                
                                key = f"{int(h1):02}{int(h2):02}"
                                results[rid][ticket_type][key] = odds

                            elif ticket_type == 'wide':
                                # H1(2)+H2(2)+Min(5)+Max(5)+Pop(3) = 17
                                h1 = chunk[0:2]
                                h2 = chunk[2:4]
                                min_s = chunk[4:9]
                                max_s = chunk[9:14]
                                
                                if h1 == '00' or h2 == '00': continue
                                if min_s == '00000': continue
                                
                                o_min = int(min_s) / 10.0
                                o_max = int(max_s) / 10.0
                                
                                key = f"{int(h1):02}{int(h2):02}"
                                results[rid][ticket_type][key] = (o_min, o_max)

                            elif ticket_type == 'sanrenpuku':
                                # H1(2)+H2(2)+H3(2)+Odds(6)+Pop(3) = 15
                                h1 = chunk[0:2]
                                h2 = chunk[2:4]
                                h3 = chunk[4:6]
                                odds_str = chunk[6:12]
                                
                                if h1 == '00': continue
                                if odds_str == '000000': continue
                                
                                odds = int(odds_str) / 10.0
                                key = f"{int(h1):02}{int(h2):02}{int(h3):02}"
                                results[rid][ticket_type][key] = odds

                            elif ticket_type == 'sanrentan':
                                # H1(2)+H2(2)+H3(2)+Odds(8)+Pop(3) = 17
                                h1 = chunk[0:2]
                                h2 = chunk[2:4]
                                h3 = chunk[4:6]
                                odds_str = chunk[6:14]
                                
                                if h1 == '00': continue
                                if odds_str == '00000000': continue
                                
                                odds = int(odds_str) / 10.0
                                key = f"{int(h1):02}{int(h2):02}{int(h3):02}"
                                results[rid][ticket_type][key] = odds
                                
                        except ValueError:
                            continue

            except Exception as e:
                logger.warning(f"Complex Odds Load Error ({tbl}): {e}")
                
        return results




    def get_race_schedule(self, start_date: str = None, end_date: str = None, limit: int = 20) -> pd.DataFrame:
        """
        開催スケジュールのサマリを取得します。
        デフォルトでは未来の開催予定または直近の開催を取得します。
        
        Args:
            start_date (str): 'YYYY-MM-DD' (Optional)
            end_date (str): 'YYYY-MM-DD' (Optional)
            limit (int): 取得する日数制限 (Descending order of date)

        Returns:
            pd.DataFrame: columns=['date', 'venue', 'races_count', 'main_race_name']
        """
        tbl_race = self._get_table_name(['jvd_race_shosai', 'race_shosai', 'jvd_ra'])
        is_pckeiba_short = (tbl_race == 'jvd_ra' or tbl_race == 'ra')
        col_title = "r.kyosomei_hondai" if is_pckeiba_short else "r.race_mei_honbun"

        # Construct Date Filter
        date_cond = "1=1"
        if start_date:
            s_flat = start_date.replace('-', '')
            date_cond += f" AND (r.kaisai_nen || r.kaisai_tsukihi) >= '{s_flat}'"
        
        if end_date:
            e_flat = end_date.replace('-', '')
            date_cond += f" AND (r.kaisai_nen || r.kaisai_tsukihi) <= '{e_flat}'"
            
        query = f"""
        WITH daily_venues AS (
            SELECT
                TO_DATE(r.kaisai_nen || r.kaisai_tsukihi, 'YYYYMMDD') AS date,
                r.keibajo_code AS venue_code,
                COUNT(r.race_bango) AS races_count,
                -- Main Race (usually 11R) Title
                MAX(CASE WHEN r.race_bango = '11' THEN {col_title} ELSE NULL END) AS main_race_name
            FROM {tbl_race} r
            WHERE {date_cond}
            GROUP BY date, venue_code
        )
        SELECT * FROM daily_venues
        ORDER BY date DESC, venue_code ASC
        LIMIT {limit * 3} 
        """
        # Limit * 3 because multiple venues per day
        
        try:
            df = pd.read_sql(query, self.engine)
            
            # Map codes
            venue_map = {
                '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
                '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
            }
            df['venue'] = df['venue_code'].map(venue_map).fillna(df['venue_code'])
            
            return df
        except Exception as e:
            logger.error(f"Schedule Load Error: {e}")
            return pd.DataFrame()


import numpy as np

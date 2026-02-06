
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import ProgrammingError
from datetime import datetime, timedelta
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
        return pd.DataFrame()

class JraVanDataLoader:
    """
    PC-KEIBA Database (JRA-VAN) のデータをロードするクラス。
    """
    def __init__(self):
        user = os.environ.get('POSTGRES_USER', 'postgres')
        password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'postgres')
        connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        masked_pwd = '*' * 5
        logger.info(f"Connecting to DB: user={user} host={host} port={port} dbname={dbname} pwd={masked_pwd}")
        
        import time
        for i in range(5):
            try:
                self.engine = create_engine(connection_str)
                with self.engine.connect() as conn:
                    pass
                logger.info("  DB Connection successful.")
                break
            except Exception as e:
                logger.warning(f"  DB Connection attempt {i+1} failed: {e}")
                time.sleep(2)
        else:
             self.engine = create_engine(connection_str) # Create anyway to let it fail later if needed

    def _get_table_name(self, candidates: list[str]) -> str:
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

    def _merge_odds_10min(self, df: pd.DataFrame, history_start_date: str, target_date: str = None) -> pd.DataFrame:
        """
        [v08] 時系列オッズ (10分前) を apd_sokuho_o1/o2 から取得してマージする
        Args:
            target_date: 'YYYY-MM-DD'. If provided, ONLY load odds for races on this date (JIT optimization).
        """
        if 'race_id' not in df.columns: return df
        
        # o1: Win/Place, o2: Umaren
        has_sokuho = self._get_table_name(['apd_sokuho_o1']) == 'apd_sokuho_o1'
        if not has_sokuho: return df
        
        # [Optimization] 
        # If target_date is specified, we ONLY care about odds for that day (for prediction).
        # We ignore history rows for odds fetching.
        if target_date:
            # Filter unique_races to only those on target_date
            # Use race_id string matching (YYYYMMDD start) to be robust against Date types
            target_date_compact = target_date.replace('-', '')
            target_races = df[df['race_id'].astype(str).str.startswith(target_date_compact)]['race_id'].unique()
            logger.info(f"Odds Fetch: Targeted Optimization for {target_date} (Races: {len(target_races)})")
            
            # CRITICAL: If target_date is set but no races found, DO NOT fallback to full scan.
            # Just return empty result or skip.
            if len(target_races) == 0:
                logger.warning(f"No target races found for date {target_date}. Skipping odds fetch to avoid full scan.")
                return df
        else:
            target_races = df['race_id'].unique()

        # Decide JIT Mode based on filtered race count
        is_jit_mode = len(target_races) < 100
        
        HISTORY_ODDS_LIMIT_YEAR = "2024"
        load_year = max(history_start_date.split('-')[0] if history_start_date else "1900", HISTORY_ODDS_LIMIT_YEAR)
        
        logger.info(f"時系列オッズ(10分前)を取得中 (Win/Place/Umaren)... JIT Mode={is_jit_mode}")
        
        # --- 共通処理: Race Time & Deadline ---
        race_times = df[['race_id', 'date', 'start_time_str']].drop_duplicates().copy()
        
        def parse_hhmm(s):
            if pd.isna(s) or len(str(s)) < 3: return pd.Timedelta(0)
            s = str(s).zfill(4)
            return pd.Timedelta(hours=int(s[:2]), minutes=int(s[2:]))
            
        race_times['start_delta'] = race_times['start_time_str'].apply(parse_hhmm)
        race_times['date'] = pd.to_datetime(race_times['date'])
        race_times['start_time'] = race_times['date'] + race_times['start_delta']
        race_times['deadline'] = race_times['start_time'] - pd.Timedelta(minutes=10)
        
        # Optimied WHERE clause builder
        def build_odds_where(target_races_list):
            if is_jit_mode and len(target_races_list) > 0:
                # Construct (nen, keibajo, kai, nichi, bango) IN (...)
                conditions = []
                for rid in target_races_list:
                    if len(rid) != 12: continue
                    nen = rid[:4]
                    kbj = rid[4:6]
                    kai = rid[6:8] # str
                    nich = rid[8:10] # str
                    ban = rid[10:12] # str
                    conditions.append(f"('{nen}', '{kbj}', '{kai}', '{nich}', '{ban}')")
                
                if not conditions: 
                    # Should be unreachable if len > 0 checks logic
                    return f" WHERE kaisai_nen >= '{load_year}'" 
                return f" WHERE (kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) IN ({', '.join(conditions)})"
            else:
                return f" WHERE kaisai_nen >= '{load_year}'"

        where_clause = build_odds_where(target_races)

        # --- 1. Win & Place (apd_sokuho_o1) ---
        query_o1 = "SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, happyo_tsukihi_jifun, odds_tansho, odds_fukusho FROM apd_sokuho_o1" + where_clause
            
        try:
            odds_o1 = pd.read_sql(query_o1, self.engine)
            if not odds_o1.empty:
                for c in ['kaisai_nen', 'keibajo_code', 'kaisai_kai', 'kaisai_nichime', 'race_bango']:
                    odds_o1[c] = odds_o1[c].astype(str)
                
                odds_o1['race_id'] = odds_o1['kaisai_nen'] + odds_o1['keibajo_code'] + odds_o1['kaisai_kai'] + odds_o1['kaisai_nichime'] + odds_o1['race_bango']
                odds_o1['timestamp_str'] = odds_o1['kaisai_nen'] + odds_o1['happyo_tsukihi_jifun']
                odds_o1['timestamp'] = pd.to_datetime(odds_o1['timestamp_str'], format='%Y%m%d%H%M', errors='coerce')
                
                merged_o1 = pd.merge(odds_o1, race_times[['race_id', 'deadline']], on='race_id', how='inner')
                valid_o1 = merged_o1[merged_o1['timestamp'] <= merged_o1['deadline']].copy()
                
                if not valid_o1.empty:
                    valid_o1 = valid_o1.sort_values(['race_id', 'timestamp'], ascending=[True, False])
                    latest_o1 = valid_o1.drop_duplicates(subset=['race_id'], keep='first')
                    
                    latest_o1 = latest_o1[['race_id', 'odds_tansho', 'odds_fukusho']].rename(
                        columns={'odds_tansho': 'odds_win_str', 'odds_fukusho': 'odds_place_str'}
                    )
                    df = pd.merge(df, latest_o1, on='race_id', how='left')
                    df['odds_10min_str'] = df['odds_win_str']
                    
        except Exception as e:
            logger.warning(f"Failed to load o1 (Win/Place): {e}")

        # --- 2. Umaren (apd_sokuho_o2) ---
        query_o2 = "SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, happyo_tsukihi_jifun, odds_umaren FROM apd_sokuho_o2" + where_clause
             
        try:
            odds_o2 = pd.read_sql(query_o2, self.engine)
            if not odds_o2.empty:
                odds_o2['race_id'] = odds_o2['kaisai_nen'] + odds_o2['keibajo_code'] + odds_o2['kaisai_kai'] + odds_o2['kaisai_nichime'] + odds_o2['race_bango']
                odds_o2['timestamp_str'] = odds_o2['kaisai_nen'] + odds_o2['happyo_tsukihi_jifun']
                odds_o2['timestamp'] = pd.to_datetime(odds_o2['timestamp_str'], format='%Y%m%d%H%M', errors='coerce')
                
                merged_o2 = pd.merge(odds_o2, race_times[['race_id', 'deadline']], on='race_id', how='inner')
                valid_o2 = merged_o2[merged_o2['timestamp'] <= merged_o2['deadline']].copy()
                
                if not valid_o2.empty:
                    valid_o2 = valid_o2.sort_values(['race_id', 'timestamp'], ascending=[True, False])
                    latest_o2 = valid_o2.drop_duplicates(subset=['race_id'], keep='first')
                    
                    latest_o2 = latest_o2[['race_id', 'odds_umaren']].rename(
                        columns={'odds_umaren': 'odds_umaren_str'}
                    )
                    df = pd.merge(df, latest_o2, on='race_id', how='left')
                    
        except Exception as e:
            logger.warning(f"Failed to load o2 (Umaren): {e}")
            
        return df

    def load_for_horses(self, target_horse_ids: list[str], target_date: str, history_start_date: str = "2016-01-01", skip_training: bool = False) -> pd.DataFrame:
        """
        [Optimization] 特定の馬リストに対応する過去走データ ＋ 指定日の全レースデータ を取得する
        JIT予測用: データ量を削減しつつ、必要な馬の全過去走と、当日のレースコンテキストを維持する。
        """
        # 1. Base Loader Logic Reuse?
        # Recreating query is safer for complex OR filtering.
        
        # Params
        jra_only = False
        
        print("DEBUG: Constructing optimized JIT query...", flush=True)
        self.tables = ['jvd_ra', 'jvd_se', 'jvd_hr', 'jvd_um', 'jvd_ks', 'jvd_ch', 'jvd_bt', 'jvd_hc']
        tbl_race = self._get_table_name(['jvd_race_shosai', 'race_shosai', 'jvd_ra'])
        tbl_seiseki = self._get_table_name(['jvd_seiseki', 'seiseki', 'jvd_se'])
        tbl_uma = self._get_table_name(['jvd_uma_master', 'uma_master', 'jvd_um'])
        is_pckeiba_short = (tbl_race == 'jvd_ra' or tbl_race == 'ra')
        col_pass = ["res.corner_1", "res.corner_2", "res.corner_3", "res.corner_4"]

        # Columns (Copy from load - abbreviated_
        if is_pckeiba_short:
            col_title = "r.race_mei_honbun" # Typo in original? No, r.kyosomei_hondai
            if tbl_race == 'jvd_ra': col_title = "r.kyosomei_hondai" 
            col_state = "CASE WHEN CAST(r.track_code AS INTEGER) BETWEEN 10 AND 22 THEN r.babajotai_code_shiba ELSE r.babajotai_code_dirt END"
            col_rank = "TRIM(res.kakutei_chakujun)"
            col_last3f = "res.kohan_3f"
            col_pop = "res.tansho_ninkijun"
            col_horse_name = "res.bamei"
            col_sex = "res.seibetsu_code"
            col_sire = "uma.ketto_joho_01a"
            col_mare = "uma.ketto_joho_02a"
            col_bms = "uma.ketto_joho_03a"
            col_honshokin = "res.kakutoku_honshokin"
            col_fukashokin = "res.kakutoku_fukashokin"
            col_time_diff = "res.time_sa" 
            col_blinker = "res.blinker_shiyo_kubun"
            col_mining = "res.mining_kubun"
            col_yoso_time = "res.yoso_soha_time"
            col_yoso_diff_p = "res.yoso_gosa_plus"
            col_yoso_diff_m = "res.yoso_gosa_minus"
            col_yoso_rank = "res.yoso_juni"
        else:
            # Fallback Standard
            col_title = "r.race_mei_honbun"
            col_state = "r.baba_jotai_code"
            col_rank = "TRIM(res.kakutei_chakusun)"
            col_last3f = "res.agari_3f"
            col_pop = "res.ninki"
            col_horse_name = "uma.bamei"
            col_sex = "uma.seibetsu_code"
            col_sire = "uma.fushu_ketto_toroku_bango"
            col_mare = "uma.boshu_ketto_toroku_bango"
            col_bms = "uma.hahachichi_ketto_toroku_bango"
            col_honshokin = "res.honshokin"
            col_fukashokin = "res.fukashokin"
            col_time_diff = "res.time_sa"
            col_blinker = "res.blinker_shiyo_kubun"
            col_mining = "res.mining_kubun"
            col_yoso_time = "res.yoso_soha_time"
            col_yoso_diff_p = "res.yoso_gosa_plus"
            col_yoso_diff_m = "res.yoso_gosa_minus"
            col_yoso_rank = "res.yoso_juni"

        # Construct WHERE Logic
        # (History for Horses) OR (Target Date Context)
        # Ensure IDs are stripped and safe
        safe_ids = [str(hid).strip() for hid in target_horse_ids] if target_horse_ids else []
        ids_str = ", ".join([f"'{hid}'" for hid in safe_ids]) if safe_ids else "''"
        
        # Debug: Check ID format
        if safe_ids:
            logger.info(f"DEBUG: First 3 Horse IDs: {safe_ids[:3]}")
        
        # Note: res.ketto_toroku_bango is often CHAR(10) padded. 
        # Cast to text AND TRIM to ensure matching with python strings (which are stripped)
        
        # [Fix] Broaden JIT history to include context:
        # 1. Target horses' history + contextual Top 3
        # 2. Target day's members' trainer/jockey activity for last 60 days
        target_date_compact = target_date.replace('-', '')
        history_history_date_filter = f"CONCAT(r.kaisai_nen, r.kaisai_tsukihi) < '{target_date_compact}' AND CONCAT(r.kaisai_nen, r.kaisai_tsukihi) >= '{history_start_date.replace('-','')}'"
        
        # 60 days filter for stable_form
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        start_60d = (target_dt - timedelta(days=60)).strftime('%Y%m%d')
        entity_date_filter = f"CONCAT(r.kaisai_nen, r.kaisai_tsukihi) < '{target_date_compact}' AND CONCAT(r.kaisai_nen, r.kaisai_tsukihi) >= '{start_60d}'"

        history_race_subquery = f"(SELECT DISTINCT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) FROM {tbl_seiseki} WHERE TRIM(ketto_toroku_bango::text) IN ({ids_str}))"
        
        where_str = f"""WHERE 
            (CONCAT(r.kaisai_nen, r.kaisai_tsukihi) = '{target_date_compact}')
            OR (
                {history_history_date_filter}
                AND CONCAT(r.kaisai_nen, r.keibajo_code, r.kaisai_kai, r.kaisai_nichime, r.race_bango) IN {history_race_subquery}
                AND (
                    TRIM(res.ketto_toroku_bango::text) IN ({ids_str})
                    OR res.kakutei_chakujun::integer <= 3
                )
            )
            OR (
                {entity_date_filter}
                AND (
                    res.kishu_code IN (SELECT DISTINCT kishu_code FROM {tbl_seiseki} WHERE CONCAT(kaisai_nen, kaisai_tsukihi) = '{target_date_compact}')
                    OR res.chokyoshi_code IN (SELECT DISTINCT chokyoshi_code FROM {tbl_seiseki} WHERE CONCAT(kaisai_nen, kaisai_tsukihi) = '{target_date_compact}')
                )
            )"""
        
        logger.info(f"DEBUG: Optimized Context-Aware JIT query constructed.")
        # logger.info(f"DEBUG: Full WHERE: {where_str}") # Too long
        
        # Verify columns mapping in Select
        query = f"""
        SELECT
            r.shusso_tosu::integer AS field_size,
            CONCAT(r.kaisai_nen, r.keibajo_code, r.kaisai_kai, r.kaisai_nichime, r.race_bango) AS race_id,
            TO_DATE(r.kaisai_nen || r.kaisai_tsukihi, 'YYYYMMDD') AS date,
            r.hasso_jikoku AS start_time_str,
            r.keibajo_code AS venue,
            r.race_bango::integer AS race_number,
            r.kyori::integer AS distance,
            r.track_code AS surface,
            r.tenko_code AS weather,
            r.tenko_code AS weather_code,
            {col_state} AS state,
            {col_state} AS going_code,
            {col_title} AS title,
            r.grade_code AS grade_code,
            r.kyoso_shubetsu_code AS kyoso_shubetsu_code,
            r.kyoso_joken_code AS kyoso_joken_code,
            res.ketto_toroku_bango AS horse_id,
            {col_sire} AS sire_id,
            {col_mare} AS mare_id,
            {col_bms} AS bms_id,
            {col_sex} AS sex,
            res.kishu_code AS jockey_id,
            res.chokyoshi_code AS trainer_id,
            res.wakuban::integer AS frame_number,
            res.umaban::integer AS horse_number,
            {col_rank} AS rank_str,
            res.soha_time AS raw_time,
            {col_time_diff} AS time_diff,
            {col_last3f} AS last_3f,
            r.zenhan_3f AS first_3f,
            res.tansho_odds AS odds,
            {col_pop} AS popularity,
            res.bataiju AS weight,
            res.zogen_sa AS weight_diff_val,
            res.zogen_fugo AS weight_diff_sign,
            res.ijo_kubun_code AS abnormal_code,
            res.barei AS age,
            res.kyakushitsu_hantei AS running_style,
            {col_horse_name} AS horse_name,
            res.futan_juryo AS impost,
            {col_honshokin} AS honshokin,
            {col_fukashokin} AS fukashokin,
            {col_mining} AS mining_kubun,
            {col_yoso_time} AS yoso_soha_time,
            {col_yoso_diff_p} AS yoso_gosa_plus,
            {col_yoso_diff_m} AS yoso_gosa_minus,
            {col_yoso_rank} AS yoso_juni,
            {col_blinker} AS blinker,
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
        {where_str}
        ORDER BY date, race_id
        """

        # Reuse processing logic
        logger.info("JRA-VANデータをロード中 (JIT Optimized)...")
        print("DEBUG: Executing pd.read_sql with chunksize...", flush=True)
        try:
            chunks = []
            chunk_count = 0
            for chunk in pd.read_sql(query, self.engine, chunksize=10000):
                chunks.append(chunk)
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"DEBUG: Loaded {chunk_count} chunks...", flush=True)
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.DataFrame()
                
            print(f"DEBUG: Read SQL done. Rows: {len(df)}", flush=True)
            logger.info(f"ロード件数 (重複削除前): {len(df)} 件")

            # Post-Processing Reuse
            before_len = len(df)
            df.drop_duplicates(subset=['race_id', 'horse_number'], inplace=True)
            if len(df) != before_len:
                logger.warning(f"重複データを削除しました: {before_len} -> {len(df)} 件")

            if len(df) == 0:
                logger.warning("注意: 取得データが0件です。")
                return df

            # 10分前オッズ
            df = self._merge_odds_10min(df, history_start_date, target_date=target_date)

            # Training Data
            if not skip_training:
                # Merge logic re-uses self._merge_training_data
                # But we can optimize it too by passing horse_ids
                df = self._merge_training_data(df, history_start_date, target_horse_ids)
            else:
                logger.info("  Skip loading training data (jvd_hc).")

            # --- Python側での前処理 (Same as load) ---
            df['rank'] = pd.to_numeric(df['rank_str'], errors='coerce')
            df['odds'] = pd.to_numeric(df['odds'], errors='coerce') / 10.0
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

            if 'odds_win_str' in df.columns:
                df['odds_10min'] = pd.to_numeric(df['odds_win_str'], errors='coerce') / 10.0
            
            if 'odds_10min' not in df.columns:
                df['odds_10min'] = df['odds']
            else:
                df['odds_10min'] = df['odds_10min'].fillna(df['odds'])
            
            df['first_3f'] = pd.to_numeric(df['first_3f'], errors='coerce') / 10.0
            # [Fix] last_3f is stored as string (e.g. '344' = 34.4s) - convert to seconds
            df['last_3f'] = pd.to_numeric(df['last_3f'], errors='coerce') / 10.0
            df['time_diff'] = pd.to_numeric(df['time_diff'], errors='coerce') / 10.0
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df['impost'] = pd.to_numeric(df['impost'], errors='coerce') / 10.0
            df['weight_ratio'] = df['impost'] / df['weight'].replace(0, np.nan)
            df['honshokin'] = pd.to_numeric(df.get('honshokin', 0), errors='coerce').fillna(0)
            df['fukashokin'] = pd.to_numeric(df.get('fukashokin', 0), errors='coerce').fillna(0)
            df['blinker'] = pd.to_numeric(df.get('blinker', 0), errors='coerce').fillna(0).astype(int)

            def convert_time(t):
                if pd.isna(t): return None
                try:
                    t_str = str(int(t)).zfill(4)
                    return int(t_str[:-3]) * 60 + int(t_str[-3:-1]) + int(t_str[-1]) * 0.1
                except: return None
            df['time'] = df['raw_time'].apply(convert_time)

            # [Fix] Scale 3F times only if they are stored as integers (e.g. 352 for 35.2s)
            # If they are already float (35.2), don't scale.
            for c in ['last_3f', 'first_3f']:
                if c in df.columns:
                    vals = pd.to_numeric(df[c], errors='coerce')
                    # If any value is > 100, it's likely 10x scaled.
                    if vals.max() > 100:
                        df[c] = vals / 10.0
                    else:
                        df[c] = vals

            def convert_weight_diff(row):
                try:
                    val = int(row['weight_diff_val'])
                    return -val if row['weight_diff_sign'] == '-' else val
                except: return 0
            df['weight_diff'] = df.apply(convert_weight_diff, axis=1)

            def make_passing_rank(row):
                vals = []
                # Use aliased column names from SQL
                pass_cols = ['pass_1', 'pass_2', 'pass_3', 'pass_4']
                for c in pass_cols:
                    v = row.get(c)
                    if pd.notnull(v) and v not in [0, '0', '00']: vals.append(str(v).replace('.0', ''))
                return "-".join(vals) if vals else None
            df['passing_rank'] = df.apply(make_passing_rank, axis=1)

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
                except: return 'Unknown'
            df['surface'] = df['surface'].apply(map_surface)

            state_map = {1: '良', 2: '稍重', 3: '重', 4: '不良'}
            df['state'] = pd.to_numeric(df['state'], errors='coerce').map(state_map).fillna('Unknown')
            
            # Raw codes (Keep numeric)
            df['weather_code'] = pd.to_numeric(df['weather_code'], errors='coerce').fillna(0).astype(int)
            df['going_code'] = pd.to_numeric(df['going_code'], errors='coerce').fillna(0).astype(int)
            
            # [Fix] Add corner column aliases for corner_dynamics feature block
            # pass_1/2/3/4 -> corner_1/2/3/4
            for i in range(1, 5):
                if f'pass_{i}' in df.columns:
                    df[f'corner_{i}'] = df[f'pass_{i}']
            
            # [Fix] Add heads_count (number of horses in race) for corner position normalization
            df['heads_count'] = df.groupby('race_id')['horse_number'].transform('count')

            # [Fix] Ensure horse_id is clean for grouping
            if 'horse_id' in df.columns:
                df['horse_id'] = df['horse_id'].astype(str).str.strip()
            
            # [Fix] Ensure date is datetime for correct sorting
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            if 'date' in df.columns and len(df) > 0:
                logger.info(f"データ読み込み期間: {df['date'].min()} ~ {df['date'].max()}")
                logger.info(f"レース数: {df['race_id'].nunique():,}件, レコード数: {len(df):,}件")
                
                # Debug Grouping
                try:
                    grp_counts = df.groupby('horse_id').size()
                    logger.info(f"DEBUG: Rows per horse stats:\n{grp_counts.describe()}")
                except:
                    logger.info("DEBUG: Groupby check failed.")

            logger.info(f"前処理完了: {len(df)} 件")
            return df

        except ProgrammingError as e:
            logger.error(f"データロードエラー: {e}")
            raise e
        except Exception as e:
            logger.error(f"データロードエラー: {e}")
            raise e

    def _merge_training_data(self, df: pd.DataFrame, history_start_date: str, horse_ids: list[str] = None) -> pd.DataFrame:
        """
        [v09] 調教データ (jvd_hc:坂路, jvd_wc:コース) を取得してマージする (Python側結合)
        レース当日に最も近い直近の調教記録を採用する。
        - jvd_hc: training_course_cat=1 (Hanro)
        - jvd_wc: training_course_cat=course (Wood/Poly/etc, e.g. 2,3)
        """
        if 'horse_id' not in df.columns: return df
        
        has_hc = self._get_table_name(['jvd_hc']) == 'jvd_hc'
        has_wc = self._get_table_name(['jvd_wc']) == 'jvd_wc'
        
        if not has_hc and not has_wc: return df
        
        logger.info("調教データ(jvd_hc, jvd_wc)を取得中...")
        
        # WHERE Clause Helper
        def build_where(alias=''):
            clauses = []
            if history_start_date:
                y = history_start_date.split('-')[0]
                clauses.append(f"SUBSTRING(chokyo_nengappi, 1, 4) >= '{y}'")
            if horse_ids:
                ids_str = ", ".join([f"'{hid}'" for hid in horse_ids])
                clauses.append(f"ketto_toroku_bango IN ({ids_str})")
            return " WHERE " + " AND ".join(clauses) if clauses else ""

        where_sql = build_where()
        dfs = []

        # --- 1. jvd_hc (Hanro) ---
        if has_hc:
            query_hc = f"""
            SELECT 
                ketto_toroku_bango as horse_id, 
                chokyo_nengappi as chokyo_date,
                1 as training_course_cat, 
                time_gokei_4f as training_time_4f,
                time_gokei_3f as training_time_3f,
                lap_time_1f as training_time_last1f
            FROM jvd_hc
            {where_sql}
            """
            try:
                hc_df = pd.read_sql(query_hc, self.engine)
                if not hc_df.empty:
                    dfs.append(hc_df)
                    logger.info(f"  Loaded {len(hc_df)} records from jvd_hc.")
            except Exception as e:
                 logger.warning(f"Failed to load jvd_hc: {e}")

        # --- 2. jvd_wc (Wood/Course) ---
        if has_wc:
            query_wc = f"""
            SELECT 
                ketto_toroku_bango as horse_id, 
                chokyo_nengappi as chokyo_date,
                course as training_course_cat, 
                time_gokei_4f as training_time_4f,
                time_gokei_3f as training_time_3f,
                lap_time_1f as training_time_last1f
            FROM jvd_wc
            {where_sql}
            """
            try:
                wc_df = pd.read_sql(query_wc, self.engine)
                if not wc_df.empty:
                    dfs.append(wc_df)
                    logger.info(f"  Loaded {len(wc_df)} records from jvd_wc.")
            except Exception as e:
                 logger.warning(f"Failed to load jvd_wc: {e}")
        
        if not dfs: return df
        
        # Concat
        full_train_df = pd.concat(dfs, ignore_index=True)
        
        # Merge Processing
        try:
            full_train_df['chokyo_dt'] = pd.to_datetime(full_train_df['chokyo_date'], format='%Y%m%d', errors='coerce')
            full_train_df = full_train_df.dropna(subset=['chokyo_dt'])
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Prioritize Hanro (1) over Wood (2) if same day -> Sort by date asc, course_type desc -> last is merged (wait, merge_asof direction='backward' matches closest <= date)
            # If multiple rows have same date (closest to race), merge_asof behaviour: "If exact match, take last row".
            # So if we want Hanro priority on same day:
            # Sort: date ASC, course_type ASC (1, 2). Last is 2 (Wood). Wood picked.
            # Sort: date ASC, course_type DESC (2, 1). Last is 1 (Hanro). Hanro picked.
            # Thus: ascending=[True, False]
            full_train_df = full_train_df.sort_values(['chokyo_dt', 'training_course_cat'], ascending=[True, False]).reset_index(drop=True)
            
            logger.info(f"  Performing merge_asof... (df: {len(df)}, train: {len(full_train_df)})")
            
            merged = pd.merge_asof(
                df, full_train_df, 
                left_on='date', right_on='chokyo_dt', 
                by='horse_id', direction='backward'
            )
            
            mask = (merged['date'] - merged['chokyo_dt']).dt.days > 10
            cols_tc = ['training_time_4f', 'training_time_3f', 'training_time_last1f', 'training_course_cat']
            merged.loc[mask, cols_tc] = np.nan
            
            for col in ['training_time_4f', 'training_time_3f', 'training_time_last1f']:
                merged[col] = pd.to_numeric(merged[col], errors='coerce') / 10.0
            
            logger.info(f"  Training data merged. (Available: {merged['training_course_cat'].notnull().sum()} records)")
            return merged

        except Exception as e:
             logger.warning(f"Failed to merge training data: {e}")
             import traceback
             logger.warning(traceback.format_exc())
             return df

    def load(self, limit: int = None, jra_only: bool = False, 
             history_start_date: str = "2014-01-01", end_date: str = None,
             skip_odds: bool = False, skip_training: bool = False,
             horse_ids: list[str] = None) -> pd.DataFrame:
        if skip_training:
            logger.info("  Skip loading training data (jvd_hc).")
            print("DEBUG: skip_training=True selected.", flush=True)

        print("DEBUG: Constructing query...", flush=True)
        self.tables = ['jvd_ra', 'jvd_se', 'jvd_hr', 'jvd_um', 'jvd_ks', 'jvd_ch', 'jvd_bt', 'jvd_hc']
        tbl_race = self._get_table_name(['jvd_race_shosai', 'race_shosai', 'jvd_ra'])
        tbl_seiseki = self._get_table_name(['jvd_seiseki', 'seiseki', 'jvd_se'])
        tbl_uma = self._get_table_name(['jvd_uma_master', 'uma_master', 'jvd_um'])
        is_pckeiba_short = (tbl_race == 'jvd_ra' or tbl_race == 'ra')
        col_pass = ["res.corner_1", "res.corner_2", "res.corner_3", "res.corner_4"]

        if is_pckeiba_short:
            logger.info("PC-KEIBA短縮名スキーマ (jvd_ra) を検出しました。")
            col_title = "r.kyosomei_hondai"
            col_state = "CASE WHEN CAST(r.track_code AS INTEGER) BETWEEN 10 AND 22 THEN r.babajotai_code_shiba ELSE r.babajotai_code_dirt END"
            col_rank = "TRIM(res.kakutei_chakujun)"
            col_last3f = "res.kohan_3f"
            col_pop = "res.tansho_ninkijun"
            col_horse_name = "res.bamei"
            col_sex = "res.seibetsu_code"
            col_sire = "uma.ketto_joho_01a"
            col_mare = "uma.ketto_joho_02a"
            col_bms = "uma.ketto_joho_03a"
            col_honshokin = "res.kakutoku_honshokin"
            col_fukashokin = "res.kakutoku_fukashokin"
            col_time_diff = "res.time_sa" 
            col_blinker = "res.blinker_shiyo_kubun"
            col_mining = "res.mining_kubun"
            col_yoso_time = "res.yoso_soha_time"
            col_yoso_diff_p = "res.yoso_gosa_plus"
            col_yoso_diff_m = "res.yoso_gosa_minus"
            col_yoso_rank = "res.yoso_juni"
        else:
            logger.info("標準スキーマ (jvd_race_shosai) を使用します。")
            col_title = "r.race_mei_honbun"
            col_state = "r.baba_jotai_code"
            col_rank = "TRIM(res.kakutei_chakusun)"
            col_last3f = "res.agari_3f"
            col_pop = "res.ninki"
            col_horse_name = "uma.bamei"
            col_sex = "uma.seibetsu_code"
            col_sire = "uma.fushu_ketto_toroku_bango"
            col_mare = "uma.boshu_ketto_toroku_bango"
            col_bms = "uma.hahachichi_ketto_toroku_bango" # Assuming standard column name, needs verification if table differs
            col_honshokin = "res.honshokin"
            col_fukashokin = "res.fukashokin"
            col_time_diff = "res.time_sa"
            col_blinker = "res.blinker_shiyo_kubun"
            col_mining = "res.mining_kubun"
            col_yoso_time = "res.yoso_soha_time"
            col_yoso_diff_p = "res.yoso_gosa_plus"
            col_yoso_diff_m = "res.yoso_gosa_minus"
            col_yoso_rank = "res.yoso_juni"

        where_clauses = []
        if horse_ids:
            ids_str = ", ".join([f"'{hid}'" for hid in horse_ids])
            where_clauses.append(f"res.ketto_toroku_bango IN ({ids_str})")
        
        if history_start_date:
            date_filter = history_start_date.replace('-', '')
            where_clauses.append(f"CONCAT(r.kaisai_nen, r.kaisai_tsukihi) >= '{date_filter}'")
        if end_date:
            end_filter = end_date.replace('-', '')
            where_clauses.append(f"CONCAT(r.kaisai_nen, r.kaisai_tsukihi) <= '{end_filter}'")
        if jra_only:
            where_clauses.append("r.keibajo_code BETWEEN '01' AND '10'")
        where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        SELECT
            CONCAT(r.kaisai_nen, r.keibajo_code, r.kaisai_kai, r.kaisai_nichime, r.race_bango) AS race_id,
            TO_DATE(r.kaisai_nen || r.kaisai_tsukihi, 'YYYYMMDD') AS date,
            r.hasso_jikoku AS start_time_str,
            r.keibajo_code AS venue,
            r.race_bango::integer AS race_number,
            r.kyori::integer AS distance,
            r.track_code AS surface,
            r.tenko_code AS weather,
            r.tenko_code AS weather_code,
            {col_state} AS state,
            {col_state} AS going_code,
            {col_title} AS title,
            r.grade_code AS grade_code,
            r.kyoso_shubetsu_code AS kyoso_shubetsu_code,
            r.kyoso_joken_code AS kyoso_joken_code,
            res.ketto_toroku_bango AS horse_id,
            {col_sire} AS sire_id,
            {col_mare} AS mare_id,
            {col_bms} AS bms_id,
            {col_sex} AS sex,
            res.kishu_code AS jockey_id,
            res.chokyoshi_code AS trainer_id,
            res.wakuban::integer AS frame_number,
            res.umaban::integer AS horse_number,
            {col_rank} AS rank_str,
            res.soha_time AS raw_time,
            {col_time_diff} AS time_diff,
            {col_last3f} AS last_3f,
            r.zenhan_3f AS first_3f,
            res.tansho_odds AS odds,
            {col_pop} AS popularity,
            res.bataiju AS weight,
            res.zogen_sa AS weight_diff_val,
            res.zogen_fugo AS weight_diff_sign,
            res.ijo_kubun_code AS abnormal_code,
            res.barei AS age,
            {col_horse_name} AS horse_name,
            res.futan_juryo AS impost,
            {col_honshokin} AS honshokin,
            {col_fukashokin} AS fukashokin,
            {col_mining} AS mining_kubun,
            {col_yoso_time} AS yoso_soha_time,
            {col_yoso_diff_p} AS yoso_gosa_plus,
            {col_yoso_diff_m} AS yoso_gosa_minus,
            {col_yoso_rank} AS yoso_juni,
            {col_blinker} AS blinker,
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
        {where_str}
        ORDER BY date, race_id
        """
        if limit: query += f" LIMIT {limit}"

        logger.info("JRA-VANデータをロード中...")
        print("DEBUG: Executing pd.read_sql with chunksize...", flush=True)
        try:
            chunks = []
            chunk_count = 0
            for chunk in pd.read_sql(query, self.engine, chunksize=10000):
                chunks.append(chunk)
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"DEBUG: Loaded {chunk_count} chunks...", flush=True)
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.DataFrame()
                
            print(f"DEBUG: Read SQL done. Rows: {len(df)}", flush=True)
            logger.info(f"ロード件数 (重複削除前): {len(df)} 件")

            # 重複削除
            before_len = len(df)
            df.drop_duplicates(subset=['race_id', 'horse_number'], inplace=True)
            if len(df) != before_len:
                logger.warning(f"重複データを削除しました: {before_len} -> {len(df)} 件")

            if len(df) == 0:
                logger.warning("注意: 取得データが0件です。")
                return df

            # Python側での時系列オッズ・調教データ結合
            # 10分前オッズをマージ
            if not skip_odds:
                df = self._merge_odds_10min(df, history_start_date)
            else:
                logger.info("  Skip loading 10min odds.")

            if not skip_training:
                df = self._merge_training_data(df, history_start_date, horse_ids)
            else:
                logger.info("  Skip loading training data (jvd_hc).")

            # --- Python側での前処理 ---
            df['rank'] = pd.to_numeric(df['rank_str'], errors='coerce')
            df['odds'] = pd.to_numeric(df['odds'], errors='coerce') / 10.0
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

            # [Fix] odds_10min の数値変換とフォールバック
            # 2026年の過去データシミュレーション用に、10分前オッズが欠損している場合は確定オッズを代用する
            if 'odds_win_str' in df.columns:
                df['odds_10min'] = pd.to_numeric(df['odds_win_str'], errors='coerce') / 10.0
            
            if 'odds_10min' not in df.columns:
                df['odds_10min'] = df['odds']
            else:
                df['odds_10min'] = df['odds_10min'].fillna(df['odds'])
            df['first_3f'] = pd.to_numeric(df['first_3f'], errors='coerce') / 10.0
            df['time_diff'] = pd.to_numeric(df['time_diff'], errors='coerce') / 10.0
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df['impost'] = pd.to_numeric(df['impost'], errors='coerce') / 10.0
            df['weight_ratio'] = df['impost'] / df['weight'].replace(0, np.nan)
            df['honshokin'] = pd.to_numeric(df.get('honshokin', 0), errors='coerce').fillna(0)
            df['fukashokin'] = pd.to_numeric(df.get('fukashokin', 0), errors='coerce').fillna(0)
            df['blinker'] = pd.to_numeric(df.get('blinker', 0), errors='coerce').fillna(0).astype(int)

            def convert_time(t):
                if pd.isna(t): return None
                try:
                    t_str = str(int(t)).zfill(4)
                    return int(t_str[:-3]) * 60 + int(t_str[-3:-1]) + int(t_str[-1]) * 0.1
                except: return None
            df['time'] = df['raw_time'].apply(convert_time)

            def convert_weight_diff(row):
                try:
                    val = int(row['weight_diff_val'])
                    return -val if row['weight_diff_sign'] == '-' else val
                except: return 0
            df['weight_diff'] = df.apply(convert_weight_diff, axis=1)

            def make_passing_rank(row):
                vals = []
                # Use aliased column names from SQL
                pass_cols = ['pass_1', 'pass_2', 'pass_3', 'pass_4']
                for c in pass_cols:
                    v = row.get(c)
                    if pd.notnull(v) and v not in [0, '0', '00']: vals.append(str(v).replace('.0', ''))
                return "-".join(vals) if vals else None
            df['passing_rank'] = df.apply(make_passing_rank, axis=1)

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
                except: return 'Unknown'
            df['surface'] = df['surface'].apply(map_surface)

            state_map = {1: '良', 2: '稍重', 3: '重', 4: '不良'}
            df['state'] = pd.to_numeric(df['state'], errors='coerce').map(state_map).fillna('Unknown')
            
            # Raw codes (Keep numeric)
            df['weather_code'] = pd.to_numeric(df['weather_code'], errors='coerce').fillna(0).astype(int)
            df['going_code'] = pd.to_numeric(df['going_code'], errors='coerce').fillna(0).astype(int)

            if 'date' in df.columns and len(df) > 0:
                logger.info(f"データ読み込み期間: {df['date'].min()} ~ {df['date'].max()}")
                logger.info(f"レース数: {df['race_id'].nunique():,}件, レコード数: {len(df):,}件")

            logger.info(f"前処理完了: {len(df)} 件")
            return df

        except ProgrammingError as e:
            logger.error(f"データロードエラー: {e}")
            raise e
        except Exception as e:
            logger.error(f"データロードエラー: {e}")
            raise e
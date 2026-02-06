
import pandas as pd
import numpy as np
import logging
from ..loader import JraVanDataLoader

logger = logging.getLogger(__name__)

def compute_odds_fluctuation(df: pd.DataFrame) -> pd.DataFrame:
    """
    時系列オッズ (apd_sokuho_o1) を解析し、オッズ変動特徴量を計算する。
    
    Args:
        df: race_id, horse_number, date, start_time_str を含むDataFrame
        
    Returns:
        odds_features: race_id, horse_number をキーとする特徴量DataFrame
    """
    if df.empty:
        return pd.DataFrame()
        
    required_cols = ['race_id', 'horse_number', 'date']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns for odds fluctuation: {required_cols}")
        return pd.DataFrame()

    # 1. 対象レースと期間の特定
    target_races = df['race_id'].unique()
    dates = pd.to_datetime(df['date']).unique()
    start_year = str(dates.min().year)
    
    # 2. DBから時系列オッズ取得
    loader = JraVanDataLoader()
    
    # [Fix] apd_sokuho_o1 (速報) にデータがない2026年以降の対応として、
    # jvd_o1 (公式現状オッズ) も含めるように UNION ALL する。
    # これにより、時系列データがなくても最新のオッズ snapshot を利用可能にする。
    query = f"""
        SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
               happyo_tsukihi_jifun, odds_tansho
        FROM apd_sokuho_o1
        WHERE kaisai_nen >= '{start_year}'
        UNION ALL
        SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
               happyo_tsukihi_jifun, odds_tansho
        FROM jvd_o1
        WHERE kaisai_nen >= '{start_year}'
    """
    
    try:
        logger.info(f"Fetching time-series odds data from {start_year}...")
        df_ts = pd.read_sql(query, loader.engine)
        
        if df_ts.empty:
            return pd.DataFrame()
            
        # race_id 構築
        df_ts['race_id'] = (
            df_ts['kaisai_nen'].astype(str) + 
            df_ts['keibajo_code'].astype(str).str.zfill(2) + 
            df_ts['kaisai_kai'].astype(str).str.zfill(2) + 
            df_ts['kaisai_nichime'].astype(str).str.zfill(2) + 
            df_ts['race_bango'].astype(str).str.zfill(2)
        )
        
        # 対象レースのみにフィルタ
        df_ts = df_ts[df_ts['race_id'].isin(target_races)].copy()
        
        if df_ts.empty:
            return pd.DataFrame()

        # Timestamp 変換 (YYYY + MMDDHHMM)
        df_ts['timestamp_str'] = df_ts['kaisai_nen'] + df_ts['happyo_tsukihi_jifun']
        df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp_str'], format='%Y%m%d%H%M', errors='coerce')
        
        # 3. 発走時刻情報の結合
        # dfからユニークな race_id, start_time を取得
        if 'start_time_str' in df.columns:
            def parse_hhmm(s):
                if pd.isna(s) or len(str(s)) < 3: return pd.Timedelta(0)
                s = str(s).zfill(4)
                return pd.Timedelta(hours=int(s[:2]), minutes=int(s[2:]))
            
            race_times = df[['race_id', 'date', 'start_time_str']].drop_duplicates().copy()
            race_times['start_time'] = race_times['date'] + race_times['start_time_str'].apply(parse_hhmm)
        else:
            # start_time_strがない場合は処理不能（あるいはデフォルト対応）
            logger.warning("start_time_str not found, cannot calc relative time features.")
            return pd.DataFrame()

        df_ts = pd.merge(df_ts, race_times[['race_id', 'start_time']], on='race_id', how='inner')
        
        # 4. 時点特定 (Final vs 10min before)
        # Final: 最も遅いレコード (発走後含む可能性があるが、通常は確定オッズとして扱うにはjvd_seが良い。
        # ここでは「直前オッズ」として、発走時刻以前の最新データを採用する)
        
        df_ts['time_diff'] = (df_ts['start_time'] - df_ts['timestamp']).dt.total_seconds() / 60.0 # Minutes
        
        # Filter records before start_time
        df_valid = df_ts[df_ts['time_diff'] >= 0].copy()
        
        # 10分前: 10 <= diff < 20 (あるいは diff >= 10 の中で最新)
        # ここでは「締め切り10分前」= 発走10? 12? 分前。厳密には「発走10分前」をターゲットにする。
        
        # Group by race_id
        # Latest (Final Pre-Race)
        idx_final = df_valid.groupby('race_id')['timestamp'].idxmax()
        df_final = df_valid.loc[idx_final, ['race_id', 'odds_tansho']].rename(columns={'odds_tansho': 'odds_str_final'})
        
        # 10min Before (e.g., closest to 10 min, or at least 10 min remaining)
        df_10min_candidates = df_valid[df_valid['time_diff'] >= 10].copy()
        if not df_10min_candidates.empty:
            idx_10min = df_10min_candidates.groupby('race_id')['timestamp'].idxmax()
            df_10min = df_valid.loc[idx_10min, ['race_id', 'odds_tansho']].rename(columns={'odds_tansho': 'odds_str_10min'})
        else:
            df_10min = pd.DataFrame(columns=['race_id', 'odds_str_10min'])

        # 60min Before (or Earliest available if race < 60min span, but typically we want ~1h before)
        df_60min_candidates = df_valid[df_valid['time_diff'] >= 55].copy() # Allow slight buffer
        if not df_60min_candidates.empty:
            idx_60min = df_60min_candidates.groupby('race_id')['timestamp'].idxmax()
            df_60min = df_valid.loc[idx_60min, ['race_id', 'odds_tansho']].rename(columns={'odds_tansho': 'odds_str_60min'})
        else:
            # If no 60min data, fallback to earliest data available (Morning Line equivalent)
            idx_earliest = df_valid.groupby('race_id')['timestamp'].idxmin()
            df_60min = df_valid.loc[idx_earliest, ['race_id', 'odds_tansho']].rename(columns={'odds_tansho': 'odds_str_60min'})
            
        # Merge strings back to base
        # We need to expand parsed odds to horse_number level.
        
        def parse_odds_string(row, col_name, suffix):
            """
            Returns DataFrame with [race_id, horse_number, feature_name]
            """
            results = []
            rid = row['race_id']
            s = row[col_name]
            if not isinstance(s, str): return []
            
            chunk_size = 8
            for i in range(0, len(s), chunk_size):
                chunk = s[i:i+chunk_size]
                if len(chunk) < chunk_size: break
                
                try:
                    umaban = int(chunk[0:2])
                    odds_raw = int(chunk[2:6])
                    ninki = int(chunk[6:8])
                    
                    if odds_raw == 9999: odds_val = np.nan
                    else: odds_val = odds_raw / 10.0
                    
                    results.append({
                        'race_id': rid,
                        'horse_number': umaban,
                        f'odds_{suffix}': odds_val,
                        f'ninki_{suffix}': ninki
                    })
                except:
                    continue
            return results

        # Process Final
        parsed_final = []
        for _, row in df_final.iterrows():
            parsed_final.extend(parse_odds_string(row, 'odds_str_final', 'final'))
        
        df_parsed_final = pd.DataFrame(parsed_final)
        
        # Process 10min
        parsed_10min = []
        for _, row in df_10min.iterrows():
            parsed_10min.extend(parse_odds_string(row, 'odds_str_10min', '10min'))
            
        df_parsed_10min = pd.DataFrame(parsed_10min)

        # Process 60min
        parsed_60min = []
        for _, row in df_60min.iterrows():
            parsed_60min.extend(parse_odds_string(row, 'odds_str_60min', '60min'))
            
        df_parsed_60min = pd.DataFrame(parsed_60min)
        
        # 5. 特徴量計算
        # Merge to base df (just keys) to ensure alignment
        # Must include horse_id for FeaturePipeline merge
        base_keys = df[['race_id', 'horse_number', 'horse_id']].copy()
        
        # Ensure types for merge
        base_keys['race_id'] = base_keys['race_id'].astype(str)
        base_keys['horse_number'] = pd.to_numeric(base_keys['horse_number'], errors='coerce').fillna(0).astype(int)
        
        if not df_parsed_final.empty:
            df_parsed_final['race_id'] = df_parsed_final['race_id'].astype(str)
            df_parsed_final['horse_number'] = df_parsed_final['horse_number'].astype(int)
            base_keys = pd.merge(base_keys, df_parsed_final, on=['race_id', 'horse_number'], how='left')
        
        if not df_parsed_10min.empty:
            df_parsed_10min['race_id'] = df_parsed_10min['race_id'].astype(str)
            df_parsed_10min['horse_number'] = df_parsed_10min['horse_number'].astype(int)
            base_keys = pd.merge(base_keys, df_parsed_10min, on=['race_id', 'horse_number'], how='left')

        if not df_parsed_60min.empty:
            df_parsed_60min['race_id'] = df_parsed_60min['race_id'].astype(str)
            df_parsed_60min['horse_number'] = df_parsed_60min['horse_number'].astype(int)
            base_keys = pd.merge(base_keys, df_parsed_60min, on=['race_id', 'horse_number'], how='left')
            
        # Ratio & Diff
        # odds_ratio_10min: Final / 10min ( < 1.0 means odds dropped at the end)
        base_keys['odds_ratio_10min'] = base_keys['odds_final'] / base_keys['odds_10min']
        base_keys['rank_diff_10min'] = base_keys['ninki_final'] - base_keys['ninki_10min']
        
        # odds_ratio_60_10: 10min / 60min ( < 1.0 means odds dropped during the hour leading up to 10min mark)
        # This is LEAK-FREE for Meta-Model (uses only info up to 10min before race)
        base_keys['odds_ratio_60_10'] = base_keys['odds_10min'] / base_keys['odds_60min']
        
        # 乖離特徴量 (Log Ratio)
        base_keys['odds_log_ratio_10min'] = np.log(base_keys['odds_final'] + 1e-9) - np.log(base_keys['odds_10min'] + 1e-9)

        # Output columns
        feature_cols = [
            'odds_ratio_10min', 
            'odds_ratio_60_10',
            'rank_diff_10min',
            'odds_log_ratio_10min',
            'odds_final', 
            'odds_10min',
            'odds_60min'
        ]
        
        # Return only features with index keys
        return base_keys[['race_id', 'horse_number', 'horse_id'] + feature_cols]

    except Exception as e:
        logger.error(f"Error computing odds fluctuation: {e}")
        return pd.DataFrame()

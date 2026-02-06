"""
Auto Predict v13 Market Residual
リアルタイム予測・Discord通知スクリプト

特徴:
- レース発走10分前のオッズを使用
- delta_logit + p_market_snapshot で予測再計算
- 三連複 BOX4 戦略

Usage:
    docker compose exec app python src/scripts/auto_predict_v13.py
    docker compose exec app python src/scripts/auto_predict_v13.py --dry-run
    docker compose exec app python src/scripts/auto_predict_v13.py --date 2025-12-21
"""
import os
import sys
import json
import time
import requests
import argparse
import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from scipy.special import expit, logit as scipy_logit
from itertools import combinations
from sqlalchemy import create_engine

# プロジェクトルート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '../../logs/auto_predict_v13.log'))
    ]
)
logger = logging.getLogger(__name__)

# 定数
STATE_FILE_PATH = os.path.join(os.path.dirname(__file__), '../../data/state/notified_races_v13.json')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models/v13_market_residual')

# 場コード
VENUE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

def load_env_manual():
    """手動で.envファイルを読み込む"""
    try:
        env_path = os.path.join(os.path.dirname(__file__), '../../.env')
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, val = line.split('=', 1)
                        os.environ[key.strip()] = val.strip()
    except Exception as e:
        logger.warning(f".env読み込み失敗: {e}")

def get_db_engine():
    """DB接続エンジンを取得"""
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")


class OddsFetcher:
    """時系列オッズ取得クラス"""
    
    def __init__(self, engine):
        self.engine = engine
    
    def fetch_latest_odds(self, race_id: str, race_dt: datetime, before_minutes: int = 10) -> Optional[Dict[int, float]]:
        """
        指定レースの最新オッズを取得 (発走N分前時点)
        
        Args:
            race_id: YYYY+PP+KK+NN+RR 形式
            race_dt: 発走日時 (datetime)
            before_minutes: 発走何分前までのオッズを取得するか
        
        Returns:
            {horse_number: odds} の辞書
        """
        # race_idからキー要素を抽出
        kaisai_nen = race_id[:4]
        keibajo = race_id[4:6]
        kaisai_kai = race_id[6:8]
        kaisai_nichi = race_id[8:10]
        race_bango = race_id[10:12]
        
        # 基準時刻 (発走 - N分)
        target_dt = race_dt - timedelta(minutes=before_minutes)
        # MMDDHHMM 形式に変換 (apd_sokuho_o1.happyo_tsukihi_jifunと比較用)
        target_ts_str = target_dt.strftime('%m%d%H%M')
        
        query = f"""
        SELECT happyo_tsukihi_jifun, odds_tansho
        FROM apd_sokuho_o1
        WHERE kaisai_nen = '{kaisai_nen}'
          AND keibajo_code = '{keibajo}'
          AND kaisai_kai = '{kaisai_kai}'
          AND kaisai_nichime = '{kaisai_nichi}'
          AND race_bango = '{race_bango}'
          AND happyo_tsukihi_jifun <= '{target_ts_str}'
        ORDER BY happyo_tsukihi_jifun DESC
        LIMIT 1
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            if df.empty:
                return None
            
            odds_str = df.iloc[0]['odds_tansho']
            return self._parse_odds_string(odds_str)
        except Exception as e:
            logger.error(f"オッズ取得エラー: {e}")
            return None
    
    def _parse_odds_string(self, odds_str: str) -> Dict[int, float]:
        """
        オッズ文字列をパース
        Format: 28頭分 × 8文字 (馬番2 + オッズ4 + 人気2)
        """
        result = {}
        if not odds_str or len(odds_str) < 8:
            return result
        
        for i in range(28):  # 最大28頭
            start = i * 8
            if start + 8 > len(odds_str):
                break
            
            chunk = odds_str[start:start + 8]
            try:
                horse_num = int(chunk[0:2])
                odds_val = int(chunk[2:6]) / 10.0
                
                if horse_num > 0 and odds_val > 0:
                    result[horse_num] = odds_val
            except:
                continue
        
        return result


class V13Predictor:
    """v13 market_residual 予測クラス (parquet専用)
    
    運用時は事前に前処理パイプラインを実行してparquetを最新化してください:
        docker compose exec app python src/preprocessing/run_preprocessing.py
    """
    
    def __init__(self):
        self.models = self._load_models()
        self.engine = get_db_engine()
        self.odds_fetcher = OddsFetcher(self.engine)
        
        # 前処理済みデータ (parquet) のキャッシュ
        self._preprocessed_cache = None
        self._parquet_path = os.path.join(os.path.dirname(__file__), '../../data/processed/preprocessed_data.parquet')
    
    def _load_models(self) -> List[lgb.Booster]:
        """v13モデルをロード"""
        models = []
        for fold in ['2022', '2023', '2024']:
            path = os.path.join(MODEL_DIR, f'v13_fold_{fold}.txt')
            if os.path.exists(path):
                models.append(lgb.Booster(model_file=path))
                logger.info(f"モデルロード: {path}")
        
        if not models:
            raise FileNotFoundError(f"モデルが見つかりません: {MODEL_DIR}")
        
        return models
    
    def _get_preprocessed_cache(self) -> pd.DataFrame:
        """前処理済みデータをキャッシュから取得（初回のみロード）"""
        if self._preprocessed_cache is None:
            if os.path.exists(self._parquet_path):
                self._preprocessed_cache = pd.read_parquet(self._parquet_path)
                logger.info(f"前処理データキャッシュ: {len(self._preprocessed_cache)} rows")
            else:
                raise FileNotFoundError(
                    f"前処理データが見つかりません: {self._parquet_path}\n"
                    "→ 先に前処理を実行してください:\n"
                    "   docker compose exec app python src/preprocessing/run_preprocessing.py"
                )
        return self._preprocessed_cache
    
    def get_features(self, date_str: str, race_ids: List[str]) -> pd.DataFrame:
        """
        parquetから特徴量を取得
        
        Args:
            date_str: YYYYMMDD形式の日付
            race_ids: 対象レースIDリスト
        
        Returns:
            特徴量付きDataFrame
        
        Raises:
            ValueError: parquetにデータがない場合
        """
        cache = self._get_preprocessed_cache()
        
        # race_id を文字列に変換して検索
        cache['race_id_str'] = cache['race_id'].astype(str)
        race_ids_str = [str(rid) for rid in race_ids]
        
        result = cache[cache['race_id_str'].isin(race_ids_str)].copy()
        result = result.drop(columns=['race_id_str'], errors='ignore')
        
        found_ids = set(result['race_id'].astype(str).unique()) if not result.empty else set()
        missing_ids = [rid for rid in race_ids if str(rid) not in found_ids]
        
        if missing_ids:
            logger.warning(
                f"parquetにデータなし: {len(missing_ids)} races\n"
                f"  不足レース: {missing_ids[:5]}...\n"
                "→ 前処理を再実行してください:\n"
                "   docker compose exec app python src/preprocessing/run_preprocessing.py"
            )
        
        if not result.empty:
            logger.info(f"parquetから取得: {len(result)} rows, {len(found_ids)} races")
        
        return result
    
    def predict_race(self, race_df: pd.DataFrame, snapshot_odds: Dict[int, float]) -> pd.DataFrame:
        """
        レース予測を実行 (paper_trade_run.py と同じロジック)
        
        Args:
            race_df: 特徴量付きデータ
            snapshot_odds: {horse_number: odds} 時系列オッズ (T-10m)
        
        Returns:
            予測結果DataFrame (prob_residual_softmax, rank 付き)
        """
        df = race_df.copy()
        
        # === LEAK GUARD ===
        # 未来情報(rank等)が含まれていたら例外を投げる
        forbidden_cols = ['rank', 'rank_result', 'kakutei_chakujun', 'payout', 'time', 'agari']
        leaks = [c for c in forbidden_cols if c in df.columns]
        if leaks:
            raise ValueError(f"Leakage detected! Forbidden columns found in input: {leaks}")
        
        # === LEAK PREVENTION ===
        # snapshot oddsをマージ
        df['odds_snapshot'] = df['horse_number'].map(snapshot_odds)
        
        # Snapshot odds から人気順を計算して上書き (Parquetの確定情報を隠蔽)
        if 'odds_snapshot' in df.columns and df['odds_snapshot'].notna().any():
            # オッズ昇順でランク付け (欠損は最下位扱い)
            temp_odds = df['odds_snapshot'].fillna(float('inf'))
            # method='min'で同率は同じ順位
            df['popularity'] = temp_odds.rank(method='min').astype(int)
            
            # odds/tansho_oddsも上書き
            df['odds'] = df['odds_snapshot']
            df['tansho_odds'] = df['odds_snapshot']
        else:
            logger.warning("Snapshot odds not available. Using parquet features intact (Potential Leak if past race).")
        
        # モデル推論
        feature_cols = self.models[0].feature_name()
        
        # 特徴量準備
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0
        
        X = df[feature_cols].fillna(0)
        
        # Ensemble prediction
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        avg_pred = np.mean(preds, axis=0)
        
        # === paper_trade_run.py と同じ変換 ===
        # Store raw logit
        df['raw_score'] = avg_pred
        
        # expit (sigmoid)
        df['prob_residual_raw'] = expit(avg_pred)
        
        # Calculate market probability from T-10m snapshot odds
        df['p_market_raw'] = 1.0 / df['odds_snapshot'].replace(0, np.nan)
        df['p_market'] = df['p_market_raw'] / df['p_market_raw'].sum()
        
        # Softmax per race
        def softmax_race(group):
            exp_vals = np.exp(group - group.max())
            return exp_vals / exp_vals.sum()
        
        df['prob_residual_softmax'] = softmax_race(df['prob_residual_raw'].values)
        
        # Calculate edge (model vs market) - for display
        df['edge'] = df['prob_residual_softmax'] - df['p_market']
        
        # score_logit_snap は raw_score に統一 (表示用)
        df['score_logit_snap'] = df['raw_score']
        
        # ランク (softmax確率でランク付け)
        df['rank'] = df['prob_residual_softmax'].rank(ascending=False)
        
        return df.sort_values('rank')


class SanrenpukuBoxStrategy:
    """三連複BOX4戦略"""
    
    def __init__(self, box_size: int = 4, bet_unit: int = 100):
        self.box_size = box_size
        self.bet_unit = bet_unit
    
    def generate_tickets(self, df: pd.DataFrame) -> List[Dict]:
        """買い目を生成"""
        top_horses = df.nsmallest(self.box_size, 'rank')['horse_number'].astype(int).tolist()
        
        tickets = []
        for combo in combinations(top_horses, 3):
            tickets.append({
                'type': 'sanrenpuku',
                'horses': sorted(combo),
                'bet': self.bet_unit
            })
        
        return tickets
    
    def format_tickets(self, tickets: List[Dict]) -> str:
        """買い目を文字列にフォーマット"""
        if not tickets:
            return "買い目なし"
        
        horses = set()
        for t in tickets:
            horses.update(t['horses'])
        
        horses_str = '-'.join([f'{h:02}' for h in sorted(horses)])
        total_bet = sum(t['bet'] for t in tickets)
        n_tickets = len(tickets)
        
        return f"🎯 三連複 BOX{self.box_size}\n`{horses_str}`\n{n_tickets}点 × ¥{self.bet_unit} = ¥{total_bet:,}"


class DiscordNotifier:
    """Discord通知"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send(self, race_meta: Dict, df: pd.DataFrame, tickets_msg: str) -> bool:
        """予測結果をDiscordに送信"""
        if not self.webhook_url:
            logger.error("Webhook URLが設定されていません")
            return False
        
        venue = VENUE_MAP.get(race_meta.get('venue', ''), race_meta.get('venue', ''))
        race_num = race_meta.get('race_number', '')
        title = race_meta.get('title', '')
        start_time = race_meta.get('start_time', '')
        
        header = f"🏇 【{venue}{race_num}R】{title} ({start_time})"
        
        # Top 5 予測
        top5 = df.nsmallest(5, 'rank')
        prediction_lines = []
        for _, row in top5.iterrows():
            h_num = f"{int(row['horse_number']):02}"
            h_name = row.get('horse_name', '')[:8]
            odds = row.get('odds_snapshot', row.get('odds', 0))
            score = row.get('score_logit_snap', 0)
            prediction_lines.append(f"`{h_num}` {h_name} (単{odds:.1f}, Sc:{score:.2f})")
        
        prediction_text = "\n".join(prediction_lines)
        
        embed = {
            "title": header,
            "description": f"**📊 予測 (T-10m Odds)**\n{prediction_text}\n\n{tickets_msg}",
            "color": 0x00AA00,
            "footer": {"text": "Keiiba-AI v13 (Market Residual + T-10m Snapshot)"}
        }
        
        payload = {"username": "競馬AI v13", "embeds": [embed]}
        
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info(f"通知送信成功: {race_meta.get('race_id', '')}")
            return True
        except Exception as e:
            logger.error(f"通知送信失敗: {e}")
            return False


class AutoPredictV13:
    """メインクラス"""
    
    def __init__(self, dry_run: bool = False, target_date: str = None):
        self.dry_run = dry_run
        self.target_date = target_date
        self.state_file = STATE_FILE_PATH
        self.notified_races = self._load_state()
        
        load_env_manual()
        
        self.predictor = V13Predictor()
        self.strategy = SanrenpukuBoxStrategy(box_size=4)
        
        webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
        self.notifier = DiscordNotifier(webhook_url)
        
        self.engine = get_db_engine()
    
    def _load_state(self) -> set:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return set(json.load(f))
            except:
                pass
        return set()
    
    def _save_state(self):
        if self.dry_run:
            return
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(list(self.notified_races), f)
    
    def _load_race_list(self, date_str: str) -> pd.DataFrame:
        """当日のレース一覧をDBから取得"""
        # date_str: YYYYMMDD
        year = date_str[:4]
        mmdd = date_str[4:8]
        
        query = f"""
        SELECT 
            CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
            keibajo_code as venue,
            race_bango as race_number,
            kaisai_tsukihi,
            hasso_jikoku as start_time,
            kyosomei_hondai as title
        FROM jvd_ra
        WHERE kaisai_nen = '{year}'
          AND kaisai_tsukihi = '{mmdd}'
          AND keibajo_code BETWEEN '01' AND '10'
          AND data_kubun = '7'
        ORDER BY keibajo_code, race_bango
        """
        
        return pd.read_sql(query, self.engine)
    
    def _load_race_entries(self, race_id: str) -> pd.DataFrame:
        """出馬表を取得"""
        kaisai_nen = race_id[:4]
        keibajo = race_id[4:6]
        kaisai_kai = race_id[6:8]
        kaisai_nichi = race_id[8:10]
        race_bango = race_id[10:12]
        
        query = f"""
        SELECT 
            res.umaban as horse_number,
            res.bamei as horse_name,
            res.kishu_code as jockey_id,
            res.tansho_odds as odds,
            res.wakuban as frame_number,
            res.futan_juryo as impost
        FROM jvd_se res
        WHERE res.kaisai_nen = '{kaisai_nen}'
          AND res.keibajo_code = '{keibajo}'
          AND res.kaisai_kai = '{kaisai_kai}'
          AND res.kaisai_nichime = '{kaisai_nichi}'
          AND res.race_bango = '{race_bango}'
        ORDER BY res.umaban
        """
        
        df = pd.read_sql(query, self.engine)
        df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')
        df['odds'] = pd.to_numeric(df['odds'], errors='coerce') / 10.0
        df['frame_number'] = pd.to_numeric(df['frame_number'], errors='coerce').fillna(0)
        df['impost'] = pd.to_numeric(df['impost'], errors='coerce').fillna(0)
        df['race_id'] = race_id
        
        return df
    
    def run(self):
        """メイン処理 (単日)"""
        logger.info("=== Auto Predict v13 開始 ===")
        
        now = datetime.now()
        if self.target_date:
            date_str = self.target_date.replace('-', '')
        else:
            date_str = now.strftime('%Y%m%d')
        
        logger.info(f"対象日: {date_str}")
        
        race_list = self._load_race_list(date_str)
        if race_list.empty:
            logger.info("本日のレースなし")
            return
            
        self._process_races(race_list, now if not self.target_date else None, date_str)
        
        self._save_state()
        logger.info("=== Auto Predict v13 完了 ===")

    def batch_run_year(self, year: str, jra_only: bool = True) -> pd.DataFrame:
        """指定年の全レースをまとめて予測"""
        logger.info(f"=== Batch Predict Year: {year} ===")
        
        query = f"""
        SELECT DISTINCT kaisai_tsukihi as date
        FROM jvd_ra
        WHERE kaisai_nen = '{year}'
          AND data_kubun = '7'
        """
        if jra_only:
             query += " AND keibajo_code BETWEEN '01' AND '10'"
             
        df_dates = pd.read_sql(query, self.engine)
        dates = sorted(df_dates['date'].unique())
        logger.info(f"Found {len(dates)} dates.")
        
        all_results = []
        
        for d in dates:
            date_str = f"{year}{d}"
            # logger.info(f"Processing {date_str}...") # Reduce log noise
            
            race_list = self._load_race_list(date_str)
            if race_list.empty:
                continue
                
            results = self._process_races_batch(race_list, date_str)
            if results:
                all_results.extend(results)
                
        if not all_results:
            return pd.DataFrame()
            
        logger.info(f"Generated {len(all_results)} race predictions.")
        return pd.concat(all_results, ignore_index=True)

    def _process_races_batch(self, race_list: pd.DataFrame, date_str: str) -> List[pd.DataFrame]:
        """バッチ処理用"""
        targets = []
        for _, row in race_list.iterrows():
            race_id = row['race_id']
            start_time_str = str(row['start_time']).zfill(4)
            try:
                race_dt = datetime.strptime(f"{date_str}{start_time_str}", "%Y%m%d%H%M")
                targets.append((row, race_dt))
            except:
                continue
        
        if not targets:
            return []
            
        target_race_ids = [row['race_id'] for row, _ in targets]
        all_features_df = self.predictor.get_features(date_str, race_ids=target_race_ids)
        
        if all_features_df.empty:
            return []
            
        results = []
        for row, race_dt in targets:
            race_id = row['race_id']
            entries = all_features_df[all_features_df['race_id'].astype(str) == str(race_id)].copy()
            if entries.empty:
                continue
            
            # Strict Snapshot Odds (No Fallback)
            snapshot_odds = self.predictor.odds_fetcher.fetch_latest_odds(race_id, race_dt, before_minutes=10)
            
            if not snapshot_odds:
                continue
            
            # Cleanup parquet potential leaks before prediction
            # odds/popularity are overwritten by predict_race (safe), but rank/time must be dropped
            drop_cols = ['rank', 'time', 'agari', 'kakutei_chakujun']
            entries = entries.drop(columns=[c for c in drop_cols if c in entries.columns], errors='ignore')

            try:
                pred_df = self.predictor.predict_race(entries, snapshot_odds)
                
                # Metadata
                pred_df['race_id'] = race_id
                pred_df['post_time'] = race_dt
                pred_df['snapshot_time_used'] = race_dt - timedelta(minutes=10)
                pred_df['odds_tminus10m'] = pred_df['odds_snapshot']
                pred_df['popularity_tminus10m'] = pred_df['popularity']
                pred_df['p_market_tminus10m'] = pred_df['p_market']
                
                save_cols = [
                    'race_id', 'horse_number', 'post_time', 'snapshot_time_used',
                    'odds_tminus10m', 'popularity_tminus10m', 'p_market_tminus10m',
                    'raw_score', 'prob_residual_softmax', 'rank'
                ]
                if 'delta_logit' in pred_df.columns:
                     save_cols.append('delta_logit')
                     
                results.append(pred_df[save_cols])
            except Exception as e:
                # logger.error(f"Error predicting {race_id}: {e}")
                pass
                
        return results

    def _process_races(self, race_list: pd.DataFrame, now: Optional[datetime], date_str: str):
        """通常実行用"""
        targets = []
        for _, row in race_list.iterrows():
            race_id = row['race_id']
            if race_id in self.notified_races: continue
            
            start_time_str = str(row['start_time']).zfill(4)
            try:
                race_dt = datetime.strptime(f"{date_str}{start_time_str}", "%Y%m%d%H%M")
            except: continue
            
            if now:
                diff_min = (race_dt - now).total_seconds() / 60
                if 5 <= diff_min <= 15:
                    targets.append((row, race_dt))
            else:
                targets.append((row, race_dt))
        
        if not targets:
            logger.info("対象レースなし (5-15分前のレースがない)")
            return

        target_race_ids = [r['race_id'] for r, _ in targets]
        all_features_df = self.predictor.get_features(date_str, race_ids=target_race_ids) # Parquet
        
        if all_features_df.empty: return

        for row, race_dt in targets:
            race_id = row['race_id']
            entries = all_features_df[all_features_df['race_id'].astype(str) == str(race_id)].copy()
            if entries.empty: continue
            
            # Drop leaks
            drop_cols = ['rank', 'time', 'agari', 'kakutei_chakujun']
            entries = entries.drop(columns=[c for c in drop_cols if c in entries.columns], errors='ignore')
            
            snapshot_odds = self.predictor.odds_fetcher.fetch_latest_odds(race_id, race_dt, before_minutes=10)
            if not snapshot_odds:
                logger.warning(f"オッズ取得失敗: {race_id}")
                continue # No Fallback

            try:
                result_df = self.predictor.predict_race(entries, snapshot_odds)
                tickets = self.strategy.generate_tickets(result_df)
                tickets_msg = self.strategy.format_tickets(tickets)
                
                race_meta = {
                    'race_id': race_id, 'venue': row['venue'], 'race_number': int(row['race_number']),
                    'title': row['title'] or '', 'start_time': f"{start_time_str[:2]}:{start_time_str[2:]}"
                }
                
                if self.dry_run:
                    logger.info(f"[DRY-RUN] {race_meta}")
                    print(result_df[['horse_number', 'horse_name', 'odds_snapshot', 'popularity', 'score_logit_snap', 'rank']].head(8))
                    print(tickets_msg)
                else:
                    if self.notifier.send(race_meta, result_df, tickets_msg):
                        self.notified_races.add(race_id)
                    time.sleep(1.5)
            except Exception as e:
                logger.error(f"Error {race_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Auto Predict v13 (T-10m Odds + 三連複BOX4)')
    parser.add_argument('--dry-run', action='store_true', help='通知を送信せずに実行')
    parser.add_argument('--date', type=str, help='対象日付 (YYYY-MM-DD or YYYYMMDD)')
    parser.add_argument('--year', type=str, help='指定年をまとめて処理 (Batch Mode)')
    parser.add_argument('--jra_only', action='store_true', default=True, help='JRAのみ')
    parser.add_argument('--out', type=str, help='出力先parquetパス (Batch Mode用)')
    parser.add_argument('--run_leak_proof', action='store_true', help='Leak Proof Mode Flag (Dummy for compatibility)')
    
    args = parser.parse_args()
    
    predictor = AutoPredictV13(dry_run=args.dry_run, target_date=args.date)
    
    if args.year:
        if not args.out:
            print("Error: --out is required when using --year")
            return
        df = predictor.batch_run_year(args.year, args.jra_only)
        if not df.empty:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            df.to_parquet(args.out)
            logger.info(f"Saved {len(df)} rows to {args.out}")
        else:
            logger.warning("No predictions generated.")
    else:
        predictor.run()


if __name__ == "__main__":
    main()

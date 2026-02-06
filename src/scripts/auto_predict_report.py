"""
Auto Predict Report Script
リアルタイム予測を行い、全頭スコアをMarkdownレポートとして出力し、Discord通知を行う。

Based on src/scripts/auto_predict_v13.py
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
from scipy.special import expit
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
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '../../logs/auto_predict_report.log'))
    ]
)
logger = logging.getLogger(__name__)

# 定数
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models/v13_market_residual')
REPORT_DIR = os.path.join(os.path.dirname(__file__), '../../reports/jra/daily')

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
        # シンプル化: 常に最新オッズを取るロジックにするか、指定時間前をとるか
        # 運用では「直前」に実行することが多いが、T-10m指定があるならそれに従う
        # ここでは auto_predict_v13.py と同様のクエリを使用
        kaisai_nen = race_id[:4]
        keibajo = race_id[4:6]
        kaisai_kai = race_id[6:8]
        kaisai_nichi = race_id[8:10]
        race_bango = race_id[10:12]
        
        target_dt = race_dt - timedelta(minutes=before_minutes)
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
            return self._parse_odds_string(df.iloc[0]['odds_tansho'])
        except Exception as e:
            logger.error(f"オッズ取得エラー: {e}")
            return None
    
    def _parse_odds_string(self, odds_str: str) -> Dict[int, float]:
        result = {}
        if not odds_str or len(odds_str) < 8: return result
        for i in range(28):
            start = i * 8
            if start + 8 > len(odds_str): break
            chunk = odds_str[start:start + 8]
            try:
                horse_num = int(chunk[0:2])
                odds_val = int(chunk[2:6]) / 10.0
                if horse_num > 0 and odds_val > 0:
                    result[horse_num] = odds_val
            except: continue
        return result

class V13Predictor:
    """v13 market_residual 予測クラス"""
    def __init__(self):
        self.models = self._load_models()
        self.engine = get_db_engine()
        self.odds_fetcher = OddsFetcher(self.engine)
        self._preprocessed_cache = None
        self._parquet_path = os.path.join(os.path.dirname(__file__), '../../data/processed/preprocessed_data_v11.parquet')
        
        # Load Calibrator if exists
        self.calibrator = None
        calib_path = os.path.join(os.path.dirname(__file__), '../../models/calibrator.pkl')
        if os.path.exists(calib_path):
            try:
                import joblib
                self.calibrator = joblib.load(calib_path)
                logger.info(f"Calibrator loaded: {calib_path}")
            except Exception as e:
                logger.warning(f"Failed to load calibrator: {e}")
    
    def _load_models(self) -> List[lgb.Booster]:
        models = []
        for fold in ['2022', '2023', '2024']:
            path = os.path.join(MODEL_DIR, f'v13_fold_{fold}.txt')
            if os.path.exists(path):
                models.append(lgb.Booster(model_file=path))
        if not models:
            raise FileNotFoundError(f"モデルが見つかりません: {MODEL_DIR}")
        return models
    
    def _get_preprocessed_cache(self) -> pd.DataFrame:
        if self._preprocessed_cache is None:
            if os.path.exists(self._parquet_path):
                self._preprocessed_cache = pd.read_parquet(self._parquet_path)
            else:
                raise FileNotFoundError(f"前処理データが見つかりません: {self._parquet_path}")
        return self._preprocessed_cache
    
    def get_features(self, date_str: str, race_ids: List[str]) -> pd.DataFrame:
        cache = self._get_preprocessed_cache()
        cache['race_id_str'] = cache['race_id'].astype(str)
        race_ids_str = [str(rid) for rid in race_ids]
        result = cache[cache['race_id_str'].isin(race_ids_str)].copy()
        return result.drop(columns=['race_id_str'], errors='ignore')
    
    def predict_race(self, race_df: pd.DataFrame, snapshot_odds: Dict[int, float]) -> pd.DataFrame:
        df = race_df.copy()
        
        # Leak prevention
        forbidden_cols = ['rank', 'rank_result', 'kakutei_chakujun', 'payout', 'time', 'agari']
        leaks = [c for c in forbidden_cols if c in df.columns]
        if leaks:
            raise ValueError(f"Leakage detected! {leaks}")
        
        df['odds_snapshot'] = df['horse_number'].map(snapshot_odds)
        if 'odds_snapshot' in df.columns and df['odds_snapshot'].notna().any():
            temp_odds = df['odds_snapshot'].fillna(float('inf'))
            df['popularity'] = temp_odds.rank(method='min').astype(int)
            df['odds'] = df['odds_snapshot']
            df['tansho_odds'] = df['odds_snapshot']
        
        feature_cols = self.models[0].feature_name()
        for c in feature_cols:
            if c not in df.columns: df[c] = 0
            
        X = df[feature_cols].fillna(0)
        
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        avg_pred = np.mean(preds, axis=0)
        
        df['raw_score'] = avg_pred
        df['prob_residual_raw'] = expit(avg_pred)
        
        # Handle missing odds for market probability
        if 'odds_snapshot' in df.columns and df['odds_snapshot'].notna().any():
             # Avoid division by zero
            df['p_market_raw'] = 1.0 / df['odds_snapshot'].replace(0, np.nan)
            df['p_market'] = df['p_market_raw'] / df['p_market_raw'].sum()
        else:
            df['p_market_raw'] = 0.0
            df['p_market'] = 0.0
        
        # Softmax per race
        def softmax_race(group):
            exp_vals = np.exp(group - group.max())
            return exp_vals / exp_vals.sum()
        
        df['prob_residual_softmax'] = softmax_race(df['prob_residual_raw'].values)
        
        # === Calibration (Match Backtest) ===
        if self.calibrator:
            try:
                # Scikit-learn expects 2D array for Logistic, 1D for Isotonic
                if hasattr(self.calibrator, "predict_proba"):
                    raw_scores = df['raw_score'].values.reshape(-1, 1)
                    df['prob_calib'] = self.calibrator.predict_proba(raw_scores)[:, 1]
                else:
                    # IsotonicRegression or similar
                    df['prob_calib'] = self.calibrator.predict(df['raw_score'].values)
                
                # Rank by Calibrated Prob
                df['rank'] = df['prob_calib'].rank(ascending=False)
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                df['prob_calib'] = df['prob_residual_softmax']
                df['rank'] = df['prob_residual_softmax'].rank(ascending=False)
        else:
            # Fallback to Softmax
            df['prob_calib'] = df['prob_residual_softmax']
            df['rank'] = df['prob_residual_softmax'].rank(ascending=False)

        df['score_logit_snap'] = df['raw_score']
        
        return df.sort_values('rank')

class ReportGenerator:
    """Markdownレポート生成クラス"""
    @staticmethod
    def generate_markdown(date_str: str, all_predictions: List[Dict]) -> str:
        """
        Args:
            date_str: YYYY-MM-DD
            all_predictions: List of {meta: Dict, df: pd.DataFrame}
        """
        lines = []
        lines.append(f"# 競馬AI 予測レポート ({date_str})")
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        total_races = len(all_predictions)
        venues = sorted(list(set([d['meta']['venue'] for d in all_predictions])))
        lines.append(f"- 対象レース数: {total_races}")
        lines.append(f"- 開催場: {', '.join(venues)}")
        lines.append("")
        
        # Details
        for item in all_predictions:
            meta = item['meta']
            df = item['df']
            
            # Header
            venue = VENUE_MAP.get(meta['venue'], meta['venue'])
            race_num = meta['race_number']
            title = meta['title']
            start_time = meta['start_time']
            
            lines.append(f"## {venue} {race_num}R {title} ({start_time})")
            lines.append("")
            lines.append("| Rank | No. | Horse Name | Score | Odds (T-10) | Prob(%) |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            
            for _, row in df.iterrows():
                rank = int(row['rank'])
                h_num = int(row['horse_number'])
                h_name = row.get('horse_name', '')
                score = row.get('raw_score', 0)
                odds = row.get('odds_snapshot', 0)
                prob = row.get('prob_residual_softmax', 0) * 100
                
                # Highlight top horses
                rank_str = f"**{rank}**" if rank <= 3 else str(rank)
                
                odds_str = f"{odds:.1f}" if odds > 0 else "---"
                lines.append(f"| {rank_str} | {h_num:02} | {h_name} | {score:.3f} | {odds_str} | {prob:.1f}% |")
            
            lines.append("")
            
        return "\n".join(lines)

class AutoPredictReport:
    """メインクラス"""
    def __init__(self, dry_run: bool = False, target_date: str = None):
        self.dry_run = dry_run
        self.target_date = target_date
        load_env_manual()
        
        self.predictor = V13Predictor()
        self.webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
        self.engine = get_db_engine()
        
        os.makedirs(REPORT_DIR, exist_ok=True)
        
    def _load_race_list(self, date_str: str) -> pd.DataFrame:
        year = date_str[:4]
        mmdd = date_str[4:8]
        query = f"""
        SELECT 
            CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
            keibajo_code as venue,
            race_bango as race_number,
            hasso_jikoku as start_time,
            kyosomei_hondai as title
        FROM jvd_ra
        WHERE kaisai_nen = '{year}' AND kaisai_tsukihi = '{mmdd}'
          AND data_kubun IN ('7', '5', '4', '2', '1')
          AND keibajo_code BETWEEN '01' AND '10'
        ORDER BY keibajo_code, race_bango
        """
        return pd.read_sql(query, self.engine)

        # Notify Discord (Per Race)
        if not self.dry_run:
            self._send_race_notification(meta, pred_df)
            time.sleep(1) # Rate limit

    def run(self):
        logger.info("=== Auto Predict Report Start ===")
        
        now = datetime.now()
        if self.target_date:
            target_dt = datetime.strptime(self.target_date, '%Y-%m-%d')
            date_str = target_dt.strftime('%Y%m%d')
            display_date = target_dt.strftime('%Y-%m-%d')
        else:
            date_str = now.strftime('%Y%m%d')
            display_date = now.strftime('%Y-%m-%d')
            
        logger.info(f"Target Date: {display_date}")
        
        race_list = self._load_race_list(date_str)
        if race_list.empty:
            logger.warning("No races found.")
            return

        # Fetch Features Batch
        race_ids = race_list['race_id'].tolist()
        features = self.predictor.get_features(date_str, race_ids)
        
        if features.empty:
            logger.error("No features available in parquet.")
            return
            
        predictions = []
        
        logger.info(f"Processing {len(race_list)} races...")
        for _, row in race_list.iterrows():
            race_id = row['race_id']
            # Get specific race features
            race_feat = features[features['race_id'].astype(str) == str(race_id)].copy()
            if race_feat.empty: continue
            
            # Race Time
            start_time_str = str(row['start_time']).zfill(4)
            race_dt = datetime.strptime(f"{date_str}{start_time_str}", "%Y%m%d%H%M")
            
            # Drop leaks
            drop_cols = ['rank', 'time', 'agari', 'kakutei_chakujun']
            race_feat = race_feat.drop(columns=[c for c in drop_cols if c in race_feat.columns], errors='ignore')
            
            # Get Odds
            snapshot_odds = self.predictor.odds_fetcher.fetch_latest_odds(race_id, race_dt, before_minutes=10)
            
            if not snapshot_odds:
                # Try Immediate
                snapshot_odds = self.predictor.odds_fetcher.fetch_latest_odds(race_id,  race_dt + timedelta(hours=1), before_minutes=0)
            
            if not snapshot_odds:
                logger.warning(f"No odds for {race_id} (Predicting details only)")
                snapshot_odds = {}
                
            try:
                pred_df = self.predictor.predict_race(race_feat, snapshot_odds)
                
                meta = {
                    'race_id': race_id,
                    'venue': VENUE_MAP.get(row['venue'], row['venue']),
                    'race_number': int(row['race_number']),
                    'title': row['title'],
                    'start_time': f"{start_time_str[:2]}:{start_time_str[2:]}"
                }
                
                predictions.append({'meta': meta, 'df': pred_df})

                # Notify Discord immediately
                if not self.dry_run:
                    self._send_race_notification(meta, pred_df)
                    time.sleep(1) 

            except Exception as e:
                logger.error(f"Prediction error {race_id}: {e}")

        if not predictions:
            logger.warning("No predictions generated.")
            return

        # Generate Full Report (for archive)
        md_content = ReportGenerator.generate_markdown(display_date, predictions)
        filename = f"{display_date}_predictions.md"
        filepath = os.path.join(REPORT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"Report saved to: {filepath}")
        logger.info("=== Completed ===")

    def _send_race_notification(self, meta, df):
        if not self.webhook_url:
            return

        venue = meta['venue']
        r_num = meta['race_number']
        title = meta['title']
        time_str = meta['start_time']
        
        header = f"🏇 【{venue}{r_num}R】{title} ({time_str})"
        
        # Top 5 予測 (v13 Format)
        top5 = df.nsmallest(5, 'rank')
        prediction_lines = []
        for _, row in top5.iterrows():
            h_num = f"{int(row['horse_number']):02}"
            h_name = row.get('horse_name', '')[:8]
            odds = row.get('odds_snapshot', 0)
            score = row.get('raw_score', 0) # Use Raw score for consistency with v13 log
            
            # v13 displays: `01` Horse (単3.5, Sc:0.88)
            prediction_lines.append(f"`{h_num}` {h_name} (単{odds:.1f}, Sc:{score:.2f})")
        
        prediction_text = "\n".join(prediction_lines)
        
        # Construct Embed
        embed = {
            "title": header,
            "description": f"**📊 予測 (T-10m Odds)**\n{prediction_text}",
            "color": 0x00AA00,
            "footer": {"text": "Keiiba-AI v13 (Market Residual + T-10m Snapshot)"}
        }
        
        payload = {
            "username": "競馬AI v13 Report",
            "embeds": [embed]
        }
        
        try:
            requests.post(self.webhook_url, json=payload)
            logger.info(f"Sent notification for {venue} {r_num}R")
        except Exception as e:
            logger.error(f"Discord Send Error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--date', type=str, help='YYYY-MM-DD')
    args = parser.parse_args()
    
    app = AutoPredictReport(dry_run=args.dry_run, target_date=args.date)
    app.run()

if __name__ == "__main__":
    main()

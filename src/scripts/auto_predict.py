import os
import sys
import json
import time
import requests
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from scipy.stats import entropy
import pickle

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.ensemble import EnsembleModel
from src.inference.preprocessor import InferencePreprocessor
from src.inference.loader import InferenceDataLoader
from src.model.calibration import ProbabilityCalibrator

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '../../logs/auto_predict.log'))
    ]
)
logger = logging.getLogger(__name__)


# .env 手動読み込み
def load_env_manual():
    try:
        env_path = os.path.join(os.path.dirname(__file__), '../../.env')
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    if '=' in line:
                        key, val = line.split('=', 1)
                        os.environ[key.strip()] = val.strip()
    except Exception as e:
        logger.warning(f".env reading failed: {e}")

# 定数
STATE_FILE_PATH = os.path.join(os.path.dirname(__file__), '../../data/state/notified_races.json')

class NotificationManager:
    """Discord通知を管理するクラス"""
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def _calculate_confidence(self, df: pd.DataFrame) -> tuple[str, str]:
        """ハザード(波乱度)と自信度を判定"""
        # 勝率分布のエントロピー
        probs = df['calibrated_prob'].values
        ent = entropy(probs)
        
        top_horse = df.sort_values('score', ascending=False).iloc[0]
        top_prob = top_horse['calibrated_prob']
        top_score = top_horse['score']
        
        # 判定ロジック
        if top_prob >= 0.40 or top_score >= 1.5:
            return "S", "鉄板 (Ironclad)"
        elif top_prob >= 0.25 or top_score >= 0.8:
            return "A", "安定 (Stable)"
        elif ent > 2.0 or top_prob < 0.20:
             return "C", "波乱 (High Chaos)"
        else:
             return "B", "混戦 (Confusion)"

    def send_prediction(self, race_meta: Dict, df: pd.DataFrame):
        """
        予測結果をDiscordに送信します。
        Args:
            race_meta: レース情報
            df: 予測結果DataFrame
        Returns:
            bool: 送信成功ならTrue
        """
        if not self.webhook_url:
            logger.warning("Discord Webhook URLが設定されていません。通知をスキップします。")
            return False

        # タイトル整形
        date_str = race_meta.get('date', '')
        
        # 波乱度判定
        chart_rank, chart_desc = self._calculate_confidence(df)
        
        title_str = f"🎯 [{date_str}] {race_meta['venue_name']}{race_meta['race_number']}R {race_meta['title']} ({race_meta['start_time']}) - [{chart_rank}] {chart_desc}"

        # 予測テーブル作成 (全頭: スコア順 -> 最も純粋な強さ評価)
        top_picks = df.sort_values('score', ascending=False)
        
        description = "**🏆 本命予測 (スコア順)**\n"
        
        # ヘッダーなし、リスト形式で見やすく
        marks = ["◎", "〇", "▲", "△", "△", "△"]
        
        for i, (_, row) in enumerate(top_picks.iterrows()):
            mark = marks[i] if i < len(marks) else "  "
            h_num = str(int(row['horse_number'])).zfill(2)
            h_name = row['horse_name']
            
            ev = f"{row['expected_value']:.2f}"
            prob = f"{row['calibrated_prob']*100:.0f}%"
            score = f"{row['score']:.2f}"
            
            # Simple list format with Score
            description += f"`{mark}` `{h_num}` **{h_name}** (勝率:{prob}, EV:{ev}, Sc:{score})\n"

        # 推奨買い目 (Smart Value Logic)
        bet_strategy = self._generate_betting_strategy(df)
        
        # NetKeiba Link
        # ID形式補正: YYYY(4) + Venue(2) + Kai(2) + Nichi(2) + R(2) 
        # race_meta['race_id'] は通常この形式。
        netkeiba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_meta['race_id']}"
        description += f"\n🔗 [NetKeiba]({netkeiba_url})\n"
        
        embed = {
            "title": title_str,
            "description": description + "\n" + bet_strategy,
            "color": 0xFF0000 if top_picks.iloc[0]['expected_value'] > 1.5 else (0x00FF00 if chart_rank in ['S', 'A'] else 0xFFA500), # S/Aなら緑、それ以外はオレンジ、高EVは赤
            "footer": {
                "text": "Keiiba-AI Prediction System"
            }
        }
        
        payload = {
            "username": "ナミール",
            "embeds": [embed]
        }
        
        try:
            resp = requests.post(self.webhook_url, json=payload)
            resp.raise_for_status()
            logger.info(f"通知送信成功: {race_meta['race_id']}")
            return True
        except Exception as e:
            logger.error(f"通知送信失敗: {e}")
            return False

    def _pad_width(self, s: str, width: int) -> str:
        """全角文字を考慮してパディングする簡易関数"""
        count = 0
        for c in s:
            if ord(c) > 255: count += 2
            else: count += 1
        
        padding = width - count
        if padding > 0:
            return s + " " * padding
        else:
            return s

    def _generate_betting_strategy(self, df: pd.DataFrame) -> str:
        """推奨買い目のテキスト生成 (Smart Value Logic)"""
        # 1. スコア上位6頭を抽出 (安定群)
        top_prob_df = df.sort_values('score', ascending=False).head(6)
        
        # 2. その中で最もEVが高い馬を「狙い目」とする
        best_smart_horse = top_prob_df.sort_values('expected_value', ascending=False).iloc[0]
        
        # 3. 純粋な勝率1位 (本命)
        best_prob_horse = top_prob_df.iloc[0]
        
        msg = "**🎫 推奨買い目**\n"
        
        # A. 本命 (勝率 1位)
        p_num = int(best_prob_horse['horse_number'])
        p_name = best_prob_horse['horse_name']
        p_prob = best_prob_horse['calibrated_prob']
        p_ev = best_prob_horse['expected_value']
        
        msg += f"👑 **本命 (堅実)**: {p_num} {p_name}\n"
        msg += f"   (勝率: {p_prob*100:.1f}%, EV: {p_ev:.2f}) -> 単勝/連軸\n"
        
        # B. 狙い目 (上位5頭の中でBest EV)
        # 本命と異なる場合のみ表示
        if int(best_smart_horse['horse_number']) != p_num:
            v_num = int(best_smart_horse['horse_number'])
            v_name = best_smart_horse['horse_name']
            v_prob = best_smart_horse['calibrated_prob']
            v_ev = best_smart_horse['expected_value']
            
            # EVが1.0を超えている場合のみ推奨
            if v_ev > 1.0:
                msg += f"💰 **狙い目 (高期待値)**: {v_num} {v_name}\n"
                msg += f"   (勝率: {v_prob*100:.1f}%, EV: {v_ev:.2f}) -> 単複/ワイド相手\n"
        
        # 全体的なコメント
        if p_ev < 1.0 and best_smart_horse['expected_value'] < 1.0:
            msg += "\n⚠️ **全体的に期待値低め (見送り推奨)**\n"
            
        return msg

class AutoPredictor:
    def __init__(self, dry_run: bool = False, target_date: str = None):
        self.dry_run = dry_run
        self.target_date = target_date
        self.state_file = STATE_FILE_PATH
        self.notified_races = self._load_state()
        
        # モデル初期化 (初回のみロード)
        self.loader = InferenceDataLoader()
        self.preprocessor = InferencePreprocessor()
        self.calibrator = self._load_calibrator()
        self.model = self._load_model() # Ensemble
        
        # Load env vars manually to ensure Webhook URL is present
        load_env_manual()
        webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
        if not webhook_url:
            logger.error("❌ DISCORD_WEBHOOK_URL is not set. Notifications will fail.")
            
        self.notifier = NotificationManager(webhook_url)
        
    def _load_state(self) -> set:
        """通知済みレースIDの読み込み"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()

    def _save_state(self):
        """通知済みレースIDの保存"""
        if self.dry_run: return
        
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(list(self.notified_races), f)

    def _load_model(self):
        logger.info("モデル(Ensemble)をロード中...")
        model = EnsembleModel()
        model_dir = os.path.join(os.path.dirname(__file__), '../../models')
        # 優先順位: v5 (JRA Specialist)
        path = os.path.join(model_dir, 'ensemble_v5.pkl')
        
        if os.path.exists(path):
            model.load_model(path)
            return model
        else:
            logger.error(f"モデルファイルが見つかりません: {path}")
            return None

    def _load_calibrator(self):
        model_dir = os.path.join(os.path.dirname(__file__), '../../models')
        path = os.path.join(model_dir, 'calibrator.pkl')
        if os.path.exists(path):
            try:
                calib = ProbabilityCalibrator() # クラス定義済みと仮定
                calib.load(path)
                return calib
            except:
                with open(path, 'rb') as f:
                     # Calibratorクラスが見つからない場合のフォールバック(pickle直読みは危険だが、calibration.pyからクラスを持ってくるべき)
                     pass
                return None
        return None

    def run(self):
        """メイン実行ループ"""
        logger.info("自動予測プロセスを実行します...")
        
        # 1. 開催日/現在時刻の取得
        now = datetime.now()
        if self.target_date:
            today_str = self.target_date.replace('-', '')
        else:
            today_str = now.strftime('%Y%m%d')

        # 2. レース一覧取得
        race_list_df = self.loader.load_race_list(today_str)
        if race_list_df.empty:
            logger.info("本日の開催レースはありません。")
            return

        # 3. 直前レースのフィルタリング
        targets = []
        for _, row in race_list_df.iterrows():
            race_id = row['race_id']
            if race_id in self.notified_races:
                continue
                
            start_time_str = row['start_time']
            if not start_time_str: continue

            try:
                race_dt = datetime.strptime(f"{today_str}{start_time_str}", "%Y%m%d%H%M")
            except ValueError:
                continue

            if self.target_date:
                targets.append(row)
            else:
                diff = race_dt - now
                minutes = diff.total_seconds() / 60
                if 15 <= minutes <= 35:
                     targets.append(row)
        
        if not targets:
            logger.info("現在、直前の通知対象レースはありません。")
            return
            
        logger.info(f"通知対象レース: {len(targets)} 件")

        # 4. 推論 & 通知
        target_ids = [r['race_id'] for r in targets]
        
        try:
            raw_df = self.loader.load(target_date=today_str, race_ids=target_ids)
        except Exception as e:
            logger.error(f"データロードエラー: {e}")
            return
            
        if raw_df.empty:
            logger.warning("レースデータが空です。")
            return

        # 前処理
        X, ids = self.preprocessor.preprocess(raw_df)
        
        # 予測 (Score)
        try:
            scores = self.model.predict(X)
        except Exception as e:
            logger.error(f"予測エラー: {e}")
            return

        # Calibration
        if self.calibrator:
            calibrated_probs = self.calibrator.predict(scores)
        else:
            # Softmax Fallback
            calibrated_probs = softmax(scores) # 簡易

        # Normalize to sum to 1.0 (Race-wise)
        # Note: This simple normalization assumes raw_df contains exactly one race or we loop.
        # However, raw_df contains MULTIPLE races.
        # Use pandas groupby transform to normalize per race_id.
        
        # Determine Race IDs for grouping
        # ids df has 'race_id'.
        
        # 結果結合
        result_df = ids.copy()
        result_df['score'] = scores
        
        # 1. Softmax (Group by Race)
        # scipy.special.softmax handles array, but we need group-wise
        from scipy.special import softmax
        result_df['prob'] = result_df.groupby('race_id')['score'].transform(lambda x: softmax(x))

        # 2. Calibration
        if self.calibrator:
            result_df['calibrated_prob'] = self.calibrator.predict(result_df['prob'].values)
        else:
            result_df['calibrated_prob'] = result_df['prob']

        # 3. Normalize per Race (Safe-guard)
        race_sums = result_df.groupby('race_id')['calibrated_prob'].transform('sum')
        result_df['calibrated_prob'] = result_df['calibrated_prob'] / race_sums
        
        # EV計算
        result_df['odds'] = result_df['odds'].replace(0, 1.0)
        result_df['expected_value'] = result_df['calibrated_prob'] * result_df['odds']

        # 通知ループ
        for race_meta in targets:
            race_id = race_meta['race_id']
            race_df = result_df[result_df['race_id'] == race_id].copy()
            if race_df.empty: continue
            
            venue_map = {
                '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
                '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
            }
            venue_code = race_meta['venue']
            race_meta_dict = {
                'race_id': race_id,
                'title': race_meta['title'],
                'race_number': race_meta['race_number'],
                'start_time': race_meta['start_time'][:2] + ":" + race_meta['start_time'][2:],
                'venue_name': venue_map.get(venue_code, 'Unknown'),
                'date': self.target_date if self.target_date else datetime.now().strftime('%Y/%m/%d')
            }
            
            logger.info(f"通知送信: {race_meta_dict['title']}")
            
            if not self.dry_run:
                success = self.notifier.send_prediction(race_meta_dict, race_df)
                if success:
                    self.notified_races.add(race_id)
                time.sleep(1.0) # Rate Limit回避
            else:
                logger.info("DRY-RUN: 通知をスキップしました。")
                print(race_df[['horse_name', 'score', 'calibrated_prob']].sort_values('score', ascending=False).head())

        self._save_state()

def main():
    parser = argparse.ArgumentParser(description='Automated Prediction & Notification')
    parser.add_argument('--dry-run', action='store_true', help='通知を送信せずに実行')
    parser.add_argument('--date', type=str, help='対象日付 (YYYYMMDD or YYYY-MM-DD)')
    args = parser.parse_args()
    
    # 日付正規化
    target_date = args.date
    if target_date and '-' not in target_date:
        # YYYYMMDD -> YYYY-MM-DD
        target_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"

    predictor = AutoPredictor(dry_run=args.dry_run, target_date=target_date)
    predictor.run()

if __name__ == "__main__":
    main()

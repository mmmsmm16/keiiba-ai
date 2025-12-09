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
from scipy.special import softmax

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


# .env 手動読み込み (Docker環境等で環境変数が反映されていない場合用)
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
            # logger.info(".env loaded manually.")
    except Exception as e:
        logger.warning(f".env reading failed: {e}")

# 定数
STATE_FILE_PATH = os.path.join(os.path.dirname(__file__), '../../data/state/notified_races.json')
# DISCORD_WEBHOOK_URL will be loaded dynamically

class NotificationManager:
    """Discord通知を管理するクラス"""
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_prediction(self, race_meta: Dict, df: pd.DataFrame):
        """
        予測結果をDiscordに送信します。
        
        Args:
            race_meta: レース情報 (race_id, title, venue, etc.)
            df: 予測結果DataFrame (馬番,馬名,EV,確率などを含む)
        """
        if not self.webhook_url:
            logger.warning("Discord Webhook URLが設定されていません。通知をスキップします。")
            return

        # タイトル整形
        date_str = race_meta.get('date', '')
        title_str = f"🎯 [{date_str}] {race_meta['venue_name']}{race_meta['race_number']}R {race_meta['title']} ({race_meta['start_time']})"

        # 予測テーブル作成 (Top 6: 勝率順)
        top_picks = df.sort_values('calibrated_prob', ascending=False).head(6)
        
        description = "**🏆 本命予測 (勝率上位)**\n"
        
        # ヘッダーなし、リスト形式で見やすく
        marks = ["◎", "〇", "▲", "△", "△", "△"]
        
        for i, (_, row) in enumerate(top_picks.iterrows()):
            mark = marks[i] if i < len(marks) else ""
            h_num = str(int(row['horse_number'])).zfill(2)
            h_name = row['horse_name']
            
            ev = f"{row['expected_value']:.2f}"
            prob = f"{row['calibrated_prob']*100:.0f}%"
            score = f"{row['score']:.2f}"
            
            # Simple list format with Score
            description += f"`{mark}` `{h_num}` **{h_name}** (勝率:{prob}, EV:{ev}, Sc:{score})\n"

        # 推奨買い目 (Smart Value Logic)
        bet_strategy = self._generate_betting_strategy(df)
        
        embed = {
            "title": title_str,
            "description": description + "\n" + bet_strategy,
            "color": 0xFF0000 if top_picks.iloc[0]['expected_value'] > 1.5 else 0x00FF00, # 高期待値なら赤
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
        # 厳密な計算は複雑なので、全角=2、半角=1として計算してスペースで埋める
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
        # 1. 勝率上位6頭を抽出 (安定群)
        top_prob_df = df.sort_values('calibrated_prob', ascending=False).head(6)
        
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
        # 優先順位: v4_2025 のみ (v1等は特徴量不整合でエラーになるためフォールバックしない)
        path = os.path.join(model_dir, 'ensemble_v4_2025.pkl')
        # if not os.path.exists(path): path = os.path.join(model_dir, 'ensemble_v1.pkl') # 削除
        # if not os.path.exists(path): path = os.path.join(model_dir, 'ensemble_model.pkl') # 削除
        
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
                from src.model.calibration import ProbabilityCalibrator
                calib = ProbabilityCalibrator()
                calib.load(path)
                return calib
            except:
                return None
        return None

    def run(self):
        """メイン実行ループ"""
        logger.info("自動予測プロセスを実行します...")
        
        # 1. 開催日/現在時刻の取得
        now = datetime.now()
        if self.target_date:
            today_str = self.target_date.replace('-', '')
            # シミュレーション時は時刻を任意に設定できないため、全レース対象にする等の工夫が必要だが、
            # ここでは「その日のレース全て」を「未通知なら」処理する動きになる。
            # ただし直前チェックもシミュレーションする場合は時刻モックが必要。
            # 今回は簡易的に「指定日なら全レースチェック」とする。
        else:
            today_str = now.strftime('%Y%m%d')

        # 2. レース一覧取得
        race_list_df = self.loader.load_race_list(today_str)
        if race_list_df.empty:
            logger.info("本日の開催レースはありません。")
            return

        # 3. 直前レースのフィルタリング (発走 15分〜30分前)
        targets = []
        
        for _, row in race_list_df.iterrows():
            race_id = row['race_id']
            if race_id in self.notified_races:
                continue
                
            start_time_str = row['start_time'] # HHMM format usually "1000"
            if not start_time_str: continue

            # 時刻パース
            try:
                # today_str (YYYYMMDD) + start_time_str (HHMM)
                race_dt = datetime.strptime(f"{today_str}{start_time_str}", "%Y%m%d%H%M")
            except ValueError:
                continue

            # ターゲット判定
            if self.target_date:
                # 指定日モードなら無条件に追加 (ドライラン用)
                targets.append(row)
            else:
                # リアルタイムモード
                diff = race_dt - now
                minutes = diff.total_seconds() / 60
                
                # 15分〜35分前くらいをターゲットにする
                if 15 <= minutes <= 35:
                     targets.append(row)
        
        if not targets:
            logger.info("現在、直前の通知対象レースはありません。")
            return
            
        logger.info(f"通知対象レース: {len(targets)} 件")

        # 4. 推論 & 通知
        # パフォーマンスのため、対象レース分まとめてロードするか、ループするか。
        # InferenceDataLoader.load は race_ids リストを受け取れるが、
        # リアルタイム特徴量のために「その日の全結果」も必要。
        # Loaderの仕様上、race_idsを指定しても内部で日付フィルタのみにして全件ロードするよう実装修正済みならOK。
        # 確認: loader.py:228 で「呼び出し元でフィルタリング」となっている。
        
        # データロード (日次で一括ロードしてメモリに乗せておくのが理想だが、ここでは毎回ロード)
        target_ids = [r['race_id'] for r in targets]
        
        # Loaderは「指定日付の全レース」をロードし、race_idsでフィルタしていない（Loader修正次第）。
        # 現状のLoaderは race_ids を渡すと SQL の WHERE IN に入れるが、
        # RealTimeFeatureのために「同日の終了したレース」が必要な場合、これでは不足する可能性がある。
        # -> Loader修正済み: race_idsから日付を取り出してその日の全レースを取得するようにしたか？
        # -> はい、loader.py の 226行目付近で実装されています。
        
        try:
            raw_df = self.loader.load(target_date=today_str, race_ids=target_ids)
        except Exception as e:
            logger.error(f"データロードエラー: {e}")
            return

        if raw_df.empty:
            return

        # 前処理
        try:
            X, ids = self.preprocessor.preprocess(raw_df)
            processed_df = pd.concat([ids, X], axis=1)
            # 重複カラムを削除 (idsとXで重複がある場合、ids側=左側を優先して残す)
            processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
        except Exception as e:
            logger.error(f"前処理エラー: {e}")
            return
            
        # 予測
        # feature columns alignment
        feature_cols = None
        # ... predict.pyと同様の特徴量名解決ロジック ...
        # 簡易化のため、モデルが feature_name() を持っていると仮定
        # EnsembleModelの場合、内部のモデル(LGBM)から特徴量名を取得する
        if isinstance(self.model, EnsembleModel):
            bst = self.model.lgbm.model
        else:
            bst = self.model.model

        logger.info(f"DEBUG: bst type: {type(bst)}")
        logger.info(f"DEBUG: dir(bst): {dir(bst)[:20]}...") # show first 20 attrs


        if hasattr(bst, 'feature_name'):
             feature_cols = bst.feature_name()
             logger.info(f"DEBUG: bst.feature_name() found. Len: {len(feature_cols)}")
        elif hasattr(bst, 'booster_'):
             feature_cols = bst.booster_.feature_name()
             logger.info(f"DEBUG: bst.booster_.feature_name() found. Len: {len(feature_cols)}")
        
        if feature_cols:
             logger.info(f"DEBUG: Using {len(feature_cols)} features for prediction.")
             # Add missing as 0
             missing = set(feature_cols) - set(processed_df.columns)
             if missing:
                 logger.info(f"DEBUG: Missing columns: {missing}")
                 for c in missing: processed_df[c] = 0
             
             X_pred = processed_df[feature_cols]
        else:
             logger.warning("DEBUG: Feature names NOT found in model. Using all numeric columns.")
             X_pred = processed_df.select_dtypes(include=[np.number])
        
        # Check for duplicates
        if X_pred.columns.duplicated().any():
            logger.warning(f"DEBUG: X_pred has duplicated columns: {X_pred.columns[X_pred.columns.duplicated()].tolist()}")
            X_pred = X_pred.loc[:, ~X_pred.columns.duplicated()]
            
        logger.info(f"DEBUG: X_pred shape checks - Shape: {X_pred.shape}")

             
        try:
            scores = self.model.predict(X_pred)
            processed_df['score'] = scores
            
            # Softmax
            processed_df['prob'] = processed_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
            
            # Calibrate
            if self.calibrator:
                processed_df['calibrated_prob'] = self.calibrator.predict(processed_df['prob'].values)
            else:
                processed_df['calibrated_prob'] = processed_df['prob']
                
            # EV
            if 'odds' in processed_df.columns:
                processed_df['expected_value'] = processed_df['calibrated_prob'] * processed_df['odds'].fillna(0)
            else:
                processed_df['expected_value'] = 0
                
        except Exception as e:
            logger.error(f"予測実行エラー: {e}")
            return

        # 5. 各レースごとに通知
        for race_meta in targets:
            race_id = race_meta['race_id']
            
            # このレースの馬データを抽出
            race_df = processed_df[processed_df['race_id'] == race_id].copy()
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
                time.sleep(1.0) # Rate Limit回避 (1秒待機)
            else:
                logger.info("DRY-RUN: 通知をスキップしました。")
                print(race_df[['horse_name', 'expected_value']].sort_values('expected_value', ascending=False).head())

        # 完了後に状態保存
        self._save_state()


def main():
    parser = argparse.ArgumentParser(description='Automated Prediction & Notification')
    parser.add_argument('--dry-run', action='store_true', help='通知を送信せずに実行')
    parser.add_argument('--date', type=str, help='対象日付 (YYYY-MM-DD)')
    args = parser.parse_args()
    
    predictor = AutoPredictor(dry_run=args.dry_run, target_date=args.date)
    predictor.run()

if __name__ == "__main__":
    main()

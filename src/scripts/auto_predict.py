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
from src.inference.optimal_strategy import OptimalStrategy

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
        
        for i, (_, row) in enumerate(top_picks.iterrows()):
            h_num = str(int(row['horse_number'])).zfill(2)
            h_name = row['horse_name']
            
            ev = f"{row['expected_value']:.2f}"
            prob = f"{row['calibrated_prob']*100:.0f}%"
            score = f"{row['score']:.2f}"
            
            # Simple list format without Mark
            description += f"`{h_num}` **{h_name}** (勝率:{prob}, EV:{ev}, Sc:{score})\n"

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

    def _calculate_betting_data(self, df: pd.DataFrame) -> dict:
        """推奨買い目のデータを計算して返す (シミュレーション用)"""
        # スコア順にソート (基本データとして使用)
        sorted_df = df.sort_values('score', ascending=False)
        top1 = sorted_df.iloc[0]
        
        # 基本情報
        top1_ev = top1.get('expected_value', 0)
        h_num = int(top1['horse_number'])
        h_str = f"{h_num:02}"
        
        # 相手馬 (Rank 2-6)
        opps = sorted_df.iloc[1:6] 
        opp_nums = [f"{int(x):02}" for x in opps['horse_number']]
        
        strategy_data = {
            "top1": top1,
            "sorted_df": sorted_df,
            "ev": top1_ev,
            "bets": [],
            "strategy_type": "Low",
            "is_strong": False
        }
        
        # 戦略判定
        if top1_ev >= 1.2:
            # High Value
            strategy_data["strategy_type"] = "High"
            # 三連複 1頭軸流し (Rank 2,3,4)
            strategy_data["bets"].append({
                "type": "sanrenpuku",
                "axis": [h_num],
                "partners": [int(x) for x in opps.iloc[:3]['horse_number']],
                "points": 3
            })
            
        elif top1_ev >= 0.8:
            # Mid Value
            strategy_data["strategy_type"] = "Mid"
            # 三連単 1着固定流し (Rank 2,3,4)
            strategy_data["bets"].append({
                "type": "sanrentan_1fix",
                "axis": [h_num],
                "partners": [int(x) for x in opps.iloc[:3]['horse_number']],
                "points": 6
            })
            # (参考) 馬連 1頭軸流し (Rank 2,3,4,5)
            # strategy_data["bets"].append({
            #     "type": "umaren",
            #     "axis": [h_num],
            #     "partners": [int(x) for x in opps.iloc[:4]['horse_number']],
            #     "points": 4
            # })
            
        else:
            # Low Value (見送り)
            strategy_data["strategy_type"] = "Low"
        
        # 強気馬券判定 (7番人気以上)
        axis_pop = int(top1['popularity']) if pd.notna(top1['popularity']) else 99
        if axis_pop >= 7:
            strategy_data["is_strong"] = True
            # 三連単 1着固定流し (Rank 2,3,4,5) -> Opps has 5 horses (Rank 2-6). 
            # Original code said: {','.join(opp_nums[:4])} which is Rank 2,3,4,5.
            strategy_data["bets"].append({
                "type": "sanrentan_1fix_strong",
                "axis": [h_num],
                "partners": [int(x) for x in opps.iloc[:4]['horse_number']],
                "points": 12
            })
            
        return strategy_data

    def _generate_betting_strategy(self, df: pd.DataFrame) -> str:
        """推奨買い目のテキスト生成 (v12 最適戦略)"""
        data = self._calculate_betting_data(df)
        sorted_df = data["sorted_df"]
        
        # --- 1. AI本命予想リスト ---
        msg = "**🤖 AI本命予想 (Ranked v12)**\n"
        symbols = ['◎', '〇', '▲', '△', '△', '△', '注']
        
        # 上位7頭を表示
        for i, (idx, row) in enumerate(sorted_df.head(7).iterrows()):
            h_num = str(int(row['horse_number'])).zfill(2)
            ev = row.get('expected_value', 0)
            score = row.get('score', 0)
            pop = int(row['popularity']) if pd.notna(row['popularity']) else 99
            short_name = str(row['horse_name'])[:5]
            symbol = symbols[i] if i < len(symbols) else '  '
            msg += f"`{symbol}{h_num} {short_name:<5}({pop}人) S{score:.2f} E{ev:.2f}`\n"
            
        msg += "\n"
        
        # --- 2. 推奨買い目 (v12 Logic) ---
        msg += "**📈 推奨買い目 (v12戦略)**\n"
        
        top1 = data["top1"]
        h_str = f"{int(top1['horse_number']):02}"
        # Rank 2-6 IDs for display
        opps = sorted_df.iloc[1:6]
        opp_nums = [f"{int(x):02}" for x in opps['horse_number']]
        
        if data["strategy_type"] == "High":
            msg += f"🔥 **High Value (EV {data['ev']:.2f})** - 鉄板/高妙味\n"
            msg += f"✅ **推奨: 三連複 1頭軸流し (3点)**\n"
            msg += f"`{h_str} - {','.join(opp_nums[:3])}` (相手: 2,3,4位)\n"
            msg += "※期待値が高いため、三連複3点で高回収(142%)を狙います。\n"
            
        elif data["strategy_type"] == "Mid":
            msg += f"✨ **Mid Value (EV {data['ev']:.2f})** - 中妙味\n"
            msg += f"✅ **推奨: 三連単 1着固定流し (6点)**\n"
            msg += f"`{h_str} -> {','.join(opp_nums[:3])}` (相手: 2,3,4位)\n"
            msg += f"💡 (安定) 馬連 1頭軸流し (4点): `{h_str} - {','.join(opp_nums[:4])}`\n"
            
        else:
            msg += f"⚠️ **Low Value (EV {data['ev']:.2f})** - 低妙味 (見送り推奨)\n"
            msg += "過剰人気のため、期待値が低いです。基本はケン(見送り)してください。\n"
            msg += f"(参考) 三連単 1着固定流し (6点): `{h_str} -> {','.join(opp_nums[:3])}`\n"
        
        # --- 3. 強気馬券 ---
        if data["is_strong"]:
            axis_pop = int(top1['popularity']) if pd.notna(top1['popularity']) else 99
            msg += "\n"
            msg += f"🔥 **強気馬券** (Top1が{axis_pop}番人気)\n"
            msg += f"✅ **三連単 1着固定流し: {h_str}→{','.join(opp_nums[:4])}** (12点)\n"
            msg += "※穴狙いで高配当を狙える条件です。\n"
        
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
        
        # v12特徴量リストのロード
        self.expected_features = self._load_feature_list()
        
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
        logger.info("モデル(Ensemble v12: TabNet Revival)をロード中...")
        model = EnsembleModel()
        # v12モデルパス (experiments配下)
        model_path = os.path.join(os.path.dirname(__file__), '../../experiments/v12_tabnet_revival/models/ensemble.pkl')
        
        if os.path.exists(model_path):
            model.load_model(model_path, device_name='cpu') # 推論はCPUで安全に
            return model
        else:
            logger.error(f"モデルファイルが見つかりません: {model_path}")
            # フォールバック (models/ensemble_v7.pkl)
            fallback_path = os.path.join(os.path.dirname(__file__), '../../models/ensemble_v7.pkl')
            if os.path.exists(fallback_path):
                 logger.warning("フォールバックモデル(v7)を使用します。")
                 model.load_model(fallback_path)
                 return model
            return None

    def _load_feature_list(self):
        """v12モデルの特徴量リストをロード (フォールバック: LightGBMモデルから取得)"""
        import json
        features_path = os.path.join(os.path.dirname(__file__), 
            '../../experiments/v12_tabnet_revival/models/tabnet.features.json')
        if os.path.exists(features_path):
            try:
                with open(features_path, 'r', encoding='utf-8') as f:
                    features = json.load(f)
                logger.info(f"v12特徴量リストをロード (JSON): {len(features)}個")
                return features
            except Exception as e:
                logger.warning(f"特徴量JSONのロード失敗: {e}. LightGBMからフォールバックします。")
        
        # Fallback: LightGBM model's feature_name()
        if self.model and hasattr(self.model, 'lgbm') and self.model.lgbm:
            try:
                lgbm_booster = self.model.lgbm.model  # lightgbm.Booster
                if hasattr(lgbm_booster, 'feature_name'):
                    features = lgbm_booster.feature_name()
                    logger.info(f"v12特徴量リストをロード (LightGBM): {len(features)}個")
                    return features
            except Exception as e:
                logger.warning(f"LightGBMからの特徴量リスト取得失敗: {e}")
        
        logger.warning("特徴量リストが見つかりませんでした。特徴量適合なしで推論します。")
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
            
            # コロンを除去 ("09:45" → "0945")
            start_time_str = str(start_time_str).replace(':', '')

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
        
        # 特徴量アダプテーション (v12モデル用)
        if self.expected_features:
            missing = set(self.expected_features) - set(X.columns)
            if missing:
                logger.warning(f"不足特徴量を0で補完: {len(missing)}個")
                for col in missing:
                    X[col] = 0.0
            # 特徴量の順序を揃える
            X = X[[c for c in self.expected_features if c in X.columns]]
        
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
        
        # 1. Softmax (Group by Race) with Temperature to avoid extreme probabilities
        # Temperature > 1.0 makes distribution more uniform
        from scipy.special import softmax
        SOFTMAX_TEMPERATURE = 3.0  # スコア差が極端な場合の緩和用
        result_df['prob'] = result_df.groupby('race_id')['score'].transform(
            lambda x: softmax(x / SOFTMAX_TEMPERATURE)
        )

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

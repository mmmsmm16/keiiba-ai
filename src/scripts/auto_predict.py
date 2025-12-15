# ... (Imports remain mostly the same, ensuring all needed are present)
import os
import sys
import json
import time
import requests
import argparse
import pandas as pd
import numpy as np
import logging
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from scipy.special import softmax
from scipy.stats import entropy
from itertools import combinations, permutations

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
RULES_PATH = os.path.join(os.path.dirname(__file__), '../../experiments/v23_regression_cv/final_rules_v23.json')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../experiments/v23_regression_cv/fold4')

class NotificationManager:
    """Discord通知を管理するクラス"""
    def __init__(self, webhook_url: str, rules: list):
        self.webhook_url = webhook_url
        self.rules = rules

    def _calculate_confidence(self, df: pd.DataFrame) -> tuple[str, str]:
        """ハザード(波乱度)と自信度を判定"""
        probs = df['prob'].values 
        ent = entropy(probs)
        
        top_horse = df.sort_values('score', ascending=False).iloc[0]
        # v23 score is regression (race normalized?) No, raw score.
        # But we calculate prob using softmax.
        top_prob = top_horse['prob']
        
        if top_prob >= 0.40:
            return "S", "鉄板 (Ironclad)"
        elif top_prob >= 0.25:
            return "A", "安定 (Stable)"
        elif ent > 2.0 or top_prob < 0.20:
             return "C", "波乱 (High Chaos)"
        else:
             return "B", "混戦 (Confusion)"

    def send_prediction(self, race_meta: Dict, df: pd.DataFrame, race_features: Dict):
        """予測結果をDiscordに送信"""
        if not self.webhook_url:
            return False

        date_str = race_meta.get('date', '')
        chart_rank, chart_desc = self._calculate_confidence(df)
        
        title_str = f"🎯 [{date_str}] {race_meta['venue_name']}{race_meta['race_number']}R {race_meta['title']} ({race_meta['start_time']}) - [{chart_rank}] {chart_desc}"

        # 1. 予測テーブル
        top_picks = df.sort_values('score', ascending=False)
        description = "**🏆 本命予測 (v23 Model)**\n"
        
        for i, (_, row) in enumerate(top_picks.iterrows()):
            h_num = str(int(row['horse_number'])).zfill(2)
            h_name = row['horse_name']
            
            # v23は回帰スコアなのでそのまま表示
            score = f"{row['score']:.2f}"
            prob_val = row.get('prob', 0)
            prob = f"{prob_val*100:.0f}%"
            
            odds_str = f"{row['odds']:.1f}" if row['odds'] > 0 else "-"
            
            description += f"`{h_num}` **{h_name}** (Odds:{odds_str}, Sc:{score}, Win%:{prob})\n"

        # 2. 推奨買い目
        bet_msg = self._generate_betting_strategy(df, race_features)
        
        # NetKeiba Link
        netkeiba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_meta['race_id']}"
        description += f"\n🔗 [NetKeiba]({netkeiba_url})\n"
        
        embed = {
            "title": title_str,
            "description": description + "\n" + bet_msg,
            "color": 0x00FF00 if chart_rank in ['S', 'A'] else 0xFFA500,
            "footer": {
                "text": "Keiiba-AI v23 (Auto-Optimized)"
            }
        }
        
        payload = {"username": "ナミール (v23)", "embeds": [embed]}
        
        try:
            resp = requests.post(self.webhook_url, json=payload)
            resp.raise_for_status()
            logger.info(f"通知送信成功: {race_meta['race_id']}")
            return True
        except Exception as e:
            logger.error(f"通知送信失敗: {e}")
            return False

    def _generate_betting_strategy(self, df: pd.DataFrame, features: Dict) -> str:
        """ルールベースで推奨買い目を生成"""
        # ルール適用
        valid_bets = []
        
        # features は race_level の特徴量 (score_gap, etc)
        # ルール条件チェック
        for rule in self.rules:
            match = True
            for feat, op, thres in rule['conditions']:
                val = features.get(feat, 0)
                if op == '<=':
                    if not (val <= thres):
                        match = False; break
                else:
                    if not (val > thres):
                        match = False; break
            if match:
                valid_bets.append(rule)
        
        if not valid_bets:
            return "⚠️ **推奨買い目なし (条件不一致)**\n様子見推奨です。\n"

        msg = "**📈 推奨買い目 (AI Optimized Rules)**\n"
        
        # Betting Logic (Generate codes)
        top_horses = df.sort_values('score', ascending=False)['horse_number'].astype(int).tolist()
        
        # 重複除外して表示
        shown_bets = set()
        
        # ルールごとの表示
        # 優先度順に並べ替えたいが、JSON順序(ROI高い順)と仮定
        for rule in valid_bets:
            bname = rule['bet_name']
            if bname in shown_bets: continue
            
            # ROIなどの補足情報
            roi = rule.get('roi', 0) * 100
            msg += f"✅ **{bname}** (期待ROI {roi:.0f}%)\n"
            
            # 実際の買い目構築 (簡易)
            codes_str = self._format_bet_codes(bname, top_horses)
            if codes_str:
                msg += f"`{codes_str}`\n"
            
            shown_bets.add(bname)
            
        return msg

    def _format_bet_codes(self, bname, top_horses):
        """買い目の文字列表現を生成"""
        try:
            if 'tansho' in bname:
                return f"単勝: {top_horses[0]:02}"
            elif 'umaren_box' in bname:
                n = int(bname[-1])
                return f"馬連BOX: {','.join([f'{x:02}' for x in top_horses[:n]])}"
            elif 'umaren_nagashi' in bname:
                return f"馬連流し: {top_horses[0]:02} - {','.join([f'{x:02}' for x in top_horses[1:5]])}"
            elif 'wide_box' in bname:
                n = int(bname[-1])
                return f"ワイドBOX: {','.join([f'{x:02}' for x in top_horses[:n]])}"
            elif 'wide_nagashi' in bname:
                 return f"ワイド流し: {top_horses[0]:02} - {','.join([f'{x:02}' for x in top_horses[1:5]])}"
            elif 'umatan_1st' in bname:
                return f"馬単1着固定: {top_horses[0]:02} -> {','.join([f'{x:02}' for x in top_horses[1:5]])}"
            elif 'umatan_box' in bname:
                n = int(bname[-1])
                return f"馬単BOX: {','.join([f'{x:02}' for x in top_horses[:n]])}"
            elif 'sanrenpuku_box' in bname:
                n = int(bname[-1])
                return f"三連複BOX: {','.join([f'{x:02}' for x in top_horses[:n]])}"
            elif 'sanrenpuku_nagashi' in bname:
                return f"三連複流し: {top_horses[0]:02} - {','.join([f'{x:02}' for x in top_horses[1:5]])} (2頭)"
            elif 'sanrentan_1st' in bname:
                return f"三連単1着固定: {top_horses[0]:02} -> {','.join([f'{x:02}' for x in top_horses[1:5]])} (M)"
            elif 'sanrentan_box' in bname:
                n = int(bname[-1])
                return f"三連単BOX: {','.join([f'{x:02}' for x in top_horses[:n]])}"
            return bname
        except:
            return f"Error formatting {bname}"

class AutoPredictor:
    def __init__(self, dry_run: bool = False, target_date: str = None):
        self.dry_run = dry_run
        self.target_date = target_date
        self.state_file = STATE_FILE_PATH
        self.notified_races = self._load_state()
        
        # モデル初期化
        self.loader = InferenceDataLoader()
        self.preprocessor = InferencePreprocessor()
        self.lgbm, self.catboost, self.meta = self._load_v23_models()
        self.rules = self._load_rules()
        
        # Load env vars manually to ensure Webhook URL is present
        load_env_manual()
        webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
        if not webhook_url:
            logger.error("❌ DISCORD_WEBHOOK_URL is not set.")
            
        self.notifier = NotificationManager(webhook_url, self.rules)
        
    def _load_state(self) -> set:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f: return set(json.load(f))
            except: return set()
        return set()

    def _save_state(self):
        if self.dry_run: return
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f: json.dump(list(self.notified_races), f)

    def _load_v23_models(self):
        logger.info(f"Loading v23 models from {MODEL_DIR}...")
        try:
            with open(os.path.join(MODEL_DIR, 'lgbm_v23.pkl'), 'rb') as f: lgbm = pickle.load(f)
            with open(os.path.join(MODEL_DIR, 'catboost_v23.pkl'), 'rb') as f: catboost = pickle.load(f)
            with open(os.path.join(MODEL_DIR, 'meta_v23.pkl'), 'rb') as f: meta = pickle.load(f)
            return lgbm, catboost, meta
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            sys.exit(1)

    def _load_rules(self):
        if not os.path.exists(RULES_PATH):
            logger.warning("Rules file not found.")
            return []
        with open(RULES_PATH, 'r') as f: return json.load(f)

    def run(self):
        logger.info("AutoPredict v23 Process Start...")
        
        now = datetime.now()
        today_str = self.target_date.replace('-', '') if self.target_date else now.strftime('%Y%m%d')

        # 1. レース一覧
        try:
            race_list_df = self.loader.load_race_list(today_str)
        except Exception as e:
            logger.error(f"Race list load failed: {e}")
            return

        # Prepare venue_name
        venue_map = {
            '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
            '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
        }
        race_list_df['venue_name'] = race_list_df['venue'].map(venue_map).fillna(race_list_df['venue'])

        if race_list_df.empty:
            logger.info("No races today.")
            return

        # 2. フィルタリング (15-35分前)
        targets = []
        for _, row in race_list_df.iterrows():
            race_id = row['race_id']
            if race_id in self.notified_races: continue
                
            start_time_str = str(row['start_time']).replace(':', '')
            try:
                race_dt = datetime.strptime(f"{today_str}{start_time_str}", "%Y%m%d%H%M")
            except: continue

            if self.target_date:
                targets.append(row)
            else:
                diff = race_dt - now
                minutes = diff.total_seconds() / 60
                if 15 <= minutes <= 35:
                     targets.append(row)
        
        if not targets:
            logger.info("No target races for notification.")
            return

        logger.info(f"Targets: {len(targets)} races")

        # 3. データロード・推論
        target_ids = [r['race_id'] for r in targets]
        raw_df = self.loader.load(target_date=today_str, race_ids=target_ids)
        if raw_df.empty: return

        # 前処理
        X, ids = self.preprocessor.preprocess(raw_df)
        
        # 特徴量補完 (v23モデル用)
        # pickleなどから特徴量リストを取得するのが正道だが、簡易的にLGBMから取得
        expected_cols = self.lgbm.feature_name()
        
        # カラム合わせ
        for col in expected_cols:
            if col not in X.columns: X[col] = 0.0
        X = X[expected_cols]

        # 推論 (Ensemble)
        try:
            p1 = self.lgbm.predict(X)
            p2 = self.catboost.predict(X)
            scores = self.meta.predict(np.column_stack([p1, p2]))
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return

        # 結果結合
        result_df = ids.copy()
        result_df['score'] = scores
        # Softmax for Prob
        result_df['prob'] = result_df.groupby('race_id')['score'].transform(lambda x: softmax(x))

        # 通知ループ
        for race_meta in targets:
            race_id = race_meta['race_id']
            race_df = result_df[result_df['race_id'] == race_id].copy()
            if race_df.empty: continue
            
            # 特徴量抽出 (for Betting Rules)
            # ここでルール判定用の特徴量(score_gap, etc)を計算
            race_feats = self._calc_race_features(race_df, race_meta, today_str)

            # 通知
            if not self.dry_run:
                success = self.notifier.send_prediction(race_meta, race_df, race_feats)
                if success: self.notified_races.add(race_id)
                time.sleep(1.0)
            else:
                logger.info(f"[DRY-RUN] {race_meta['title']}")
                print(race_df.sort_values('score', ascending=False).head())
                print("Features:", race_feats)
                print(self.notifier._generate_betting_strategy(race_df, race_feats))

        self._save_state()

    def _calc_race_features(self, df, meta, date_str):
        """ルール適用に必要な特徴量を計算"""
        sorted_df = df.sort_values('score', ascending=False)
        top_scores = sorted_df['score'].tolist()
        top_odds = sorted_df['odds'].head(3).tolist()
        
        score_gap = top_scores[0] - top_scores[1] if len(top_scores) > 1 else 0
        score_conc = sum(top_scores[:3]) / df['score'].sum() if df['score'].sum() > 0 else 0
        avg_top3 = np.mean(top_odds) if top_odds else 0
        
        venue_code = int(str(meta['race_id'])[4:6])
        
        # surface判定: proc_df(df)にsurfaceがあれば使う
        surf = 0
        if 'surface' in df.columns:
            try: surf = int(df['surface'].iloc[0]) - 1 
            except: pass
            if surf < 0: surf = 0

        # distance
        dist = 1600
        if 'distance' in df.columns:
             dist = float(df['distance'].iloc[0])
        elif 'distance' in meta:
             dist = float(meta['distance'])
             
        return {
            'score_gap': score_gap,
            'top1_odds': top_odds[0] if top_odds else 0,
            'avg_top3_odds': avg_top3,
            'score_conc': score_conc,
            'n_horses': len(df),
            'distance': dist,
            'surface': surf,
            'venue': venue_code - 1,
            'month': datetime.strptime(date_str, '%Y%m%d').month
        }

def main():
    parser = argparse.ArgumentParser(description='Automated Prediction & Notification (v23)')
    parser.add_argument('--dry-run', action='store_true', help='通知を送信せずに実行')
    parser.add_argument('--date', type=str, help='対象日付 (YYYYMMDD or YYYY-MM-DD)')
    args = parser.parse_args()
    
    target_date = args.date
    if target_date and '-' in target_date:
        # YYYY-MM-DD -> YYYYMMDD (load_race_list expects YYYYMMDD?)
        # Actually logic inside run() handles replacement.
        # But let's standardize to YYYY-MM-DD for consistency
        pass

    predictor = AutoPredictor(dry_run=args.dry_run, target_date=target_date)
    predictor.run()

if __name__ == "__main__":
    main()

"""
Auto Predict No Odds (Ranker Only) Script
Ranker Onlyモデル (LambdaRank, v18) を用いて日次予測を行い、Discord通知を送信する。
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List, Optional

# プロジェクトルートを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.category_aggregators import CategoryAggregator
from src.preprocessing.incremental_aggregators import IncrementalCategoryAggregator
from src.preprocessing.aggregators import HistoryAggregator
from src.preprocessing.advanced_features import AdvancedFeatureEngineer
from src.preprocessing.experience_features import ExperienceFeatureEngineer
from src.preprocessing.relative_features import RelativeFeatureEngineer
from src.preprocessing.opposition_features import OppositionFeatureEngineer
from src.preprocessing.rating_features import RatingFeatureEngineer
from src.utils.discord import NotificationManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 定数
RANKER_MODEL_PATH = "models/eval/ranker_eval_v19.pkl"
CACHE_PATH = "data/cache/jra_base/advanced.parquet"
REPORT_DIR = "reports/jra/daily"

# 場コード
VENUE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

class RankerOnlyPredictor:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading Ranker model from {model_path}...")
        payload = joblib.load(model_path)
        
        self.model = payload['model']
        self.feature_cols = payload['feature_cols']
        
        # コンポーネントの初期化
        self.loader = JraVanDataLoader()
        self.cleanser = DataCleanser()
        self.engineer = FeatureEngineer()
        self.hist_agg = HistoryAggregator()
        self.adv_eng = AdvancedFeatureEngineer()
        self.exp_eng = ExperienceFeatureEngineer()
        self.rel_eng = RelativeFeatureEngineer()
        self.opp_eng = OppositionFeatureEngineer()
        self.rating_eng = RatingFeatureEngineer()
        
    def predict_date(self, target_date: str):
        logger.info(f"=== Predicting for {target_date} ===")
        
        # 1. データのロード (当日 + 直近)
        # 過去データ（特徴量集計用）と当日データをロード
        df = self.loader.load(history_start_date=target_date, end_date=target_date, jra_only=True)
        if len(df) == 0:
            logger.warning(f"No race data found for {target_date}")
            return None
        
        # 2. 前処理 (Backtestと同一のロジック)
        df = self.cleanser.cleanse(df)
        df = self.engineer.add_features(df)
        
        # カテゴリ集計 (Incrementalモードで実行)
        logger.info("Loading master data for aggregation...")
        try:
            master_df = pd.read_parquet(CACHE_PATH)
        except Exception as e:
            logger.error(f"Failed to load cache from {CACHE_PATH}: {e}")
            return None

        master_df['date'] = pd.to_datetime(master_df['date'])
        
        inc_cat_agg = IncrementalCategoryAggregator()
        # 履歴から状態を初期化 (ターゲット日前日まで)
        inc_cat_agg.fit(master_df[master_df['date'] < target_date])
        df = inc_cat_agg.transform_update(df)
        
        # その他特徴量 (直近2年分程度をコンテキストとして結合して計算)
        ctx_date = (pd.to_datetime(target_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        context_df = master_df[(master_df['date'] >= ctx_date) & (master_df['date'] < target_date)].copy()
        
        # ターゲット日付のデータを結合して特徴量生成
        proc_df = pd.concat([context_df, df], ignore_index=True)
        proc_df = proc_df.sort_values(['date', 'race_id'])
        
        # ヒストリ・高度特徴量等の生成
        proc_df = self.hist_agg.aggregate(proc_df)
        proc_df = self.adv_eng.add_features(proc_df)
        proc_df = self.exp_eng.add_features(proc_df)
        proc_df = self.rel_eng.add_features(proc_df)
        proc_df = self.opp_eng.add_features(proc_df)
        proc_df = self.rating_eng.add_features(proc_df)
        
        # 当日分の抽出
        test_df = proc_df[proc_df['date'] == target_date].copy()
        
        if len(test_df) == 0:
            logger.warning("No data rows remain after preprocessing.")
            return None

        # 3. 予測
        X = test_df[self.feature_cols].copy()
        
        # 重複カラム名の解消
        if X.columns.duplicated().any():
            logger.warning("Duplicate columns found, keeping first occurrences.")
            X = X.loc[:, ~X.columns.duplicated()]
            
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
        
        # Rankerスコアの予測
        test_df['score'] = self.model.predict(X)
        
        # 正規化と相対確率の計算 (RelProb for Visualization)
        def calc_rel_prob(group):
            s = group['score'].values
            # Softmax with temperature scaling for better contrast
            # Ranker scores are unbounded, usually small range. scaling helps.
            # Using standard scaling before softmax can stabilize.
            if len(s) > 1 and s.std() > 0:
                z = (s - s.mean()) / s.std()
            else:
                z = s - s.mean()
            
            # Use a robust temperature based on experience
            exp_s = np.exp(z * 1.5) 
            group['rel_prob'] = exp_s / exp_s.sum()
            return group
        
        test_df = test_df.groupby('race_id', group_keys=False).apply(calc_rel_prob)
        
        # 予測順位の付与
        test_df['pred_rank'] = test_df.groupby('race_id')['score'].rank(ascending=False, method='first').astype(int)
        
        return test_df

    def format_report(self, df: pd.DataFrame, target_date: str) -> str:
        lines = [f"# JRA 予測レポート ({target_date})", "## Ranker Only (v18) モデル予測", ""]
        lines.append("> [!NOTE]")
        lines.append("> ROI 88.9% (2025年実績) を記録した LambdaRank モデルによる予測です。")
        lines.append("> `RelScore` はレース内での相対的な勝利期待度（偏差値ベースのSoftmax）を表します。")
        lines.append("")
        
        race_info = df[['race_id', 'venue', 'date']].drop_duplicates()
        race_info['race_num'] = race_info['race_id'].str[-2:].astype(int)
        sorted_races = race_info.sort_values(['race_num', 'venue'])['race_id'].tolist()
        
        for race_id in sorted_races:
            r_df = df[df['race_id'] == race_id].sort_values('pred_rank')
            v_code = r_df['venue'].iloc[0] if 'venue' in r_df.columns else "Unknown"
            venue_name = VENUE_MAP.get(v_code, v_code)
            race_num = int(race_id[-2:])
            race_name = r_df['title'].iloc[0] if 'title' in r_df.columns else ""
            
            lines.append(f"### {venue_name} {race_num}R {race_name}")
            lines.append("| 予測順位 | 馬番 | 馬名 | スコア | RelScore | 騎手 |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            
            for _, row in r_df.iterrows():
                h_name = row['horse_name'] if 'horse_name' in row else f"Horse {row['horse_number']}"
                j_id = row['jockey_id'] if 'jockey_id' in row else ""
                lines.append(f"| {row['pred_rank']} | {row['horse_number']} | {h_name} | {row['score']:.4f} | **{row['rel_prob']:.1%}** | {j_id} |")
            lines.append("")
            
        return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--notify", action="store_true", help="Send Discord notification")
    args = parser.parse_args()

    try:
        predictor = RankerOnlyPredictor(RANKER_MODEL_PATH)
        results = predictor.predict_date(args.date)
        
        if results is not None and not results.empty:
            report = predictor.format_report(results, args.date)
            
            # 保存
            os.makedirs(REPORT_DIR, exist_ok=True)
            report_path = os.path.join(REPORT_DIR, f"{args.date}_ranker_only.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {report_path}")
            
            # 通知
            if args.notify:
                webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
                if webhook_url:
                    nm = NotificationManager(webhook_url)
                    nm.send_text(f"【JRA Ranker予測】{args.date}\nレポートを作成しました。")
                    
                    # サマリー
                    summary = "Top Picks (Ranker Only):\n"
                    for race_id, r_df in results.groupby('race_id'):
                        top1 = r_df[r_df['pred_rank'] == 1].iloc[0]
                        v_code = top1['venue'] if 'venue' in top1 else "??"
                        v_name = VENUE_MAP.get(v_code, v_code)
                        r_num = int(race_id[-2:])
                        summary += f"- {v_name}{r_num}R: {top1['horse_name']} (RS: {top1['rel_prob']:.1%})\n"
                    nm.send_text(summary)
        else:
            logger.warning("No predictions generated.")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

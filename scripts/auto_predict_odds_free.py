"""
Auto Predict Odds-Free Script
二値分類（オッズなし）モデルを用いて日次予測を行い、Discord通知を送信する。
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
BINARY_MODEL_PATH = "models/binary_no_odds.pkl"
RANKER_MODEL_PATH = "models/ranker_no_odds.pkl"
CACHE_PATH = "data/cache/jra_base/advanced.parquet"
REPORT_DIR = "reports/jra/daily"

# 場コード
VENUE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

class OddsFreePredictor:
    def __init__(self, binary_path: str, ranker_path: str):
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Binary model file not found: {binary_path}")
        if not os.path.exists(ranker_path):
            raise FileNotFoundError(f"Ranker model file not found: {ranker_path}")
        
        logger.info(f"Loading models...")
        bin_data = joblib.load(binary_path)
        rnk_data = joblib.load(ranker_path)
        
        self.model_bin = bin_data['model']
        self.model_rnk = rnk_data['model']
        self.feature_cols = bin_data['feature_cols']
        
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
        # 本来は caching されたデータを使うが、ここではシンプルに当日分を取得
        df = self.loader.load(history_start_date=target_date, end_date=target_date, jra_only=True)
        if len(df) == 0:
            logger.warning(f"No race data found for {target_date}")
            return
        
        # 2. 前処理 (Backtestと同一のロジック)
        df = self.cleanser.cleanse(df)
        df = self.engineer.add_features(df)
        
        # カテゴリ集計 (Incrementalモードで実行)
        # 注意: 運用環境では master_df が巨大なため、キャッシュを活用する必要がある
        # ここでは簡易化のため、CACHE_PATH から master_df を読み込む
        logger.info("Loading master data for aggregation...")
        master_df = pd.read_parquet(CACHE_PATH)
        master_df['date'] = pd.to_datetime(master_df['date'])
        
        inc_cat_agg = IncrementalCategoryAggregator()
        # 履歴から状態を初期化
        inc_cat_agg.fit(master_df[master_df['date'] < target_date])
        df = inc_cat_agg.transform_update(df)
        
        # その他特徴量
        ctx_date = (pd.to_datetime(target_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        context_df = master_df[(master_df['date'] >= ctx_date) & (master_df['date'] < target_date)].copy()
        proc_df = pd.concat([context_df, df], ignore_index=True)
        proc_df = proc_df.sort_values(['date', 'race_id'])
        
        proc_df = self.hist_agg.aggregate(proc_df)
        proc_df = self.adv_eng.add_features(proc_df)
        proc_df = self.exp_eng.add_features(proc_df)
        proc_df = self.rel_eng.add_features(proc_df)
        proc_df = self.opp_eng.add_features(proc_df)
        proc_df = self.rating_eng.add_features(proc_df)
        
        # 当日分の抽出
        test_df = proc_df[proc_df['date'] == target_date].copy()
        
        # 3. 予測
        X = test_df[self.feature_cols].copy()
        
        # 重複カラム名の解消 (万が一存在する場合)
        if X.columns.duplicated().any():
            logger.warning("Duplicate columns found, keeping first occurrences.")
            X = X.loc[:, ~X.columns.duplicated()]
            
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
        
        test_df['prob_binary'] = self.model_bin.predict_proba(X)[:, 1]
        test_df['score_rank'] = self.model_rnk.predict(X)
        
        # 正規化 (Min-Max) およびアンサンブル
        def calc_ensemble_score(group):
            # Binary
            b = group['prob_binary']
            if b.max() == b.min(): group['norm_bin'] = 0.5
            else: group['norm_bin'] = (b - b.min()) / (b.max() - b.min())
            
            # Rank
            r = group['score_rank']
            if r.max() == r.min(): group['norm_rnk'] = 0.5
            else: group['norm_rnk'] = (r - r.min()) / (r.max() - r.min())
            
            # Weighted Ensemble (Optimal ratio: 0.6 : 0.4)
            group['ensemble_score'] = 0.6 * group['norm_bin'] + 0.4 * group['norm_rnk']
            
            # Softmax for display (RelScore) - derived from ensemble_score
            s = group['ensemble_score'].values
            exp_s = np.exp(s * 10) # 差を強調するためスケーリング
            group['rel_score'] = exp_s / exp_s.sum()
            
            return group
        
        test_df = test_df.groupby('race_id', group_keys=False).apply(calc_ensemble_score)
        
        test_df['pred_rank'] = test_df.groupby('race_id')['ensemble_score'].rank(ascending=False, method='first').astype(int)
        
        return test_df

    def format_report(self, df: pd.DataFrame, target_date: str) -> str:
        lines = [f"# JRA 予測レポート ({target_date})", "## オッズフリー予測モデル (複勝圏内確率)", ""]
        
        # レース番号（時間順）でソートするために、一意なレースのリストを取得
        race_info = df[['race_id', 'venue', 'date']].drop_duplicates()
        race_info['race_num'] = race_info['race_id'].str[-2:].astype(int)
        
        # レース番号 -> 競馬場コード の順でソート (これが概ね出走時間順になる)
        sorted_races = race_info.sort_values(['race_num', 'venue'])['race_id'].tolist()
        
        for race_id in sorted_races:
            r_df = df[df['race_id'] == race_id].sort_values('pred_rank')
            v_code = r_df['venue'].iloc[0] if 'venue' in r_df.columns else "Unknown"
            venue_name = VENUE_MAP.get(v_code, v_code)
            race_num = int(race_id[-2:])
            
            lines.append(f"### {venue_name} {race_num}R")
            lines.append("| 予測順位 | 馬番 | 馬名 | 複勝率 | スコア | 総合評価 | 相対確率 |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            
            for _, row in r_df.iterrows():
                horse_name = row['horse_name'] if 'horse_name' in row else f"Horse {row['horse_number']}"
                lines.append(f"| {row['pred_rank']} | {row['horse_number']} | {horse_name} | {row['prob_binary']:.1%} | {row['score_rank']:.2f} | {row['ensemble_score']:.3f} | {row['rel_score']:.1%} |")
            lines.append("")
            
        return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--notify", action="store_true", help="Send Discord notification")
    args = parser.parse_args()

    predictor = OddsFreePredictor(BINARY_MODEL_PATH, RANKER_MODEL_PATH)
    results = predictor.predict_date(args.date)
    
    if results is not None and not results.empty:
        report = predictor.format_report(results, args.date)
        
        # 保存
        os.makedirs(REPORT_DIR, exist_ok=True)
        report_path = os.path.join(REPORT_DIR, f"{args.date}_odds_free.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
        
        # 通知
        if args.notify:
            webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
            if webhook_url:
                notifier = NotificationManager(webhook_url)
                notifier.send_text(f"【JRA予測】{args.date} オッズフリーモデル予測完了\n詳細はファイルを確認してください。")
                # 簡略化したサマリーも送る
                summary = "Top Predictions:\n"
                for race_id, r_df in results.groupby('race_id'):
                    top1 = r_df[r_df['pred_rank'] == 1].iloc[0]
                    v_code = top1['venue'] if 'venue' in top1 else "??"
                    venue_name = VENUE_MAP.get(v_code, v_code)
                    summary += f"- {venue_name}{int(race_id[-2:])}R: {top1['horse_name']} (E-Score: {top1['ensemble_score']:.3f})\n"
                notifier.send_text(summary)
            else:
                logger.warning("DISCORD_WEBHOOK_URL not set.")
    else:
        logger.warning("No predictions generated.")

if __name__ == "__main__":
    main()

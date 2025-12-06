import os
import csv
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """
    実験結果を管理・記録するクラス。
    CSV形式の一覧ログと、JSON形式の詳細ログを保存します。
    """
    def __init__(self, experiment_dir='experiments', experiment_name=None):
        self.experiment_dir = experiment_dir
        self.os_experiment_dir = os.path.join(os.getcwd(), experiment_dir) # Absolute path
        os.makedirs(self.os_experiment_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"exp_{self.timestamp}"

        # History CSV file path
        self.history_file = os.path.join(self.os_experiment_dir, 'history.csv')

        # Initialize CSV header if not exists
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'experiment_name', 'model_type', 'metrics_json_path', 'note'])

    def log_result(self, model_type: str, metrics: dict, params: dict = None, note: str = ""):
        """
        実験結果を記録します。

        Args:
            model_type (str): モデルの種類 (例: "Ensemble_LGBM_Cat_TabNet")
            metrics (dict): 評価指標 (例: {'ndcg': 0.5, 'rmse': 1.2})
            params (dict): ハイパーパラメータや設定 (詳細ログ用)
            note (str): 任意のメモ
        """
        # Save detailed JSON log
        detail_log_name = f"{self.experiment_name}_detail.json"
        detail_log_path = os.path.join(self.os_experiment_dir, detail_log_name)

        log_data = {
            'timestamp': self.timestamp,
            'experiment_name': self.experiment_name,
            'model_type': model_type,
            'metrics': metrics,
            'params': params or {},
            'note': note
        }

        with open(detail_log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        # Append to CSV history
        # We flatten some key metrics for easy viewing, but main storage is JSON
        # For CSV, we just store basic info and path to JSON
        with open(self.history_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.timestamp,
                self.experiment_name,
                model_type,
                detail_log_name,
                note
            ])

        logger.info(f"実験結果を記録しました: {self.experiment_name}")
        logger.info(f"詳細ログ: {detail_log_path}")
        logger.info(f"履歴CSV: {self.history_file}")

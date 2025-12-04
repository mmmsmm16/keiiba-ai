import os
import pandas as pd
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)

class RawDataLoader:
    """
    データベースから学習に必要な生データをロードするクラス。
    """
    def __init__(self):
        user = os.environ.get('POSTGRES_USER', 'user')
        password = os.environ.get('POSTGRES_PASSWORD', 'password')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'keiba')
        connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(connection_str)

    def load(self, limit: int = None) -> pd.DataFrame:
        """
        results, races, horses テーブルを結合してデータを取得します。

        Args:
            limit (int, optional): 取得するレコード数の上限（テスト用）。

        Returns:
            pd.DataFrame: 結合済みの生データ。
        """
        query = """
        SELECT
            r.race_id,
            r.date,
            r.venue,
            r.race_number,
            r.distance,
            r.surface,
            r.weather,
            r.state,
            res.horse_id,
            res.jockey_id,
            res.trainer_id,
            res.frame_number,
            res.horse_number,
            res.rank,
            res.time,
            res.last_3f,
            res.odds,
            res.popularity,
            res.weight,
            res.weight_diff,
            res.age,
            h.sex,
            h.name as horse_name,
            h.sire_id,
            h.mare_id
        FROM results res
        JOIN races r ON res.race_id = r.race_id
        LEFT JOIN horses h ON res.horse_id = h.horse_id
        ORDER BY r.date, r.race_id, res.rank
        """

        if limit:
            query += f" LIMIT {limit}"

        logger.info("データベースからデータをロード中...")
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"データのロード完了: {len(df)} 件")
            return df
        except Exception as e:
            logger.error(f"データのロードに失敗しました: {e}")
            raise e

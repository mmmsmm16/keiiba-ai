import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    PostgreSQLデータベースにデータをロードするクラス。
    """
    def __init__(self, connection_str: str = None):
        if connection_str is None:
            # 環境変数から構築
            user = os.environ.get('POSTGRES_USER', 'user')
            password = os.environ.get('POSTGRES_PASSWORD', 'password')
            host = os.environ.get('POSTGRES_HOST', 'db')
            port = os.environ.get('POSTGRES_PORT', '5432')
            dbname = os.environ.get('POSTGRES_DB', 'keiba')
            connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

        self.engine = create_engine(connection_str)

    def save_race_data(self, data: Dict[str, pd.DataFrame]):
        """
        レースデータをデータベースに保存します。
        冪等性を確保するため、race_idの既存データを削除してから保存します。

        Args:
            data (Dict[str, pd.DataFrame]): 'races', 'results', 'horses', 'payouts' を含む辞書。
        """
        df_races = data.get('races')
        if df_races is None or df_races.empty:
            logger.warning("保存するレースデータがありません。")
            return

        race_id = df_races.iloc[0]['race_id']

        with self.engine.begin() as conn:
            # 1. このレースの既存データをクリア（子テーブルから先に削除）
            logger.info(f"レース {race_id} の既存データを削除中")
            conn.execute(text("DELETE FROM payouts WHERE race_id = :race_id"), {"race_id": race_id})
            conn.execute(text("DELETE FROM results WHERE race_id = :race_id"), {"race_id": race_id})
            conn.execute(text("DELETE FROM races WHERE race_id = :race_id"), {"race_id": race_id})

            # 2. 馬情報のUpsert
            # Postgresの ON CONFLICT DO NOTHING を使用して単純な馬情報をUpsert
            # Pandasのto_sqlはUpsertを簡単にはサポートしていないため、生SQLを使用
            # 今回のMVPでは、既存のIDと競合する場合は何もしない（DO NOTHING）

            df_horses = data.get('horses')
            if df_horses is not None and not df_horses.empty:
                self._upsert_horses(conn, df_horses)

            # 3. レース情報の挿入
            df_races.to_sql('races', conn, if_exists='append', index=False)

            # 4. 結果情報の挿入
            df_results = data.get('results')
            if df_results is not None and not df_results.empty:
                df_results.to_sql('results', conn, if_exists='append', index=False)

            # 5. 払い戻し情報の挿入
            df_payouts = data.get('payouts')
            if df_payouts is not None and not df_payouts.empty:
                df_payouts.to_sql('payouts', conn, if_exists='append', index=False)

            logger.info(f"レース {race_id} のデータを正常に保存しました")

    def _upsert_horses(self, conn, df_horses: pd.DataFrame):
        """
        生SQLを使用して馬情報をUpsertします。
        """
        # 一括処理としては効率が悪いですが、安全です。
        # INSERT文を構築
        # 注: レースページからは限られた情報（ID、名前、性別）しか得られません。
        # すでに詳細情報がある場合は上書きしたくないため、DO NOTHING が適切です。

        statement = text("""
            INSERT INTO horses (horse_id, name, sex)
            VALUES (:horse_id, :name, :sex)
            ON CONFLICT (horse_id) DO NOTHING
        """)

        for _, row in df_horses.iterrows():
            conn.execute(statement, {
                'horse_id': row['horse_id'],
                'name': row['name'],
                'sex': row['sex']
            })

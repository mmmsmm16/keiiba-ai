import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads data into PostgreSQL database.
    """
    def __init__(self, connection_str: str = None):
        if connection_str is None:
            # Construct from env vars
            user = os.environ.get('POSTGRES_USER', 'user')
            password = os.environ.get('POSTGRES_PASSWORD', 'password')
            host = os.environ.get('POSTGRES_HOST', 'db')
            port = os.environ.get('POSTGRES_PORT', '5432')
            dbname = os.environ.get('POSTGRES_DB', 'keiba')
            connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

        self.engine = create_engine(connection_str)

    def save_race_data(self, data: Dict[str, pd.DataFrame]):
        """
        Saves race data to the database.
        Handles deletion of existing data for the race_id to ensure idempotency.

        Args:
            data (Dict[str, pd.DataFrame]): Dictionary containing 'races', 'results', 'horses', 'payouts'.
        """
        df_races = data.get('races')
        if df_races is None or df_races.empty:
            logger.warning("No race data to save.")
            return

        race_id = df_races.iloc[0]['race_id']

        with self.engine.begin() as conn:
            # 1. Clear existing data for this race (children first)
            logger.info(f"Clearing existing data for race {race_id}")
            conn.execute(text("DELETE FROM payouts WHERE race_id = :race_id"), {"race_id": race_id})
            conn.execute(text("DELETE FROM results WHERE race_id = :race_id"), {"race_id": race_id})
            conn.execute(text("DELETE FROM races WHERE race_id = :race_id"), {"race_id": race_id})

            # 2. Insert Horses (Upsert)
            # Postgres upsert using ON CONFLICT DO NOTHING for simple horse info
            # Pandas to_sql doesn't do upsert easily.
            # We will iterate and execute raw SQL for horses or use a temporary table strategy.
            # For simplicity in this MVP, we will fetch existing IDs and filter.
            # But that's slow.
            # Let's use INSERT ... ON CONFLICT DO NOTHING

            df_horses = data.get('horses')
            if df_horses is not None and not df_horses.empty:
                self._upsert_horses(conn, df_horses)

            # 3. Insert Races
            df_races.to_sql('races', conn, if_exists='append', index=False)

            # 4. Insert Results
            df_results = data.get('results')
            if df_results is not None and not df_results.empty:
                df_results.to_sql('results', conn, if_exists='append', index=False)

            # 5. Insert Payouts
            df_payouts = data.get('payouts')
            if df_payouts is not None and not df_payouts.empty:
                df_payouts.to_sql('payouts', conn, if_exists='append', index=False)

            logger.info(f"Successfully saved data for race {race_id}")

    def _upsert_horses(self, conn, df_horses: pd.DataFrame):
        """
        Upserts horse data using raw SQL.
        """
        # This is a bit inefficient for bulk, but safe.
        # Construct INSERT statement
        # Note: We only have limited info (id, name, sex) from race page.
        # We don't want to overwrite if we have more info already.
        # So DO NOTHING is appropriate.

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

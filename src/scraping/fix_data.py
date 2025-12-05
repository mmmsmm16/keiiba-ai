import os
import sys
from sqlalchemy import create_engine, text
import logging

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_data():
    """
    データの整合性をチェックし、不完全なデータ（レース情報はあるが結果がない）を削除します。
    """
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'keiba')
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    try:
        engine = create_engine(connection_str)
    except Exception as e:
        logger.error(f"DB接続エラー: {e}")
        return

    with engine.begin() as conn:
        # 1. 不整合データの検索
        # resultsテーブルにレコードが存在しないracesテーブルのレコードを探す
        logger.info("不整合データを検索中...")
        query = text("""
            SELECT r.race_id
            FROM races r
            LEFT JOIN results res ON r.race_id = res.race_id
            WHERE res.race_id IS NULL
        """)
        result = conn.execute(query)
        incomplete_races = [row[0] for row in result]

        if not incomplete_races:
            logger.info("不整合なデータは見つかりませんでした。正常です。")
            return

        logger.info(f"{len(incomplete_races)} 件の不整合データ（結果がないレース）が見つかりました。")
        logger.info(f"削除対象ID（一部）: {incomplete_races[:5]} ...")

        # 2. 削除実行
        # payoutsとracesから削除（resultsはそもそも無いので不要）
        # 安全のためループ処理で実行 (SQLAlchemyのIN句展開トラブル回避)

        if len(incomplete_races) > 0:
            logger.info("削除を実行中...")

            del_payouts = text("DELETE FROM payouts WHERE race_id = :id")
            del_races = text("DELETE FROM races WHERE race_id = :id")

            for rid in incomplete_races:
                conn.execute(del_payouts, {"id": rid})
                conn.execute(del_races, {"id": rid})

            logger.info("削除が完了しました。")
            logger.info("再度 bulk_loader.py を実行すると、これらのレースは再取得されます。")

if __name__ == "__main__":
    fix_data()

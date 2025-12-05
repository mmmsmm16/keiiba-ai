import os
import sys
from sqlalchemy import create_engine, text
import logging

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(message)s') # 見やすいようにメッセージのみ出力
logger = logging.getLogger(__name__)

def inspect_db():
    """
    データベース内のテーブル定義（スキーマ）を出力するツール。
    PC-KEIBA等で作成されたテーブル構造を確認するために使用します。
    """
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'keiba')
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    try:
        engine = create_engine(connection_str)
        with engine.connect() as conn:
            # テーブル一覧を取得
            logger.info("========== データベース スキーマ情報 ==========")
            query = text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in conn.execute(query)]

            if not tables:
                logger.warning("テーブルが見つかりません。データインポートが完了していない可能性があります。")
                return

            logger.info(f"検出されたテーブル数: {len(tables)}")

            for table in tables:
                logger.info(f"\n[Table: {table}]")
                # カラム情報を取得
                col_query = text("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = :table AND table_schema = 'public'
                    ORDER BY ordinal_position
                """)
                cols = conn.execute(col_query, {"table": table})
                for col in cols:
                    logger.info(f"  - {col[0]}: {col[1]}")

            logger.info("\n=============================================")
            logger.info("この出力を開発者(AI)に共有してください。")

    except Exception as e:
        logger.error(f"DB接続エラー: {e}")
        logger.error("docker-composeが起動しているか確認してください。")

if __name__ == "__main__":
    inspect_db()

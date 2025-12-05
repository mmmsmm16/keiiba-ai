import os
import sys
from sqlalchemy import create_engine, text
import logging

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def inspect_db():
    """
    全データベースとテーブル定義を出力するツール。
    PC-KEIBAがどこにデータを作成したか特定するために使用します。
    """
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432') # 内部ポートは固定

    # まずデフォルトの postgres DB に接続してDB一覧を取得
    root_conn_str = f"postgresql://{user}:{password}@{host}:{port}/postgres"

    try:
        engine = create_engine(root_conn_str)
        with engine.connect() as conn:
            logger.info("========== データベース一覧 ==========")
            query = text("SELECT datname FROM pg_database WHERE datistemplate = false")
            dbs = [row[0] for row in conn.execute(query)]
            for db in dbs:
                logger.info(f"- {db}")

        # 各データベースの中身を確認
        for db_name in dbs:
            logger.info(f"\n========== データベース: {db_name} のテーブル ==========")
            db_conn_str = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            try:
                db_engine = create_engine(db_conn_str)
                with db_engine.connect() as db_conn:
                    query = text("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                        ORDER BY table_name
                    """)
                    tables = [row[0] for row in db_conn.execute(query)]

                    if not tables:
                        logger.info("(テーブルなし)")
                        continue

                    for table in tables:
                        logger.info(f"[Table: {table}]")
                        # カラム情報を取得 (最初の3つだけ表示して省略)
                        col_query = text("""
                            SELECT column_name, data_type
                            FROM information_schema.columns
                            WHERE table_name = :table AND table_schema = 'public'
                            ORDER BY ordinal_position
                        """)
                        cols = db_conn.execute(col_query, {"table": table}).fetchall()
                        for col in cols:
                            logger.info(f"  - {col[0]}: {col[1]}")

            except Exception as e:
                logger.error(f"  アクセスエラー: {e}")

    except Exception as e:
        logger.error(f"DB接続エラー: {e}")
        logger.error("docker-composeが起動しているか確認してください。")

if __name__ == "__main__":
    inspect_db()

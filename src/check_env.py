import os
import psycopg2
import sys
from sqlalchemy import create_engine, text

def check_db_connection():
    """
    docker-compose.yml で定義されたPostgreSQLデータベースへの接続を確認します。
    """
    db_user = os.environ.get('POSTGRES_USER', 'user')
    db_password = os.environ.get('POSTGRES_PASSWORD', 'password')
    db_name = os.environ.get('POSTGRES_DB', 'keiba')
    db_host = os.environ.get('POSTGRES_HOST', 'db') # 'db' は docker-compose のサービス名
    db_port = os.environ.get('POSTGRES_PORT', '5432')

    print(f"データベース {db_name} ({db_host}:{db_port}) にユーザー {db_user} として接続を試みています...")

    try:
        # 接続文字列の構築

        connection_str = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_str)

        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("データベースへの接続に成功しました！")

            # テーブルの存在確認
            result = connection.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]
            print(f"検出されたテーブル: {tables}")

            expected_tables = {'races', 'horses', 'results', 'payouts'}
            if expected_tables.issubset(set(tables)):
                print("期待される全てのテーブルが存在します。")
            else:
                print(f"不足しているテーブル: {expected_tables - set(tables)}")

    except Exception as e:
        print(f"データベースへの接続エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_db_connection()

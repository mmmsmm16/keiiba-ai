import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_db():
    """
    Diagnoses the PC-KEIBA database connection and table status.
    """
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')

    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    logger.info(f"Connecting to database: {dbname} at {host}:{port} as {user}...")

    try:
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            logger.info("Connection successful!")

            # Check for JRA-VAN tables (Long and Short names)
            expected_tables = [
                'jvd_race_shosai',
                'jvd_seiseki',
                'jvd_uma_master',
                'jvd_haraimodoshi',
                'jvd_ra', # Short name Race
                'jvd_se', # Short name Seiseki
                'jvd_um', # Short name Uma
                'jvd_hr', # Short name Haraimodoshi
                'race_shosai',
                'seiseki'
            ]

            logger.info("Checking for expected tables...")
            found_tables = []

            # Get all tables in public schema
            query = text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            result = conn.execute(query)
            all_tables = [row[0] for row in result]

            logger.info(f"Found {len(all_tables)} tables in 'public' schema.")

            if not all_tables:
                logger.warning("WARNING: No tables found in the database. Did you run the PC-KEIBA import?")
            else:
                for t in all_tables:
                    logger.info(f" - {t}")
                    if t in expected_tables:
                        found_tables.append(t)

            # Check specific expected tables and row counts
            if found_tables:
                logger.info("--- Row Counts for Key Tables ---")
                for t in found_tables:
                    try:
                        count_query = text(f"SELECT COUNT(*) FROM {t}")
                        count = conn.execute(count_query).scalar()
                        logger.info(f"Table '{t}': {count} rows")
                    except Exception as e:
                        logger.error(f"Could not count rows for {t}: {e}")
            else:
                logger.error("CRITICAL: None of the expected JRA-VAN tables (jvd_*) were found.")

    except OperationalError as e:
        logger.error(f"Connection failed: {e}")
        logger.error("Please check if the database container is running and accessible.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    diagnose_db()


import os
import pandas as pd
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def fetch_payouts(years=[2024, 2025]):
    engine = get_db_engine()
    years_str = ",".join([f"'{y}'" for y in years])
    query = text(f"SELECT * FROM jvd_hr WHERE kaisai_nen IN ({years_str})")
    
    logger.info(f"Fetching payouts for years: {years_str}")
    df = pd.read_sql(query, engine)
    
    # Construct race_id
    if not df.empty:
        df['race_id'] = (
            df['kaisai_nen'].astype(str) +
            df['keibajo_code'].astype(str) +
            df['kaisai_kai'].astype(str) +
            df['kaisai_nichime'].astype(str) +
            df['race_bango'].astype(str)
        )
    return df

def main():
    # 1. Fetch Payouts
    df_payout = fetch_payouts([2024, 2025])
    output_path = '/workspace/experiments/payouts_2024_2025.parquet'
    df_payout.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df_payout)} payout records to {output_path}")

    # 2. Inspect Predictions Columns
    pred_path = '/workspace/experiments/predictions_catboost_v7.parquet'
    if os.path.exists(pred_path):
        df_pred = pd.read_parquet(pred_path)
        logger.info(f"Predictions Columns: {df_pred.columns.tolist()}")
        if 'expected_value' in df_pred.columns:
            logger.info("Found 'expected_value' column.")
        else:
            logger.warning("'expected_value' column NOT found!")
    else:
        logger.error("Predictions file not found!")

if __name__ == "__main__":
    main()

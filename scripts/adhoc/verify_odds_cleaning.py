
import pandas as pd
import numpy as np
import logging

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mock_step_inject_odds(df: pd.DataFrame, t10_df: pd.DataFrame = None) -> pd.DataFrame:
    """修正版のロジックを模倣した関数"""
    
    # T-10データがない場合 (df_listが空だった場合を想定)
    if t10_df is None or t10_df.empty:
        logger.warning("No T-10 odds found. Clearing 'odds' and 'popularity' to prevent leakage.")
        if 'odds' in df.columns:
            df['odds'] = np.nan
        if 'popularity' in df.columns:
            df['popularity'] = np.nan
        return df

    # マージ処理（run_jra_pipeline_backtest.pyと同じロジック）
    merged = pd.merge(df, t10_df, on=['race_id', 'horse_number'], suffixes=('', '_t10'), how='left')
    
    # [Fix] 確定オッズへのフォールバックを廃止。T-10がない場合はNaNとする。
    if 'odds_t10' in merged.columns:
        merged['odds'] = merged['odds_t10']
        merged['popularity'] = merged['popularity_t10']
        merged.drop(columns=['odds_t10', 'popularity_t10'], inplace=True, errors='ignore')
    else:
        # マージ後もない場合
        merged['odds'] = np.nan
        merged['popularity'] = np.nan
    
    return merged

def test_odds_leak_prevention():
    logger.info("=== Testing Odds Leak Prevention ===")

    # 1. データ準備（確定オッズ 'odds' が入っている状態）
    df = pd.DataFrame({
        'race_id': ['1', '1', '2', '2'],
        'horse_number': [1, 2, 1, 2],
        'odds': [2.5, 3.0, 1.5, 5.0],  # 確定オッズ（リーク源）
        'popularity': [1, 2, 1, 3]
    })
    
    logger.info(f"Original DataFrame:\n{df}")

    # 2. ケースA: T-10データが全くない場合（ロード失敗時など）
    logger.info("\n--- Case A: No T-10 Data (Load Failed) ---")
    res_a = mock_step_inject_odds(df.copy(), None)
    logger.info(f"Result A:\n{res_a}")
    
    # 検証
    if res_a['odds'].isna().all():
        logger.info("✅ Case A Passed: All odds are NaN.")
    else:
        logger.error("❌ Case A Failed: Odds leaked!")
        
    # 3. ケースB: T-10データはあるが、一部欠損している場合
    logger.info("\n--- Case B: T-10 Data Exists (Partial) ---")
    t10_df = pd.DataFrame({
        'race_id': ['1', '1'],  # race_id=2 はT-10なし
        'horse_number': [1, 2],
        'odds_t10': [2.8, 3.2], # T-10オッズ（確定オッズとは違う値）
        'popularity_t10': [1, 2]
    })
    
    # mock関数は結合ロジックを含む
    res_b = mock_step_inject_odds(df.copy(), t10_df)
    logger.info(f"Result B:\n{res_b}")
    
    # 検証 race_id=1
    r1 = res_b[res_b['race_id'] == '1']
    if np.isclose(r1['odds'].values[0], 2.8):
        logger.info("✅ Case B (Race 1) Passed: Used T-10 odds.")
    else:
        logger.error(f"❌ Case B (Race 1) Failed: Value {r1['odds'].values[0]} != 2.8")

    # 検証 race_id=2 (T-10なし -> NaNになるべき。元の1.5や5.0が残っていたらNG)
    r2 = res_b[res_b['race_id'] == '2']
    if r2['odds'].isna().all():
        logger.info("✅ Case B (Race 2) Passed: Odds are NaN (No leak).")
    else:
        logger.error(f"❌ Case B (Race 2) Failed: Odds leaked! {r2['odds'].values}")

if __name__ == "__main__":
    test_odds_leak_prevention()

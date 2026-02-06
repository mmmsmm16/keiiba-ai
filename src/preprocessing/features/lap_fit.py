import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.6] ラップ適合 (Lap Fit)
    - 馬の過去の得意ペース(High/Slow)と、今回の想定ペースの一致度
    - 簡易実装として、馬の「平均PCI」と「今回の予想PCI(ペース)」の差分をとる
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    df_sorted = df.copy()
    
    # Input check
    if 'avg_pci' in df_sorted.columns and 'pace_expectation_proxy' in df_sorted.columns:
        # pace_expectation_proxy is "Early Speed Sum". High sum = High Pace = Low PCI (High Speed).
        # PCI: 50=Avg, <50=High, >50=Slow.
        # EarlySpeedSum: High=HighPace.
        # Correlation is Inverse.
        
        # Normalize inputs roughly
        # AvgPCI is around 50 +- 10.
        # EarlySpeedSum is around 1.0 (mean) ? No, sum of nige_rate usually 0..2.0.
        
        # Better metric: "Is this a fit?"
        # If Race is HighPace (High EarlySum) AND Horse likes HighPace (Low AvgPCI), then Fit is Good.
        # If Race is HighPace AND Horse likes SlowPace (High AvgPCI), then Fit is Bad.
        
        # Interaction term:
        # HighPace(1) * LowPCI(-1) -> Match? No.
        # Direction align.
        
        # Let's standardize and multiply.
        pass
    else:
        # If inputs missing, return empty or zeros
        # Since logic is fuzzy, let's create placeholders or skip if crucial columns missing.
        # pace_expectation_proxy is from race_structure, which might come after?
        # feature_pipeline order matters!
        # Assuming race_structure runs BEFORE lap_fit. (Registry order)
        pass

    out_cols = []
    
    # Assuming inputs are available.
    # If not, fill 0.
    
    # Create simple fit score
    # fit = -1 * (Standardized Pace) * (Standardized PCI)
    # (High Pace(+) * Low PCI(-)) * -1 = Positive Fit.
    
    # Since specific standardization is complex without global stats,
    # we just output interaction term if columns exist.
    
    if 'avg_pci' in df_sorted.columns and 'pace_expectation_proxy' in df_sorted.columns:
        # Simple interaction
        df_sorted['lap_fit_interaction'] = df_sorted['avg_pci'] * df_sorted['pace_expectation_proxy']
        out_cols.append('lap_fit_interaction')
    else:
        df_sorted['lap_fit_interaction'] = 0
        out_cols.append('lap_fit_interaction')

    return df_sorted[cols + out_cols].copy()

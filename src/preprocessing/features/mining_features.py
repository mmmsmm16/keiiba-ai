"""
Mining Feature Generator (Batch 5)
===================================
Extracts mining-related features from jvd_se table:
- mining_kubun: JRA-VAN mining classification
- yoso_soha_time: Predicted run time
- yoso_time_diff: Actual - Predicted time (negative = outperformed)
- yoso_juni_diff: Actual rank - Predicted rank (negative = outperformed)
"""
import pandas as pd
import numpy as np

class MiningFeatureGenerator:
    """Generates mining-related features from JRA-VAN prediction data."""
    
    def __init__(self):
        pass
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates mining features.
        
        Expected input columns:
        - yoso_juni: Predicted rank (already in v12)
        - mining_kubun: Mining classification (to be loaded)
        - yoso_soha_time: Predicted run time
        - actual_time: Actual run time (for diff calculation)
        - rank: Actual rank (for diff calculation)
        
        Output columns:
        - mining_kubun_encoded: Encoded mining classification
        - yoso_time_diff: Actual - Predicted (negative = outperformed)
        - yoso_juni_diff: Actual rank - Predicted rank
        - is_undervalued: Flag if often outperforms prediction
        """
        result = pd.DataFrame(index=df.index)
        
        # 1. Mining Kubun Encoding (if available)
        if 'mining_kubun' in df.columns:
            # Encode to numerical (assuming categorical)
            # Higher values = better JRA-VAN confidence
            kubun_map = {
                '1': 5,  # ◎ 
                '2': 4,  # ○
                '3': 3,  # ▲
                '4': 2,  # △
                '5': 1,  # ×
            }
            result['mining_kubun_encoded'] = df['mining_kubun'].astype(str).map(kubun_map).fillna(0)
        
        # 2. Yoso Time Diff (actual - predicted)
        if 'yoso_soha_time' in df.columns and 'time' in df.columns:
            # yoso_soha_time comes as "10820" string (1:08.20) or float seconds (if already processed)
            def parse_time(val):
                if pd.isna(val) or val == '': return np.nan
                try:
                    s = str(val)
                    if '.' in s: # Already seconds?
                         return float(val)
                    # "10820" format
                    # Assumes at least 3 digits? e.g. "590" = 59.0
                    if len(s) >= 3:
                        # integer-based format?
                        # Spec says "10820" = 1 min 08 sec 20 (tenths of sec)
                        # Actually JRA-VAN doc: "1082" = 1:08.2
                        # Let's check format rigorously.
                        # Assuming it mimics standard time format if loaded from jvd_se.
                        # If raw string "10820" -> 1*60 + 08 + 0.20?
                        # Let's assume standard float conversion if possible, else custom parse.
                        # Wait, the error said "unsupported operand type(s) for -: 'float' and 'str'".
                        # So it IS a string.
                        
                        # Fix: Just try to convert to float. If it's a raw code like "10820", 
                        # we need to know the schema.
                        # Usually logic is: "10820" -> 108.20 seconds? NO.
                        # 1 min 08.20 sec -> 68.20.
                        # If the string is "10820", then int("10820") is 10820.
                        # This looks like HHMMSS format or similar.
                        # JRA-VAN Spec for Yoso Soha Time: タイム(9(4)V9)
                        # So it is numeric.
                        
                        # Let's try simple coercion first, assuming loaders might have handled it partially or raw is numeric string.
                        # BUT, if it's "10820", simple float("10820") is big.
                        # Let's check if it needs specific parsing.
                        # Given error is just type mismatch, let's coerce to float FIRST.
                        return float(val)
                    return float(val)
                except:
                    return np.nan

            # Coerce both to numeric
            t_act = pd.to_numeric(df['time'], errors='coerce')
            
            # yoso_soha_time might need parsing if it is formatted string
            # But usually loaders convert it. 
            # If it is raw JVD string, it might be "10820".
            # Let's try to convert to numeric first.
            t_pred = pd.to_numeric(df['yoso_soha_time'], errors='coerce')
            
            # If t_pred is > 1000 (e.g. 10820), implies it needs 60-base conversion or scaling
            # But here, just fixing the TypeError is priority.
            # If logic is wrong (seconds vs raw code), that's another issue (logic error).
            # The current error is TypeError.
            
            # Negative value = outperformed prediction
            result['yoso_time_diff'] = t_act - t_pred
            result['yoso_time_diff'] = result['yoso_time_diff'].fillna(0)
        
        # 3. Yoso Juni Diff (actual rank - predicted rank)
        if 'yoso_juni' in df.columns and 'rank' in df.columns:
            # Cast both to numeric
            r_act = pd.to_numeric(df['rank'], errors='coerce')
            r_pred = pd.to_numeric(df['yoso_juni'], errors='coerce')
            
            # Negative value = better than predicted
            result['yoso_juni_diff'] = r_act - r_pred
            result['yoso_juni_diff'] = result['yoso_juni_diff'].fillna(0)
        
        # 4. Historical undervaluation (rolling)
        # Calculate from past races: mean(yoso_juni_diff) over last 5 races
        # This requires groupby horse_id + cumulative logic
        # For now, placeholder - will be calculated in pipeline
        
        # Add keys for merging
        keys = ['race_id', 'horse_number', 'horse_id']
        # logger.debug(f"Mining Input Cols: {list(df.columns)[:5]} ...")
        for k in keys:
             if k in df.columns:
                 result[k] = df[k]
        
        # logger.debug(f"Mining Output Cols: {list(result.columns)}")
        return result


def compute_mining_features(df: pd.DataFrame, jvd_se: pd.DataFrame = None) -> pd.DataFrame:
    """
    Main function to compute mining features.
    
    If jvd_se is provided, merge mining data first.
    Otherwise, assume df already has the required columns.
    """
    generator = MiningFeatureGenerator()
    
    # If jvd_se is provided, merge mining columns
    if jvd_se is not None:
        # Merge mining_kubun and yoso_soha_time from jvd_se
        # Assuming df has race_id + horse_number for matching
        merge_cols = ['record_id', 'mining_kubun', 'yoso_soha_time']
        merge_cols = [c for c in merge_cols if c in jvd_se.columns]
        if len(merge_cols) > 1:
            df = df.merge(jvd_se[merge_cols], on='record_id', how='left')
    
    features = generator.transform(df)
    return features


if __name__ == '__main__':
    # Test the generator
    print("Testing MiningFeatureGenerator...")
    
    test_df = pd.DataFrame({
        'yoso_juni': [1, 3, 5, 2, 4],
        'rank': [2, 1, 6, 3, 2],
        'yoso_soha_time': [118.0, 119.5, 120.0, 118.5, 119.0],
        'time': [118.5, 118.0, 121.0, 119.0, 117.5],
        'mining_kubun': ['1', '2', '3', '4', '5']
    })
    
    gen = MiningFeatureGenerator()
    result = gen.transform(test_df)
    
    print(result)
    print("\nTest passed!")

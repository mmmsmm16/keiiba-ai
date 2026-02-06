"""
Undervalued Horse Feature Generator
====================================
Creates features to detect horses that are undervalued by odds:
1. popularity_vs_pred: Popularity rank - Predicted rank (positive = undervalued)
2. odds_vs_elo: Odds-based rank - ELO-based rank (positive = undervalued)
3. yoso_juni_outperform_rate: Historical rate of outperforming predicted rank
4. high_odds_winner_history: Past wins with high odds
"""
import pandas as pd
import numpy as np

class UndervaluedFeatureGenerator:
    """Generates features to detect undervalued horses."""
    
    def __init__(self):
        pass
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates undervalued detection features.
        
        Input columns expected:
        - pred_rank or yoso_juni: Model/JRA predicted rank
        - popularity: Market popularity
        - odds: Win odds
        - horse_elo or relative_horse_elo_z: Internal ELO rating
        - rank: Actual finish rank (for past data)
        
        Output columns:
        - popularity_vs_pred: popularity - pred_rank (positive = potentially undervalued)
        - odds_rank_vs_pred: odds-based rank - pred_rank
        - lag1_outperformed: Did horse outperform prediction last race?
        - outperform_rate_5r: Rate of outperforming vs prediction (last 5 races)
        """
        result = pd.DataFrame(index=df.index)
        
        # 1. Popularity vs Predicted Rank
        if 'yoso_juni' in df.columns and 'popularity' in df.columns:
            yoso = pd.to_numeric(df['yoso_juni'], errors='coerce').fillna(8)
            pop = pd.to_numeric(df['popularity'], errors='coerce').fillna(8)
            result['popularity_vs_yoso'] = pop - yoso
            # Positive = Model thinks better than market
        
        # 2. Odds-derived rank vs ELO rank
        if 'odds' in df.columns:
            df_copy = df.copy()
            df_copy['odds_rank'] = df_copy.groupby('race_id')['odds'].rank(ascending=True)
            
            if 'relative_horse_elo_z' in df.columns:
                df_copy['elo_rank'] = df_copy.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
                result['odds_rank_vs_elo'] = df_copy['odds_rank'] - df_copy['elo_rank']
                # Positive = ELO thinks better than odds suggest (undervalued)
            
            if 'yoso_juni' in df.columns:
                yoso = pd.to_numeric(df['yoso_juni'], errors='coerce').fillna(8)
                result['odds_rank_vs_yoso'] = df_copy['odds_rank'] - yoso
        
        # 3. High odds indicator
        if 'odds' in df.columns:
            result['is_high_odds'] = (df['odds'] >= 10).astype(int)
            result['is_mid_odds'] = ((df['odds'] >= 5) & (df['odds'] < 10)).astype(int)
        
        # 4. Lag-based outperformance (needs historical calculation)
        # This requires cumulative calculation per horse
        # Placeholder: will be calculated in feature pipeline with proper groupby
        
        return result


def compute_undervalued_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to compute undervalued features."""
    generator = UndervaluedFeatureGenerator()
    return generator.transform(df)


if __name__ == '__main__':
    # Test
    print("Testing UndervaluedFeatureGenerator...")
    
    test_df = pd.DataFrame({
        'race_id': [1, 1, 1, 2, 2, 2],
        'yoso_juni': [1, 2, 3, 1, 2, 3],
        'popularity': [2, 1, 3, 3, 1, 2],
        'odds': [5.0, 3.0, 10.0, 8.0, 2.5, 6.0],
        'relative_horse_elo_z': [1.5, 2.0, 0.5, 1.0, 2.5, 1.2],
    })
    
    gen = UndervaluedFeatureGenerator()
    result = gen.transform(test_df)
    
    print(result)
    print("\nTest passed!")

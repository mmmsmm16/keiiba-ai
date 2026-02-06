import pandas as pd

def analyze():
    df = pd.read_csv("reports/simulations/i2_combinations_summary.csv")
    
    # 1. Wide Analysis (Frequency Layer)
    # Goal: Hit Day Rate >= 20%, ROI >= 85%
    # Sort by ROI Train desc, then Test
    print("=== WIDE STRATEGIES (Frequency Layer) ===")
    w_df = df[df['strategy'].str.startswith('W') | df['strategy'].str.startswith('Wide')].copy()
    
    # Filter reasonable bets count
    w_df = w_df[w_df['bets_test'] >= 100] 
    
    w_top = w_df.sort_values('roi_test', ascending=False)
    print(w_top[['strategy', 'filter', 'roi_train', 'roi_test', 'hit_day_test', 'bets_test']].head(10).to_string())
    
    # Select Best Wide
    # Criteria: ROI Test >= 0.85, Hit Day Test >= 0.20
    w_valid = w_df[(w_df['roi_test'] >= 0.85) & (w_df['hit_day_test'] >= 0.20)].sort_values('roi_test', ascending=False)
    print("\n[Wide Candidates (ROI>=85%, HitDay>=20%)]")
    if not w_valid.empty:
        print(w_valid[['strategy', 'filter', 'roi_train', 'roi_test', 'hit_day_test', 'bets_test']].head(5).to_string())
    else:
        print("None found.")

    # 2. Trio Analysis (Optional)
    print("\n=== TRIO STRATEGIES ===")
    t_df = df[df['strategy'].str.startswith('T3F') | df['strategy'].str.startswith('Trio')].copy()
    t_df = t_df[t_df['bets_test'] >= 100]
    t_top = t_df.sort_values('roi_test', ascending=False)
    print(t_top[['strategy', 'filter', 'roi_train', 'roi_test', 'hit_day_test', 'bets_test']].head(5).to_string())

    # 3. Trifecta Analysis (Optional High Risk)
    print("\n=== TRIFECTA STRATEGIES ===")
    tr_df = df[df['strategy'].str.startswith('T3T') | df['strategy'].str.startswith('Tri')].copy()
    tr_df = tr_df[tr_df['bets_test'] >= 100]
    tr_top = tr_df.sort_values('roi_test', ascending=False)
    print(tr_top[['strategy', 'filter', 'roi_train', 'roi_test', 'hit_day_test', 'bets_test']].head(5).to_string())

if __name__ == "__main__":
    analyze()

import pandas as pd
import numpy as np

def generate_pareto():
    # Load H2 results (Grid Search) - contains broad search
    df_h2 = pd.read_csv("reports/simulations/results/grid_search_h2.csv")
    
    # Load H4 results (Practical) - contains nested results
    try:
        df_h4 = pd.read_csv("reports/simulations/results/practical_optimization_h4.csv")
        # Rename cols to match if needed, or just use H2 for broad view
        # H2 has 'roi', 'num_bets', 'min_yearly_roi'
    except:
        df_h4 = pd.DataFrame()

    # Use H2 for the main Pareto curve as it covers more ground
    df = df_h2.copy()
    
    # Sort by bets descending
    df = df.sort_values('num_bets', ascending=False)
    
    # Pareto logic: Keep row if ROI is max among all rows with >= bets
    pareto = []
    current_max_roi = -1.0
    
    for i, row in df.iterrows():
        if row['roi'] > current_max_roi:
            pareto.append(row)
            current_max_roi = row['roi']
            
    pareto_df = pd.DataFrame(pareto).sort_values('num_bets')
    
    print("\n--- Pareto Frontier (Bets vs ROI) ---")
    print(pareto_df[['bet_type', 'num_bets', 'roi', 'min_yearly_roi', 't1', 't2', 'e1']].to_string(index=False))
    
    pareto_df.to_csv("reports/simulations/results/pareto_frontier.csv", index=False)

if __name__ == "__main__":
    generate_pareto()

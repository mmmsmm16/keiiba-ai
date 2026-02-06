
import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append(os.getcwd())
from src.probability.ticket_probabilities import compute_ticket_probs

def main():
    # Mock some races
    # Race 1: Strong favorite (Odds 1.5)
    # Race 2: Confusing race (Odds 3.0, 3.5, 4.0...)
    
    races = []
    
    # R1
    races.append({
        'race_id': 'R1_Favorite',
        'horses': range(1, 11),
        'frames': [1,2,3,4,5,6,7,7,8,8],
        'probs': [0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025]
    })
    
    # R2
    races.append({
        'race_id': 'R2_Confusing',
        'horses': range(1, 17),
        'frames': [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8],
        'probs': [0.1]*5 + [0.05]*5 + [0.02]*6 # roughly
    })
    
    report = "# Task 2: Probability Engine Validation\n\n"
    report += "## Performance & Consistency\n"
    
    times = []
    
    for r in races:
        rid = r['race_id']
        df = pd.DataFrame({
            'horse_number': r['horses'],
            'frame_number': r['frames'],
            'pred_prob': r['probs']
        })
        
        start = time.time()
        res = compute_ticket_probs(df, n_samples=20000)
        elapsed = time.time() - start
        times.append(elapsed)
        
        report += f"\n### Race: {rid} (N={len(df)})\n"
        report += f"- Execution Time: {elapsed*1000:.2f} ms\n"
        
        # Display Top 5 Win
        win = res['win'].sort_values(ascending=False).head(5)
        report += "- **Top 5 Win Probs**:\n"
        report += win.to_markdown() + "\n"
        
        # Display Top 5 Place
        place = res['place'].sort_values(ascending=False).head(5)
        report += "- **Top 5 Place Probs**:\n"
        report += place.to_markdown() + "\n"

        # Display Top 5 Umaren
        # Flatten matrix
        um = res['umaren'].stack().reset_index()
        um.columns = ['H1', 'H2', 'Prob']
        um = um[um['H1'] < um['H2']] # Unique pairs
        um = um.sort_values('Prob', ascending=False).head(5)
        report += "- **Top 5 Umaren Probs**:\n"
        report += um.to_markdown(index=False) + "\n"
        
        # Display Top 5 Wakuren
        wa = res['wakuren'].stack().reset_index()
        wa.columns = ['F1', 'F2', 'Prob']
        wa = wa[wa['F1'] <= wa['F2']]
        wa = wa.sort_values('Prob', ascending=False).head(5)
        report += "- **Top 5 Wakuren Probs**:\n"
        report += wa.to_markdown(index=False) + "\n"
        
    avg_time = np.mean(times)
    report += f"\n## Performance Summary\n- Average Time per Race (20k samples): {avg_time*1000:.2f} ms\n"
    if avg_time < 0.1:
        report += "- **Status**: Fast enough for batch processing (50k races = ~1.5 hours).\n"
    else:
        report += "- **Status**: Slightly slow. Consider reducing samples or optimizing.\n"
        
    with open('reports/phase12/task2_probability_engine_validation.md', 'w') as f:
        f.write(report)
        
    print("Report generated.")

if __name__ == "__main__":
    main()

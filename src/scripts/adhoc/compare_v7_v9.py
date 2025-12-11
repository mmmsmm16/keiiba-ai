import json
import os
import pandas as pd

def load_report(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def compare():
    v7_path = "experiments/v7_ensemble_full/reports/optimization_report.json"
    v9_path = "experiments/v9_ensemble_full/reports/optimization_report.json"

    r7 = load_report(v7_path)
    r9 = load_report(v9_path)

    print("--- Option C Comparison ---")
    c7 = next((x for x in r7.get('option_c', []) if x['type'] == 'option_c_total'), None)
    c9 = next((x for x in r9.get('option_c', []) if x['type'] == 'option_c_total'), None)

    if c7: print(f"v7 Option C: ROI {c7['roi']:.2f}% (Profit: {c7['total_return'] - c7['total_bet']:.0f})")
    else: print("v7 Option C: Not found")
    
    if c9: print(f"v9 Option C: ROI {c9['roi']:.2f}% (Profit: {c9['total_return'] - c9['total_bet']:.0f})")
    else: print("v9 Option C: Not found")

    print("\n--- Best Strategy Comparison (Top 1) ---")
    b7 = r7.get('best_strategies', [])
    b9 = r9.get('best_strategies', [])

    if b7: 
        top7 = b7[0]
        print(f"v7 Best: {top7['name']} - ROI {top7['roi']:.2f}%")
    else: print("v7 Best: None")

    if b9: 
        top9 = b9[0]
        print(f"v9 Best: {top9['name']} - ROI {top9['roi']:.2f}%")
    else: print("v9 Best: None")

    print("\n--- Summary Stats ---")
    # Quick aggregate of all strategies (Tansho/Sanrentan)
    # Just grab a standard one, e.g. Sanrentan 1-head, 5-opps
    
    def get_strat(report, name_part):
        for k in ['sanrentan', 'umaren', 'tansho']:
            for s in report.get(k, []):
                if name_part in s['name']:
                    return s
        return None

    s7 = get_strat(r7, "相手5頭")
    s9 = get_strat(r9, "相手5頭")
    
    if s7: print(f"v7 Sanrentan (5 opps): ROI {s7['roi']:.2f}%")
    if s9: print(f"v9 Sanrentan (5 opps): ROI {s9['roi']:.2f}%")

if __name__ == "__main__":
    compare()

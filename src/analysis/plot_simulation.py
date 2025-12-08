
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    parser = argparse.ArgumentParser(description='Plot Simulation History')
    parser.add_argument('--file', type=str, required=True, help='Path to simulation history csv')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return
        
    df = pd.read_csv(args.file)
    
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(df)), df['bankroll'], label='Bankroll')
    
    # Add trend line? No, just raw history
    plt.title(f"Bankroll Growth ({os.path.basename(args.file)})")
    plt.xlabel("Bets / Time")
    plt.ylabel("Bankroll (Yen)")
    plt.grid(True)
    plt.legend()
    
    output_path = args.file.replace('.csv', '.png')
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()

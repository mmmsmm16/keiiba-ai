"""
NAR Bankroll Simulation with Fixed Initial Capital

Simulates betting with a fixed initial capital (100,000 JPY) and compound growth.
Bets are limited by current bankroll, not unlimited.
"""
import os
import re
import pandas as pd
from datetime import datetime

# Configuration
INITIAL_BANKROLL = 100000  # 初期資金
MAX_BET_PER_RACE = 3000    # レースごとの最大賭け額
DAILY_BUDGET_RATIO = 0.1  # 1日あたり資金の10%までベット可能

def parse_daily_report(filepath):
    """Parse a daily report and return invest/return"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    invest_match = re.search(r'Invest:\s*([\d,\.]+)', content)
    return_match = re.search(r'Return:\s*([\d,\.]+)', content)
    
    invest = float(invest_match.group(1).replace(',', '')) if invest_match else 0
    ret = float(return_match.group(1).replace(',', '')) if return_match else 0
    
    return invest, ret

def simulate_bankroll():
    """Run bankroll simulation using existing daily reports"""
    report_dir = 'reports/nar/daily'
    
    bankroll = INITIAL_BANKROLL
    history = []
    
    # Process reports in date order
    report_files = sorted([f for f in os.listdir(report_dir) if f.endswith('_report.md')])
    
    for fname in report_files:
        date_str = fname[:8]
        fpath = os.path.join(report_dir, fname)
        
        orig_invest, orig_return = parse_daily_report(fpath)
        
        if orig_invest == 0:
            continue
            
        # Calculate daily budget based on current bankroll
        daily_budget = min(bankroll * DAILY_BUDGET_RATIO, bankroll)
        
        # Scale bets to fit within daily budget
        if orig_invest > daily_budget:
            scale_factor = daily_budget / orig_invest
        else:
            scale_factor = 1.0
        
        actual_invest = orig_invest * scale_factor
        actual_return = orig_return * scale_factor
        
        # Ensure we don't bet more than we have
        if actual_invest > bankroll:
            actual_invest = bankroll
            actual_return = (bankroll / orig_invest) * orig_return
        
        # Update bankroll
        profit = actual_return - actual_invest
        new_bankroll = bankroll + profit
        
        history.append({
            'date': date_str,
            'bankroll_before': bankroll,
            'invest': actual_invest,
            'return': actual_return,
            'profit': profit,
            'bankroll_after': new_bankroll,
            'roi': (actual_return / actual_invest * 100) if actual_invest > 0 else 0
        })
        
        bankroll = new_bankroll
        
        # Stop if bankrupt
        if bankroll <= 0:
            print(f"Bankrupt on {date_str}")
            break
    
    return pd.DataFrame(history)

def main():
    print("=" * 70)
    print("NAR Bankroll Simulation (Fixed Initial Capital)")
    print("=" * 70)
    print(f"Initial Bankroll: {INITIAL_BANKROLL:,} JPY")
    print(f"Daily Budget Ratio: {DAILY_BUDGET_RATIO * 100:.0f}% of bankroll")
    print(f"Max Bet per Race: {MAX_BET_PER_RACE:,} JPY")
    print()
    
    df = simulate_bankroll()
    
    if df.empty:
        print("No data to simulate.")
        return
    
    # Summary
    final_bankroll = df['bankroll_after'].iloc[-1]
    total_profit = final_bankroll - INITIAL_BANKROLL
    total_roi = (final_bankroll / INITIAL_BANKROLL) * 100
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Period: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(f"Days with Bets: {len(df)}")
    print(f"Initial Bankroll: {INITIAL_BANKROLL:,} JPY")
    print(f"Final Bankroll: {final_bankroll:,.0f} JPY")
    print(f"Total Profit: {total_profit:+,.0f} JPY")
    print(f"Total Return: {total_roi:.2f}%")
    print()
    
    # Monthly summary
    df['month'] = df['date'].str[:6]
    monthly = df.groupby('month').agg({
        'invest': 'sum',
        'return': 'sum',
        'profit': 'sum',
        'bankroll_after': 'last'
    }).reset_index()
    
    print("Monthly Summary:")
    print("-" * 70)
    for _, row in monthly.iterrows():
        month_roi = (row['return'] / row['invest'] * 100) if row['invest'] > 0 else 0
        print(f"{row['month']}: Invest {row['invest']:>10,.0f} | Return {row['return']:>10,.0f} | "
              f"Profit {row['profit']:>+10,.0f} | Bankroll {row['bankroll_after']:>12,.0f} | ROI {month_roi:>6.1f}%")
    
    # Save results
    output_path = 'reports/nar/backtest_2025_bankroll.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# NAR 2025 バックテストレポート (資金管理版)\n\n")
        f.write("## 概要\n")
        f.write("- **期間**: {} ~ {}\n".format(df['date'].iloc[0], df['date'].iloc[-1]))
        f.write("- **モデル**: NAR v2 (Isotonic Calibration適用)\n")
        f.write("- **券種**: 単勝・複勝・馬連\n\n")
        
        f.write("## 資金管理設定\n")
        f.write("| パラメータ | 値 |\n|---|---|\n")
        f.write(f"| 初期資金 | {INITIAL_BANKROLL:,} 円 |\n")
        f.write(f"| 1日あたりの予算比率 | 資金の {DAILY_BUDGET_RATIO*100:.0f}% |\n")
        f.write(f"| レースごとの最大賭け額 | {MAX_BET_PER_RACE:,} 円 |\n\n")
        
        f.write("## 結果サマリー\n\n")
        f.write("| 項目 | 値 |\n|---|---|\n")
        f.write(f"| 賭け日数 | **{len(df)}日** |\n")
        f.write(f"| 初期資金 | {INITIAL_BANKROLL:,} 円 |\n")
        f.write(f"| 最終資金 | **{final_bankroll:,.0f} 円** |\n")
        f.write(f"| 純利益 | **{total_profit:+,.0f} 円** |\n")
        f.write(f"| **総リターン** | **{total_roi:.2f}%** |\n\n")
        
        f.write("## 月別推移\n\n")
        f.write("| 月 | 投資額 | 回収額 | 利益 | 残高 | ROI |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in monthly.iterrows():
            month_roi = (row['return'] / row['invest'] * 100) if row['invest'] > 0 else 0
            f.write(f"| {row['month']} | {row['invest']:,.0f} | {row['return']:,.0f} | "
                   f"{row['profit']:+,.0f} | {row['bankroll_after']:,.0f} | {month_roi:.1f}% |\n")
        
        f.write("\n---\n*Generated: {}*\n".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
    
    print()
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    main()

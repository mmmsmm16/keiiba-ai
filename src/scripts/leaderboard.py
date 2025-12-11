"""
実験リーダーボード
- 全モデルの性能を比較
- 詳細な指標を表示
"""
import os
import sys
import json
import glob
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from tabulate import tabulate

# パス設定
script_dir = os.path.dirname(os.path.abspath(__file__))
# src/scripts/leaderboard.py から project_root へ
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_experiments():
    """全実験ディレクトリを検索"""
    experiments_dir = os.path.join(project_root, 'experiments')
    exp_dirs = []
    
    if os.path.exists(experiments_dir):
        for name in os.listdir(experiments_dir):
            exp_path = os.path.join(experiments_dir, name)
            if os.path.isdir(exp_path):
                models_dir = os.path.join(exp_path, 'models')
                reports_dir = os.path.join(exp_path, 'reports')
                if os.path.exists(models_dir) or os.path.exists(reports_dir):
                    exp_dirs.append({
                        'name': name,
                        'path': exp_path
                    })
    
    return sorted(exp_dirs, key=lambda x: x['name'])

def load_experiment_metrics(exp):
    """実験のメトリクスを読み込み"""
    metrics_path = os.path.join(exp['path'], 'reports', 'metrics.json')
    result = {
        'name': exp['name'],
        'roi': None,
        'accuracy': None,
        'place_rate': None,
        'bets': None,
        'model_type': None,
        'description': None,
        'status': 'unknown'
    }
    
    # metrics.jsonを読み込み
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            
            result['model_type'] = data.get('model_type', 'unknown')
            
            # メトリクスから最良の結果を取得
            metrics = data.get('metrics', {})
            strategies = data.get('strategies', {})
            
            # 主モデルの結果を探す
            for key in ['Ensemble', 'ROI', 'roi', 'lgbm', 'catboost', 'tabnet']:
                if key in metrics:
                    result['roi'] = metrics[key].get('roi')
                    result['accuracy'] = metrics[key].get('accuracy')
                    result['place_rate'] = metrics[key].get('place_rate')
                    result['bets'] = metrics[key].get('bets')
                    break
            
            # 戦略結果も確認
            if result['roi'] is None and 'max_score' in strategies:
                result['roi'] = strategies['max_score'].get('roi')
                result['accuracy'] = strategies['max_score'].get('accuracy')
                result['place_rate'] = strategies['max_score'].get('place_rate')
                result['bets'] = strategies['max_score'].get('bets')
            
            # Accuracy/PlaceRateが比率(0-1)の場合はパーセント(0-100)に変換
            if result['accuracy'] is not None and result['accuracy'] <= 1.0:
                result['accuracy'] *= 100
            
            if result['place_rate'] is not None and result['place_rate'] <= 1.0:
                result['place_rate'] *= 100
            
            result['status'] = 'completed'
        except Exception as e:
            result['status'] = f'error: {e}'
    else:
        # モデルファイルの存在確認
        models_dir = os.path.join(exp['path'], 'models')
        if os.path.exists(models_dir):
            model_files = os.listdir(models_dir)
            if model_files:
                result['status'] = 'training_done'
            else:
                result['status'] = 'no_models'
        else:
            result['status'] = 'not_started'
    
    # 設定ファイルから説明を取得
    config_candidates = glob.glob(os.path.join(project_root, 'config', 'experiments', f'exp_{exp["name"]}*.yaml'))
    if config_candidates:
        try:
            with open(config_candidates[0], 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            result['description'] = config.get('description', '')
            result['model_type'] = config.get('model', {}).get('type', result['model_type'])
        except:
            pass
    
    return result

def generate_leaderboard():
    """リーダーボードを生成"""
    print("\n" + "="*100)
    print("🏆 モデル実験リーダーボード")
    print("="*100)
    print(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    experiments = find_experiments()
    
    if not experiments:
        print("実験が見つかりません。")
        return
    
    results = []
    for exp in experiments:
        metrics = load_experiment_metrics(exp)
        results.append(metrics)
    
    # DataFrameに変換
    df = pd.DataFrame(results)
    
    # ROIでソート
    df['roi_val'] = pd.to_numeric(df['roi'], errors='coerce')
    df = df.sort_values('roi_val', ascending=False, na_position='last')
    
    # 表示用に整形
    display_data = []
    for i, row in df.iterrows():
        roi_str = f"{row['roi']:.1f}%" if pd.notna(row['roi']) else "-"
        acc_str = f"{row['accuracy']:.1f}%" if pd.notna(row['accuracy']) else "-"
        place_str = f"{row['place_rate']:.1f}%" if pd.notna(row['place_rate']) else "-"
        bets_str = f"{int(row['bets']):,}" if pd.notna(row['bets']) else "-"
        
        display_data.append({
            '順位': len(display_data) + 1 if pd.notna(row['roi']) else '-',
            'モデル': row['name'],
            'ROI': roi_str,
            '的中率': acc_str,
            '複勝率': place_str,
            'ベット数': bets_str,
            'タイプ': row['model_type'] or '-',
            'ステータス': row['status']
        })
    
    # テーブル表示
    print(tabulate(display_data, headers='keys', tablefmt='grid'))
    
    # 完了済みのみで最良モデルを表示
    completed = df[df['status'] == 'completed'].copy()
    if not completed.empty:
        best = completed.iloc[0]
        print(f"\n🥇 ベストモデル: {best['name']} (ROI: {best['roi']:.1f}%, 的中率: {best['accuracy']:.1f}%)")
    
    # CSVに保存
    output_path = os.path.join(project_root, 'reports', 'leaderboard.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.drop(columns=['roi_val'], inplace=True, errors='ignore')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n📄 リーダーボードを {output_path} に保存しました")
    
    # 詳細レポート
    print("\n" + "-"*100)
    print("📊 詳細レポート")
    print("-"*100)
    
    for i, row in df.iterrows():
        if row['status'] == 'completed' and pd.notna(row['roi']):
            profit = (row['roi'] / 100 - 1) * row['bets'] * 100 if pd.notna(row['bets']) else 0
            print(f"\n{row['name']}:")
            print(f"  ROI: {row['roi']:.1f}%")
            print(f"  的中率: {row['accuracy']:.1f}%")
            print(f"  ベット数: {int(row['bets']):,}")
            print(f"  推定利益: {profit:+,.0f}円")
            if row['description']:
                print(f"  説明: {row['description']}")

def main():
    generate_leaderboard()
    print("\n✅ リーダーボード生成完了!")

if __name__ == "__main__":
    main()

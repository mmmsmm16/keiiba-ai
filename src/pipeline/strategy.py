import os
import pandas as pd
import numpy as np
import logging
import json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from itertools import combinations, permutations
from src.pipeline.config import ExperimentConfig
from src.pipeline.evaluate import load_payout_data, NpEncoder

logger = logging.getLogger(__name__)

def optimize_strategies(config: ExperimentConfig, run_dir: str):
    """
    最適な馬券戦略を探索し、レポートを出力します。

    Args:
        config (ExperimentConfig): 実験設定オブジェクト
        run_dir (str): 実験出力ディレクトリ
    """
    logger.info("戦略最適化プロセスを開始します...")
    
    if not config.strategy.enabled:
        logger.info("戦略最適化は設定で無効化されています。")
        return

    # 予測データのロード
    reports_dir = os.path.join(run_dir, "reports")
    pred_path = os.path.join(reports_dir, "predictions.parquet")
    
    if not os.path.exists(pred_path):
        logger.error(f"予測ファイルが見つかりません: {pred_path}。評価ステップを先に実行してください。")
        return
    
    logger.info(f"予測データをロード中: {pred_path}")
    df = pd.read_parquet(pred_path)
    
    # JRAフィルター（評価時にNARを除外）
    # evaluate.pyからの出力にはすでにJRAフィルターが適用されているが、念のため再チェック
    if 'venue' in df.columns:
        jra_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        df['venue_code'] = df['venue'].astype(str).str[:2]
        before_count = len(df)
        df = df[df['venue_code'].isin(jra_codes)].copy()
        if before_count > len(df):
            logger.info(f"🏇 JRA Only Filter: {before_count} -> {len(df)} rows")
    
    # 払戻データのロード
    years = df['year'].unique().tolist()
    payout_df = load_payout_data(years)
    
    if payout_df.empty:
        logger.error("払戻データが取得できませんでした。戦略最適化をスキップします。")
        return

    # 払戻マップの構築
    payout_map = build_payout_map(payout_df)
    
    results = {
        'tansho': [],
        'umaren': [],
        'sanrentan': [],
        'option_c': [],  # Option C戦略を追加
        'best_strategies': []
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        
        # 1. 単勝最適化 (EV閾値探索)
        if "tansho" in config.strategy.target_bet_types:
            task = progress.add_task("[cyan]単勝戦略を最適化中...", total=len(np.arange(0.5, 3.0, 0.1)))
            results['tansho'] = optimize_tansho(df, progress, task)
            
        # 2. 馬連/三連単最適化 (Box/流し)
        if "umaren" in config.strategy.target_bet_types:
            # Box 3-6 -> 4 iterations
            task = progress.add_task("[green]馬連戦略を最適化中...", total=4)
            results['umaren'] = optimize_umaren(df, payout_map, progress, task)

        if "sanrentan" in config.strategy.target_bet_types:
            # Formation 4-8 -> 5 iterations
            task = progress.add_task("[magenta]三連単戦略を最適化中...", total=5)
            results['sanrentan'] = optimize_sanrentan(df, payout_map, progress, task)
        
        # 3. Option C戦略評価 (v7最適化戦略)
        task = progress.add_task("[yellow]Option C戦略を評価中...", total=1)
        results['option_c'] = evaluate_option_c(df, payout_map)
        progress.advance(task)
        
    # ベスト戦略の抽出
    all_res = []
    for k in ['tansho', 'umaren', 'sanrentan', 'option_c']:
        all_res.extend(results[k])
        
    # 高ROIかつ一定数以上の投票がある戦略を抽出
    high_roi = [r for r in all_res if r['roi'] >= config.strategy.min_roi and r['bet_count'] > 10] 
    
    # 純利益順にソート
    high_roi.sort(key=lambda x: x['total_return'] - x['total_bet'], reverse=True)
    results['best_strategies'] = high_roi[:10]
    
    # レポート保存
    out_path = os.path.join(reports_dir, "optimization_report.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4, cls=NpEncoder)
        
    logger.info(f"戦略最適化が完了しました。レポート: {out_path}")
    if high_roi:
        best = high_roi[0]
        # Rich print could be nicer here too but logger is fine
        logger.info(f"🏆 最良戦略: {best['name']} (ROI: {best['roi']:.1f}%, 純利益: +{best['total_return']-best['total_bet']:.0f}円)")

def build_payout_map(payout_df: pd.DataFrame) -> dict:
    """払戻データを高速検索用に辞書形式に変換します。"""
    p_map = {}
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        p_map[rid] = {'umaren': {}, 'sanrenpuku': {}, 'sanrentan': {}, 'tansho': {}}
        
        # 馬連
        for i in range(1, 4):
            if row.get(f'haraimodoshi_umaren_{i}a'):
                 try:
                    comb = str(row[f'haraimodoshi_umaren_{i}a'])
                    pay = int(row[f'haraimodoshi_umaren_{i}b'])
                    p_map[rid]['umaren'][comb] = pay
                 except: pass

        # 三連単
        for i in range(1, 7):
            if row.get(f'haraimodoshi_sanrentan_{i}a'):
                 try:
                    comb = str(row[f'haraimodoshi_sanrentan_{i}a'])
                    pay = int(row[f'haraimodoshi_sanrentan_{i}b'])
                    p_map[rid]['sanrentan'][comb] = pay
                 except: pass
        
        # 単勝
        if row.get('haraimodoshi_tansho_1a'):
            try:
                horse_num = str(int(row['haraimodoshi_tansho_1a'])).zfill(2)
                pay = int(row['haraimodoshi_tansho_1b'])
                p_map[rid]['tansho'][horse_num] = pay
            except: pass
    return p_map

def optimize_tansho(df: pd.DataFrame, progress, task_id) -> list:
    """単勝の期待値閾値をグリッドサーチします。"""
    res = []
    thresholds = np.arange(0.5, 3.0, 0.1)
    
    for th in thresholds:
        th = round(th, 2)
        bets = df[df['expected_value'] >= th]
        
        if not bets.empty:
            total_bet = len(bets) * 100
            total_ret = bets[bets['rank'] == 1]['odds'].sum() * 100
            roi = total_ret / total_bet * 100 if total_bet > 0 else 0
            
            res.append({
                'name': f"単勝 (EV >= {th})",
                'type': 'tansho',
                'params': {'ev_threshold': th},
                'bet_count': len(bets),
                'total_bet': total_bet,
                'total_return': total_ret,
                'roi': roi
            })
        progress.advance(task_id)
        
    return res

def optimize_umaren(df: pd.DataFrame, payout_map: dict, progress, task_id) -> list:
    """馬連BOX戦略を最適化します。"""
    res = []
    
    # 1. Box戦略 (スコア上位 N 頭 Box)
    for n in range(3, 7): # Box 3-6
        stats = {'bet': 0, 'return': 0, 'count': 0}
        
        # Groupby is somewhat slow, but acceptable for thousands of races
        for race_id, group in df.groupby('race_id'):
            if race_id not in payout_map: continue
            
            top = group.sort_values('score', ascending=False).head(n)
            if len(top) < 2: continue
            
            nums = top['horse_number'].astype(int).tolist()
            combos = list(combinations(nums, 2))
            
            stats['bet'] += len(combos) * 100
            stats['count'] += 1
            
            race_payouts = payout_map[race_id]['umaren']
            for c in combos:
                c_sorted = sorted(c)
                c_str = f"{c_sorted[0]:02}{c_sorted[1]:02}"
                if c_str in race_payouts:
                    stats['return'] += race_payouts[c_str]

        roi = stats['return'] / stats['bet'] * 100 if stats['bet'] > 0 else 0
        res.append({
            'name': f"馬連 Box {n}頭",
            'type': 'umaren_box',
            'params': {'n': n},
            'bet_count': stats['bet'] // 100,
            'total_bet': stats['bet'],
            'total_return': stats['return'],
            'roi': roi
        })
        progress.advance(task_id)
        
    return res

def optimize_sanrentan(df: pd.DataFrame, payout_map: dict, progress, task_id) -> list:
    """三連単フォーメーション戦略を最適化します。"""
    res = []
    
    # 1. フォーメーション (軸1頭 -> 相手N頭 マルチ相当)
    
    for n_opps in range(4, 9): # 相手4〜8頭
        stats = {'bet': 0, 'return': 0, 'count': 0}
        
        for race_id, group in df.groupby('race_id'):
            if race_id not in payout_map: continue
            
            sorted_horses = group.sort_values('score', ascending=False)
            if len(sorted_horses) < n_opps + 1: continue
            
            axis = sorted_horses.iloc[0]
            opps = sorted_horses.iloc[1:n_opps+1]['horse_number'].astype(int).tolist()
            axis_num = int(axis['horse_number'])
            
            perms = list(permutations(opps, 2)) 
            
            stats['bet'] += len(perms) * 100
            stats['count'] += 1
            
            race_payouts = payout_map[race_id]['sanrentan']
            
            for p in perms:
                comb_str = f"{axis_num:02}{p[0]:02}{p[1]:02}"
                if comb_str in race_payouts:
                    stats['return'] += race_payouts[comb_str]
                    
        roi = stats['return'] / stats['bet'] * 100 if stats['bet'] > 0 else 0
        res.append({
            'name': f"三連単1着流し (相手{n_opps}頭)",
            'type': 'sanrentan_form',
            'params': {'axis_type': 'score_top1', 'n_opps': n_opps},
            'bet_count': stats['bet'] // 100,
            'total_bet': stats['bet'],
            'total_return': stats['return'],
            'roi': roi
        })
        progress.advance(task_id)

    return res


def evaluate_option_c(df: pd.DataFrame, payout_map: dict) -> list:
    """
    Option C戦略を評価します。
    
    戦略ロジック:
    - 7番人気以上 → 三連単1頭軸4頭
    - 接戦(gap<0.3) → 三連単1頭軸4頭
    - その他 → 単勝
    
    2025年v7での実績: ROI 147%
    """
    res = []
    
    stats_total = {'bet': 0, 'return': 0, 'races': 0}
    stats_sanrentan = {'bet': 0, 'return': 0, 'races': 0}
    stats_tansho = {'bet': 0, 'return': 0, 'races': 0}
    
    for race_id, group in df.groupby('race_id'):
        if race_id not in payout_map:
            continue
        
        sorted_horses = group.sort_values('score', ascending=False)
        if len(sorted_horses) < 6:
            continue
        
        # Top1馬情報
        h = sorted_horses['horse_number'].astype(int).tolist()
        top1 = sorted_horses.iloc[0]
        pop = int(top1['popularity']) if pd.notna(top1.get('popularity', np.nan)) else 99
        
        # スコア差
        scores = sorted_horses['score'].head(6).values
        gap = scores[0] - scores[5] if len(scores) >= 6 else 0.5
        
        if pop >= 7 or gap < 0.3:
            # 三連単1頭軸4頭 (6点)
            axis = h[0]
            opps = h[1:4]
            perms = list(permutations(opps, 2))
            
            cost = len(perms) * 100
            ret = 0
            for p in perms:
                comb_str = f"{axis:02}{p[0]:02}{p[1]:02}"
                ret += payout_map[race_id]['sanrentan'].get(comb_str, 0)
            
            stats_sanrentan['bet'] += cost
            stats_sanrentan['return'] += ret
            stats_sanrentan['races'] += 1
        else:
            # 単勝
            axis = h[0]
            cost = 100
            ret = payout_map[race_id]['tansho'].get(f"{axis:02}", 0)
            
            stats_tansho['bet'] += cost
            stats_tansho['return'] += ret
            stats_tansho['races'] += 1
        
        stats_total['bet'] += cost
        stats_total['return'] += ret
        stats_total['races'] += 1
    
    # 総合結果
    roi_total = stats_total['return'] / stats_total['bet'] * 100 if stats_total['bet'] > 0 else 0
    res.append({
        'name': 'Option C (統合戦略)',
        'type': 'option_c_total',
        'params': {'strategy': '7番人気以上/接戦→三連単, その他→単勝'},
        'bet_count': stats_total['races'],
        'total_bet': stats_total['bet'],
        'total_return': stats_total['return'],
        'roi': roi_total
    })
    
    # 三連単部分
    roi_sanrentan = stats_sanrentan['return'] / stats_sanrentan['bet'] * 100 if stats_sanrentan['bet'] > 0 else 0
    res.append({
        'name': 'Option C (三連単部分)',
        'type': 'option_c_sanrentan',
        'params': {'condition': '7番人気以上 or 接戦'},
        'bet_count': stats_sanrentan['races'],
        'total_bet': stats_sanrentan['bet'],
        'total_return': stats_sanrentan['return'],
        'roi': roi_sanrentan
    })
    
    # 単勝部分
    roi_tansho = stats_tansho['return'] / stats_tansho['bet'] * 100 if stats_tansho['bet'] > 0 else 0
    res.append({
        'name': 'Option C (単勝部分)',
        'type': 'option_c_tansho',
        'params': {'condition': 'その他'},
        'bet_count': stats_tansho['races'],
        'total_bet': stats_tansho['bet'],
        'total_return': stats_tansho['return'],
        'roi': roi_tansho
    })
    
    logger.info(f"📊 Option C評価完了: 総合ROI {roi_total:.1f}% (三連単部分 {roi_sanrentan:.1f}%, 単勝部分 {roi_tansho:.1f}%)")
    
    return res

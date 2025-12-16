"""
Phase 5: Purchase Parameter Optimizer (2-Stage Search)
購入パラメータの2段階探索と最適化

Usage (in container):
    docker compose exec app python src/backtest/purchase_optimizer.py --period screening --stage 1
    docker compose exec app python src/backtest/purchase_optimizer.py --period verification --stage 2
"""

import sys
import os
import argparse
import logging
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.period_guard import add_period_args, parse_period_args, filter_dataframe_by_period, PeriodConfig
from betting.purchase_model import PurchaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """購入戦略設定"""
    ev_threshold: float = 0.0
    min_odds: float = None
    max_odds: float = None
    kelly_fraction: float = 0.10
    max_bet_pct: float = 0.05
    no_bet_prob_threshold: float = None  # top1_prob < this → don't bet
    bankroll: float = 100000
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def name(self) -> str:
        """戦略名を生成"""
        parts = [f"ev{self.ev_threshold:.2f}"]
        if self.min_odds:
            parts.append(f"minO{self.min_odds}")
        if self.max_odds:
            parts.append(f"maxO{self.max_odds}")
        parts.append(f"k{self.kelly_fraction:.2f}")
        return "_".join(parts)


@dataclass
class EvaluationResult:
    """評価結果"""
    strategy_name: str
    config: Dict
    # Main metrics
    mean_roi: float
    std_roi: float
    min_fold_roi: float
    max_drawdown: float
    # Details
    fold_results: List[Dict]
    bet_count_per_race: float
    total_bets: int
    total_profit: float
    # Slippage scenarios
    roi_slippage_95: float = None
    roi_slippage_90: float = None


class PurchaseOptimizer:
    """購入パラメータ最適化器（2段階探索）"""
    
    # Stage 1: 閾値系パラメータ
    STAGE1_PARAMS = {
        'ev_threshold': [0.0, 0.05, 0.10, 0.15, 0.20],
        'min_odds': [None, 2.0, 3.0, 5.0],
        'max_odds': [None, 50, 100],
    }
    
    # Stage 2: 資金配分系パラメータ
    STAGE2_PARAMS = {
        'kelly_fraction': [0.05, 0.10, 0.15, 0.20],
        'max_bet_pct': [0.02, 0.05, 0.10],
        'no_bet_prob_threshold': [None, 0.15, 0.25],
    }
    
    # 採用判定の足切り基準
    ACCEPTANCE_CRITERIA = {
        'min_fold_roi': 50.0,  # > 50%
        'max_drawdown': 30.0,  # < 30%
    }
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.pm = PurchaseModel()
        self.results: List[EvaluationResult] = []
        self.slippage_factors = [1.0, 0.95, 0.90]
    
    def load_data(self, period: PeriodConfig):
        """データロード"""
        logger.info(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        
        # 年カラム追加
        if 'year' not in self.df.columns:
            self.df['year'] = self.df['race_id'].astype(str).str[:4].astype(int)
        
        # オッズカラム
        self.odds_col = 'odds' if 'odds' in self.df.columns else 'win_odds'
        
        # オッズ有効行のみ
        valid_mask = self.df[self.odds_col].notna() & (self.df[self.odds_col] > 0)
        self.df = self.df[valid_mask].copy()
        
        # 期間フィルタ
        self.df = filter_dataframe_by_period(self.df, period)
        
        # 確率カラム確認
        if 'prob' not in self.df.columns:
            # モデル予測が入っていない場合はエラー
            # デモ用に p_market を使用
            logger.warning("No 'prob' column found. Using p_market as proxy.")
            self.df = self.pm.calculate_market_probability(self.df, odds_col=self.odds_col)
            self.df['prob'] = self.df['p_market']
        
        # ターゲット（1着か否か）
        if 'rank' in self.df.columns:
            self.df['is_winner'] = (self.df['rank'] == 1).astype(int)
        elif 'finish_position' in self.df.columns:
            self.df['is_winner'] = (self.df['finish_position'] == 1).astype(int)
        else:
            # rankがない場合のフォールバック
            logger.warning("No rank column. Cannot calculate ROI properly.")
            self.df['is_winner'] = 0
        
        logger.info(f"Data loaded: {len(self.df):,} valid rows, "
                   f"{self.df['race_id'].nunique():,} races")
    
    def evaluate_strategy(
        self,
        config: StrategyConfig,
        df: pd.DataFrame,
        slippage: float = 1.0
    ) -> Dict:
        """単一戦略の評価"""
        eval_df = df.copy()
        
        # Slippage適用
        eval_df['odds_effective'] = eval_df[self.odds_col] * slippage
        
        # フィルタ適用
        mask = pd.Series(True, index=eval_df.index)
        
        if config.min_odds is not None:
            mask &= eval_df[self.odds_col] >= config.min_odds
        if config.max_odds is not None:
            mask &= eval_df[self.odds_col] <= config.max_odds
        if config.no_bet_prob_threshold is not None:
            # レースごとのtop1_probを計算
            top1_prob = eval_df.groupby('race_id')['prob'].transform('max')
            mask &= top1_prob >= config.no_bet_prob_threshold
        
        eval_df = eval_df[mask].copy()
        
        if len(eval_df) == 0:
            return {'roi': 0, 'bet_count': 0, 'profit': 0}
        
        # EV計算
        eval_df = self.pm.calculate_expected_value(
            eval_df, prob_col='prob', odds_col='odds_effective'
        )
        
        # EV閾値フィルタ
        eval_df = eval_df[eval_df['ev'] > config.ev_threshold].copy()
        
        if len(eval_df) == 0:
            return {'roi': 0, 'bet_count': 0, 'profit': 0}
        
        # Kelly配分
        eval_df = self.pm.apply_betting_strategy(
            eval_df,
            strategy_name='kelly',
            bankroll=config.bankroll,
            fraction=config.kelly_fraction,
            max_bet_rate=config.max_bet_pct,
            prob_col='prob',
            odds_col='odds_effective'
        )
        
        # ROI計算
        total_bet = eval_df['bet_amount'].sum()
        total_return = (eval_df[eval_df['is_winner'] == 1]['bet_amount'] * 
                       eval_df[eval_df['is_winner'] == 1]['odds_effective']).sum()
        
        roi = (total_return / total_bet * 100) if total_bet > 0 else 0
        profit = total_return - total_bet
        
        return {
            'roi': roi,
            'bet_count': len(eval_df[eval_df['bet_amount'] > 0]),
            'profit': profit,
            'total_bet': total_bet,
            'total_return': total_return
        }
    
    def evaluate_walk_forward(
        self,
        config: StrategyConfig,
        fold_years: List[int]
    ) -> EvaluationResult:
        """Walk-Forward評価"""
        fold_results = []
        cumulative_profit = []
        
        for year in fold_years:
            year_df = self.df[self.df['year'] == year].copy()
            
            # 各slippageで評価
            result_1_0 = self.evaluate_strategy(config, year_df, slippage=1.0)
            result_0_95 = self.evaluate_strategy(config, year_df, slippage=0.95)
            result_0_90 = self.evaluate_strategy(config, year_df, slippage=0.90)
            
            fold_results.append({
                'year': year,
                'roi_1_0': result_1_0['roi'],
                'roi_0_95': result_0_95['roi'],
                'roi_0_90': result_0_90['roi'],
                'bet_count': result_1_0['bet_count'],
                'profit': result_1_0['profit'],
                'n_races': year_df['race_id'].nunique()
            })
            cumulative_profit.append(result_1_0['profit'])
        
        # 集計
        rois = [f['roi_1_0'] for f in fold_results]
        mean_roi = np.mean(rois) if rois else 0
        std_roi = np.std(rois) if len(rois) > 1 else 0
        min_fold_roi = min(rois) if rois else 0
        
        # Max Drawdown計算
        cumsum = np.cumsum(cumulative_profit)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        max_dd = (max(drawdowns) / config.bankroll * 100) if len(drawdowns) > 0 else 0
        
        # Slippage別平均
        roi_s95 = np.mean([f['roi_0_95'] for f in fold_results])
        roi_s90 = np.mean([f['roi_0_90'] for f in fold_results])
        
        # BetCount/Race
        total_bets = sum(f['bet_count'] for f in fold_results)
        total_races = sum(f['n_races'] for f in fold_results)
        bet_per_race = total_bets / total_races if total_races > 0 else 0
        
        return EvaluationResult(
            strategy_name=config.name(),
            config=config.to_dict(),
            mean_roi=mean_roi,
            std_roi=std_roi,
            min_fold_roi=min_fold_roi,
            max_drawdown=max_dd,
            fold_results=fold_results,
            bet_count_per_race=bet_per_race,
            total_bets=total_bets,
            total_profit=sum(f['profit'] for f in fold_results),
            roi_slippage_95=roi_s95,
            roi_slippage_90=roi_s90
        )
    
    def run_stage1_search(self, screening_year: int = 2024, top_k: int = 5) -> List[StrategyConfig]:
        """Stage 1: 閾値系パラメータの粗探索"""
        logger.info("=== Stage 1: Threshold Parameter Search ===")
        
        # 固定パラメータ
        fixed = {'kelly_fraction': 0.10, 'max_bet_pct': 0.05}
        
        # グリッド生成
        param_grid = list(product(
            self.STAGE1_PARAMS['ev_threshold'],
            self.STAGE1_PARAMS['min_odds'],
            self.STAGE1_PARAMS['max_odds']
        ))
        
        logger.info(f"Total combinations: {len(param_grid)}")
        
        results = []
        year_df = self.df[self.df['year'] == screening_year].copy()
        
        for ev_th, min_o, max_o in param_grid:
            config = StrategyConfig(
                ev_threshold=ev_th,
                min_odds=min_o,
                max_odds=max_o,
                **fixed
            )
            
            res = self.evaluate_strategy(config, year_df)
            results.append({
                'config': config,
                'roi': res['roi'],
                'bet_count': res['bet_count']
            })
        
        # ROI順でソート
        results.sort(key=lambda x: x['roi'], reverse=True)
        
        # 上位K設定を返す
        top_configs = [r['config'] for r in results[:top_k]]
        
        logger.info(f"Top {top_k} configurations (Stage 1):")
        for i, r in enumerate(results[:top_k]):
            logger.info(f"  {i+1}. ROI={r['roi']:.2f}%, Bets={r['bet_count']}, "
                       f"Config: ev={r['config'].ev_threshold}, "
                       f"minO={r['config'].min_odds}, maxO={r['config'].max_odds}")
        
        return top_configs
    
    def run_stage2_search(
        self,
        base_configs: List[StrategyConfig],
        fold_years: List[int] = [2021, 2022, 2023]
    ) -> List[EvaluationResult]:
        """Stage 2: 資金配分パラメータの微調整"""
        logger.info("=== Stage 2: Capital Allocation Parameter Search ===")
        
        all_results = []
        
        for base_config in base_configs:
            # Stage2パラメータのグリッド
            param_grid = list(product(
                self.STAGE2_PARAMS['kelly_fraction'],
                self.STAGE2_PARAMS['max_bet_pct'],
                self.STAGE2_PARAMS['no_bet_prob_threshold']
            ))
            
            for kelly_f, max_bet, no_bet_th in param_grid:
                config = StrategyConfig(
                    ev_threshold=base_config.ev_threshold,
                    min_odds=base_config.min_odds,
                    max_odds=base_config.max_odds,
                    kelly_fraction=kelly_f,
                    max_bet_pct=max_bet,
                    no_bet_prob_threshold=no_bet_th
                )
                
                result = self.evaluate_walk_forward(config, fold_years)
                all_results.append(result)
        
        logger.info(f"Total Stage 2 evaluations: {len(all_results)}")
        
        return all_results
    
    def apply_acceptance_criteria(
        self,
        results: List[EvaluationResult]
    ) -> List[EvaluationResult]:
        """足切り基準を適用"""
        accepted = []
        
        for r in results:
            if r.min_fold_roi > self.ACCEPTANCE_CRITERIA['min_fold_roi']:
                if r.max_drawdown < self.ACCEPTANCE_CRITERIA['max_drawdown']:
                    accepted.append(r)
        
        logger.info(f"Accepted after criteria: {len(accepted)}/{len(results)}")
        return accepted
    
    def select_best(self, results: List[EvaluationResult]) -> EvaluationResult:
        """最適戦略を選択（Mean ROI最大、同点ならStd最小）"""
        if not results:
            return None
        
        # Mean ROIでソート、同点ならStdでソート
        sorted_results = sorted(
            results,
            key=lambda x: (-x.mean_roi, x.std_roi)
        )
        
        return sorted_results[0]
    
    def generate_report(
        self,
        results: List[EvaluationResult],
        best: EvaluationResult,
        output_path: str
    ):
        """最適化レポート生成"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = f"""# Phase 5: Purchase Optimization Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. Acceptance Criteria

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Min Fold ROI | > {self.ACCEPTANCE_CRITERIA['min_fold_roi']}% | 最悪年の足切り |
| Max Drawdown | < {self.ACCEPTANCE_CRITERIA['max_drawdown']}% | 破産リスク足切り |

## 2. Best Strategy

| Metric | Value |
|--------|-------|
| **Strategy Name** | `{best.strategy_name if best else 'N/A'}` |
| **Mean ROI** | {best.mean_roi:.2f}% |
| **Std ROI** | {best.std_roi:.2f}% |
| **Min Fold ROI** | {best.min_fold_roi:.2f}% |
| **Max Drawdown** | {best.max_drawdown:.2f}% |
| **ROI (Slippage 0.95)** | {best.roi_slippage_95:.2f}% |
| **ROI (Slippage 0.90)** | {best.roi_slippage_90:.2f}% |
| **Bet/Race** | {best.bet_count_per_race:.2f} |

### Configuration

```yaml
{yaml.dump(best.config, default_flow_style=False) if best else 'N/A'}
```

### Fold-wise Results

| Year | ROI (1.0) | ROI (0.95) | ROI (0.90) | Bets | Profit |
|------|-----------|------------|------------|------|--------|
"""
        if best:
            for f in best.fold_results:
                report += f"| {f['year']} | {f['roi_1_0']:.2f}% | {f['roi_0_95']:.2f}% | {f['roi_0_90']:.2f}% | {f['bet_count']:,} | ¥{f['profit']:,.0f} |\n"
        
        report += f"""
## 3. Top 10 Candidates (After Acceptance Criteria)

| Rank | Strategy | Mean ROI | Std | Min Fold | Max DD |
|------|----------|----------|-----|----------|--------|
"""
        for i, r in enumerate(sorted(results, key=lambda x: -x.mean_roi)[:10]):
            report += f"| {i+1} | {r.strategy_name} | {r.mean_roi:.2f}% | {r.std_roi:.2f}% | {r.min_fold_roi:.2f}% | {r.max_drawdown:.2f}% |\n"
        
        report += """
## 4. Adoption Decision Rule

```
1. Filter: Min Fold ROI > 50%
2. Filter: Max Drawdown < 30%
3. Select: Maximum Mean ROI
4. Tiebreaker: Minimum Std
```

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")
    
    def save_optimal_config(self, best: EvaluationResult, output_path: str):
        """最適設定をYAMLで保存"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        config_data = {
            'strategy_name': best.strategy_name,
            'parameters': best.config,
            'metrics': {
                'mean_roi': best.mean_roi,
                'std_roi': best.std_roi,
                'min_fold_roi': best.min_fold_roi,
                'max_drawdown': best.max_drawdown,
                'roi_slippage_95': best.roi_slippage_95,
                'roi_slippage_90': best.roi_slippage_90
            },
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Optimal config saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Purchase Parameter Optimizer (2-Stage)"
    )
    add_period_args(parser)
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed/preprocessed_data_v11.parquet',
        help='Path to preprocessed data'
    )
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 0],
        default=0,
        help='Stage to run (1=Stage1 only, 2=Stage2 only, 0=Both)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top configs from Stage 1 to pass to Stage 2'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports',
        help='Output directory'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='config',
        help='Config output directory'
    )
    
    args = parser.parse_args()
    
    try:
        period = parse_period_args(args)
    except ValueError as e:
        logger.error(f"Period validation failed: {e}")
        sys.exit(1)
    
    optimizer = PurchaseOptimizer(args.data_path)
    optimizer.load_data(period)
    
    # フルパイプライン
    if args.stage in [0, 1]:
        # Stage 1: Screening (2024)
        logger.info("Running Stage 1 on 2024 data...")
        top_configs = optimizer.run_stage1_search(
            screening_year=2024,
            top_k=args.top_k
        )
    
    if args.stage in [0, 2]:
        # Stage 2: Walk-Forward (2021-2023)
        if args.stage == 2:
            # Stage 2のみの場合はデフォルト設定を使用
            logger.warning("Running Stage 2 only with default Stage 1 configs")
            top_configs = [
                StrategyConfig(ev_threshold=0.0),
                StrategyConfig(ev_threshold=0.05),
                StrategyConfig(ev_threshold=0.10),
            ]
        
        logger.info("Running Stage 2 Walk-Forward evaluation...")
        wf_years = [2021, 2022, 2023]
        all_results = optimizer.run_stage2_search(top_configs, wf_years)
        
        # 足切り適用
        accepted = optimizer.apply_acceptance_criteria(all_results)
        
        # 最適選択
        if accepted:
            best = optimizer.select_best(accepted)
        else:
            logger.warning("No strategy passed acceptance criteria. Using best from all.")
            best = optimizer.select_best(all_results)
        
        # レポート生成
        optimizer.generate_report(
            accepted if accepted else all_results,
            best,
            os.path.join(args.output_dir, 'phase5_win_optimization.md')
        )
        
        # 最適設定保存
        if best:
            optimizer.save_optimal_config(
                best,
                os.path.join(args.config_dir, 'optimal_win_strategy.yaml')
            )
    
    logger.info("Phase 5 optimization completed!")


if __name__ == "__main__":
    main()

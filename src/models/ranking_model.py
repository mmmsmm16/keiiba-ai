"""
Phase 6: Plackett-Luce Ranking Model with Temperature
温度付きPlackett-Luceモデルによる順位分布確率生成

Usage (in container):
    docker compose exec app python src/models/ranking_model.py --period screening --temperature 1.0
    docker compose exec app python src/models/ranking_model.py --period verification --explore_temperature
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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import log_loss

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.period_guard import add_period_args, parse_period_args, filter_dataframe_by_period, PeriodConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PlackettLuceConfig:
    """Plackett-Luce設定"""
    temperature: float = 1.0
    score_col: str = 'raw_score'  # モデルのスコア（確率ではなくraw score推奨）
    prob_col: str = 'prob'  # 確率カラム（代替用）
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PlackettLuceModel:
    """
    温度付きPlackett-Luceモデル
    
    順位確率を計算:
    - strength_i = exp(score_i / T)
    - P(horse_i wins) = strength_i / sum(strength_all)
    """
    
    def __init__(self, config: PlackettLuceConfig = None):
        self.config = config or PlackettLuceConfig()
        self.temperature = self.config.temperature
    
    def _get_scores(self, race_df: pd.DataFrame) -> np.ndarray:
        """スコアを取得（raw_score優先、なければprobを使用）"""
        if self.config.score_col in race_df.columns:
            scores = race_df[self.config.score_col].values
        elif self.config.prob_col in race_df.columns:
            # 確率をlogitに変換してスコアとして使用
            probs = np.clip(race_df[self.config.prob_col].values, 1e-7, 1 - 1e-7)
            scores = np.log(probs / (1 - probs))  # logit
        else:
            raise ValueError(f"Neither '{self.config.score_col}' nor '{self.config.prob_col}' found")
        
        return scores
    
    def compute_win_probabilities(self, race_df: pd.DataFrame) -> np.ndarray:
        """
        レース内の勝利確率を計算（温度付きsoftmax）
        
        P(i wins) = exp(score_i / T) / sum(exp(score_j / T))
        """
        scores = self._get_scores(race_df)
        
        # 温度付きスケーリング
        scaled_scores = scores / self.temperature
        
        # Numerical stability: subtract max
        scaled_scores = scaled_scores - np.max(scaled_scores)
        
        # Softmax
        exp_scores = np.exp(scaled_scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
    
    def compute_position_probabilities(
        self,
        race_df: pd.DataFrame,
        n_positions: int = 3
    ) -> Dict[int, np.ndarray]:
        """
        各馬の1着〜n_positions着になる確率を計算
        
        Plackett-Luce再帰:
        P(i is 2nd | j is 1st) = strength_i / (sum_strengths - strength_j)
        """
        scores = self._get_scores(race_df)
        n_horses = len(scores)
        
        # 温度付きスケーリング
        scaled_scores = scores / self.temperature
        scaled_scores = scaled_scores - np.max(scaled_scores)
        strengths = np.exp(scaled_scores)
        
        # 結果格納: position -> probabilities array
        position_probs = {}
        
        # 1着確率
        total_strength = np.sum(strengths)
        p_first = strengths / total_strength
        position_probs[1] = p_first
        
        # 2着以降の確率（再帰計算）
        if n_positions >= 2:
            p_second = np.zeros(n_horses)
            for j in range(n_horses):  # jが1着の場合
                remaining_strength = total_strength - strengths[j]
                if remaining_strength > 0:
                    for i in range(n_horses):  # iが2着
                        if i != j:
                            p_second[i] += p_first[j] * (strengths[i] / remaining_strength)
            position_probs[2] = p_second
        
        if n_positions >= 3:
            p_third = np.zeros(n_horses)
            for j in range(n_horses):  # jが1着
                remaining1 = total_strength - strengths[j]
                if remaining1 > 0:
                    for k in range(n_horses):  # kが2着
                        if k != j:
                            remaining2 = remaining1 - strengths[k]
                            if remaining2 > 0:
                                for i in range(n_horses):  # iが3着
                                    if i != j and i != k:
                                        p_third[i] += (p_first[j] * 
                                                       (strengths[k] / remaining1) * 
                                                       (strengths[i] / remaining2))
            position_probs[3] = p_third
        
        return position_probs
    
    def compute_top3_probability(self, race_df: pd.DataFrame) -> np.ndarray:
        """各馬の3着内確率を計算"""
        pos_probs = self.compute_position_probabilities(race_df, n_positions=3)
        p_top3 = pos_probs[1] + pos_probs[2] + pos_probs[3]
        return np.clip(p_top3, 0, 1)
    
    def transform_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """レースデータにPL確率を追加"""
        result = race_df.copy()
        
        # 勝利確率
        p_win = self.compute_win_probabilities(race_df)
        result['pl_win_prob'] = p_win
        
        # 3着内確率
        p_top3 = self.compute_top3_probability(race_df)
        result['pl_top3_prob'] = p_top3
        
        return result
    
    def transform_all(self, df: pd.DataFrame, race_id_col: str = 'race_id') -> pd.DataFrame:
        """全レースに対してPL確率を計算"""
        logger.info(f"Computing Plackett-Luce probabilities with T={self.temperature}...")
        
        results = []
        race_ids = df[race_id_col].unique()
        
        for race_id in race_ids:
            race_df = df[df[race_id_col] == race_id].copy()
            
            if len(race_df) < 2:
                # 馬が1頭以下の場合はスキップ
                continue
            
            try:
                transformed = self.transform_race(race_df)
                results.append(transformed)
            except Exception as e:
                logger.warning(f"Error processing race {race_id}: {e}")
                continue
        
        if results:
            output = pd.concat(results, ignore_index=True)
            logger.info(f"Processed {len(race_ids)} races, output: {len(output)} rows")
            return output
        else:
            return pd.DataFrame()


class TemperatureExplorer:
    """Temperature探索器"""
    
    TEMPERATURE_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    def __init__(self, df: pd.DataFrame, period: PeriodConfig):
        self.df = df
        self.period = period
        self.results = []
    
    def evaluate_temperature(self, temperature: float) -> Dict:
        """特定の温度でのLogLoss評価"""
        config = PlackettLuceConfig(temperature=temperature)
        model = PlackettLuceModel(config)
        
        # PL確率計算
        transformed = model.transform_all(self.df)
        
        if len(transformed) == 0:
            return {'temperature': temperature, 'logloss_win': np.nan, 'logloss_top3': np.nan}
        
        # ターゲット作成
        if 'rank' in transformed.columns:
            y_win = (transformed['rank'] == 1).astype(int)
            y_top3 = (transformed['rank'] <= 3).astype(int)
        else:
            return {'temperature': temperature, 'logloss_win': np.nan, 'logloss_top3': np.nan}
        
        # LogLoss計算
        p_win = np.clip(transformed['pl_win_prob'].values, 1e-7, 1 - 1e-7)
        p_top3 = np.clip(transformed['pl_top3_prob'].values, 1e-7, 1 - 1e-7)
        
        try:
            ll_win = log_loss(y_win, p_win)
            ll_top3 = log_loss(y_top3, p_top3)
        except Exception as e:
            logger.warning(f"LogLoss error at T={temperature}: {e}")
            ll_win, ll_top3 = np.nan, np.nan
        
        return {
            'temperature': temperature,
            'logloss_win': ll_win,
            'logloss_top3': ll_top3,
            'n_samples': len(transformed)
        }
    
    def explore(self, temperatures: List[float] = None) -> List[Dict]:
        """グリッド探索"""
        temperatures = temperatures or self.TEMPERATURE_GRID
        
        logger.info(f"Exploring temperatures: {temperatures}")
        
        self.results = []
        for t in temperatures:
            logger.info(f"  Evaluating T={t}...")
            result = self.evaluate_temperature(t)
            self.results.append(result)
            logger.info(f"    LogLoss(win): {result['logloss_win']:.5f}, "
                       f"LogLoss(top3): {result['logloss_top3']:.5f}")
        
        return self.results
    
    def get_best_temperature(self, metric: str = 'logloss_win') -> float:
        """最適温度を返す"""
        if not self.results:
            raise ValueError("No results. Run explore() first.")
        
        valid = [r for r in self.results if not np.isnan(r.get(metric, np.nan))]
        if not valid:
            return 1.0  # default
        
        best = min(valid, key=lambda x: x[metric])
        return best['temperature']
    
    def generate_report(self, output_path: str):
        """探索結果レポート"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = f"""# Phase 6: Temperature Exploration Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Period**: {self.period.start_year}-{self.period.end_year}

## Temperature Grid Search Results

| Temperature | LogLoss (Win) | LogLoss (Top3) | Samples |
|-------------|---------------|----------------|---------|
"""
        for r in self.results:
            report += f"| {r['temperature']:.2f} | {r.get('logloss_win', np.nan):.5f} | {r.get('logloss_top3', np.nan):.5f} | {r.get('n_samples', 0):,} |\n"
        
        best_t_win = self.get_best_temperature('logloss_win')
        best_t_top3 = self.get_best_temperature('logloss_top3')
        
        report += f"""
## Best Temperature

| Metric | Best T |
|--------|--------|
| LogLoss (Win) | **{best_t_win:.2f}** |
| LogLoss (Top3) | **{best_t_top3:.2f}** |

## Recommendation

採用推奨値: **T = {best_t_win:.2f}** (Win LogLoss 最適化)

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Plackett-Luce Ranking Model with Temperature"
    )
    add_period_args(parser)
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/derived/preprocessed_with_prob_v12.parquet',
        help='Input data path'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature parameter'
    )
    parser.add_argument(
        '--explore_temperature',
        action='store_true',
        help='Run temperature grid search'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/probabilities',
        help='Output directory for probability data'
    )
    parser.add_argument(
        '--report_dir',
        type=str,
        default='reports',
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    try:
        period = parse_period_args(args)
    except ValueError as e:
        logger.error(f"Period error: {e}")
        sys.exit(1)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    
    # Add year column if missing
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    # Filter by period
    df = filter_dataframe_by_period(df, period)
    
    # Filter to rows with prob
    if 'prob' in df.columns:
        df = df[df['prob'].notna()].copy()
    if 'raw_score' in df.columns:
        df = df[df['raw_score'].notna()].copy()
    
    logger.info(f"Data loaded: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Temperature exploration
    if args.explore_temperature:
        explorer = TemperatureExplorer(df, period)
        explorer.explore()
        explorer.generate_report(
            os.path.join(args.report_dir, 'phase6_temperature_exploration.md')
        )
        
        best_t = explorer.get_best_temperature()
        logger.info(f"Best temperature: {best_t}")
        
        # Save config
        config_path = os.path.join('config', 'plackett_luce_params.yaml')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump({
                'temperature': best_t,
                'exploration_results': explorer.results,
                'generated_at': datetime.now().isoformat()
            }, f, default_flow_style=False)
        logger.info(f"Config saved to {config_path}")
    
    else:
        # Transform with specified temperature
        config = PlackettLuceConfig(temperature=args.temperature)
        model = PlackettLuceModel(config)
        
        transformed = model.transform_all(df)
        
        # Save output
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'pl_probabilities.parquet')
        transformed.to_parquet(output_path, index=False)
        logger.info(f"Probabilities saved to {output_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

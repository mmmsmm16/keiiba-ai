"""
Phase 6: Monte Carlo Probability Sampler
組み合わせ確率のモンテカルロサンプリング（馬連/三連複/三連単）

Usage (in container):
    docker compose exec app python src/models/probability_sampler.py --period screening --samples 2000
    docker compose exec app python src/models/probability_sampler.py --period verification --samples 5000
"""

import sys
import os
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import combinations, permutations

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.period_guard import add_period_args, parse_period_args, filter_dataframe_by_period, PeriodConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlackettLuceSampler:
    """
    Plackett-Luceモデルによるモンテカルロサンプリング
    
    順位シミュレーションを繰り返して組み合わせ確率を推定
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        n_samples: int = 5000,
        seed: int = 42,
        top_n_collect: int = 8  # 集計対象のTop N頭
    ):
        self.temperature = temperature
        self.n_samples = n_samples
        self.seed = seed
        self.top_n_collect = top_n_collect
        self.rng = np.random.default_rng(seed)
    
    def _get_strengths(self, scores: np.ndarray) -> np.ndarray:
        """温度付きstrength計算"""
        scaled = scores / self.temperature
        scaled = scaled - np.max(scaled)  # numerical stability
        return np.exp(scaled)
    
    def sample_ranking(self, strengths: np.ndarray) -> np.ndarray:
        """
        1回のランキングサンプリング
        
        Plackett-Luce順位生成:
        1. 各馬の強さに比例してランダム選択 → 1着
        2. 1着を除いた残りで再度選択 → 2着
        3. 繰り返し
        """
        n_horses = len(strengths)
        remaining = list(range(n_horses))
        remaining_strengths = strengths.copy()
        ranking = []
        
        for _ in range(n_horses):
            if len(remaining) == 0:
                break
            
            # 確率計算
            total = remaining_strengths[remaining].sum()
            if total <= 0:
                # 全馬の強さが0以下（異常ケース）
                selected_idx = self.rng.integers(0, len(remaining))
            else:
                probs = remaining_strengths[remaining] / total
                selected_idx = self.rng.choice(len(remaining), p=probs)
            
            selected_horse = remaining[selected_idx]
            ranking.append(selected_horse)
            remaining.pop(selected_idx)
        
        return np.array(ranking)
    
    def sample_rankings_batch(self, strengths: np.ndarray, n_samples: int) -> np.ndarray:
        """複数回のランキングサンプリング"""
        rankings = []
        for _ in range(n_samples):
            ranking = self.sample_ranking(strengths)
            rankings.append(ranking)
        return np.array(rankings)
    
    def compute_combination_probabilities(
        self,
        race_df: pd.DataFrame,
        score_col: str = 'raw_score',
        prob_col: str = 'prob',
        horse_id_col: str = 'umaban'
    ) -> Dict:
        """
        レースの組み合わせ確率を計算
        
        Returns:
            Dict with keys: 'win', 'top3', 'pair', 'trio_unordered', 'trio_ordered'
        """
        # スコア取得
        if score_col in race_df.columns and race_df[score_col].notna().all():
            scores = race_df[score_col].values.astype(float)
        elif prob_col in race_df.columns:
            probs = np.clip(race_df[prob_col].values, 1e-7, 1 - 1e-7)
            scores = np.log(probs / (1 - probs))
        else:
            raise ValueError(f"No score column found: {score_col}, {prob_col}")
        
        n_horses = len(race_df)
        horse_ids = race_df[horse_id_col].values if horse_id_col in race_df.columns else np.arange(n_horses)
        
        # Strength計算
        strengths = self._get_strengths(scores)
        
        # サンプリング
        rankings = self.sample_rankings_batch(strengths, self.n_samples)
        
        # Top N for collection (点数爆発対策)
        top_n = min(self.top_n_collect, n_horses)
        
        # スコア順でTop N馬を特定
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_horse_ids = horse_ids[top_indices]
        
        # 確率カウント
        result = {
            'n_horses': n_horses,
            'n_samples': self.n_samples,
            'win': {},  # horse_id -> probability
            'top3': {},  # horse_id -> probability
            'pair': {},  # (h1, h2) sorted -> probability
            'trio_unordered': {},  # (h1, h2, h3) sorted -> probability
            'trio_ordered': {},  # (h1, h2, h3) order matters -> probability
        }
        
        # カウント
        win_counts = defaultdict(int)
        top3_counts = defaultdict(int)
        pair_counts = defaultdict(int)
        trio_unordered_counts = defaultdict(int)
        trio_ordered_counts = defaultdict(int)
        
        for ranking in rankings:
            # 順位からhorse_id取得
            ranked_horses = horse_ids[ranking]
            
            # Win (1着)
            if len(ranked_horses) >= 1:
                winner = ranked_horses[0]
                win_counts[winner] += 1
            
            # Top3 (3着以内)
            for i, h in enumerate(ranked_horses[:3]):
                top3_counts[h] += 1
            
            # Pair (1-2着の組、順不同)
            if len(ranked_horses) >= 2:
                h1, h2 = ranked_horses[0], ranked_horses[1]
                pair_key = tuple(sorted([h1, h2]))
                pair_counts[pair_key] += 1
            
            # Trio (1-2-3着)
            if len(ranked_horses) >= 3:
                h1, h2, h3 = ranked_horses[0], ranked_horses[1], ranked_horses[2]
                
                # 順不同 (三連複)
                trio_unordered_key = tuple(sorted([h1, h2, h3]))
                trio_unordered_counts[trio_unordered_key] += 1
                
                # 順あり (三連単)
                trio_ordered_key = (h1, h2, h3)
                trio_ordered_counts[trio_ordered_key] += 1
        
        # 確率に変換
        for h, count in win_counts.items():
            result['win'][h] = count / self.n_samples
        
        for h, count in top3_counts.items():
            result['top3'][h] = count / self.n_samples
        
        for pair, count in pair_counts.items():
            result['pair'][pair] = count / self.n_samples
        
        for trio, count in trio_unordered_counts.items():
            result['trio_unordered'][trio] = count / self.n_samples
        
        for trio, count in trio_ordered_counts.items():
            result['trio_ordered'][trio] = count / self.n_samples
        
        # 確率質量欠損チェック
        result['mass_check'] = {
            'sum_win': sum(result['win'].values()),
            'sum_top3': sum(result['top3'].values()) / 3,  # 3人が3着になるので/3
            'sum_pair': sum(result['pair'].values()),
            'sum_trio_unordered': sum(result['trio_unordered'].values()),
            'sum_trio_ordered': sum(result['trio_ordered'].values()),
        }
        
        return result


class RaceProbabilityGenerator:
    """全レースの組み合わせ確率を生成"""
    
    def __init__(
        self,
        temperature: float = 1.0,
        n_samples: int = 5000,
        seed: int = 42,
        top_n_collect: int = 8,
        min_top1_prob: float = 0.0  # フィルタ用
    ):
        self.sampler = PlackettLuceSampler(
            temperature=temperature,
            n_samples=n_samples,
            seed=seed,
            top_n_collect=top_n_collect
        )
        self.min_top1_prob = min_top1_prob
        self.results = []
    
    def process_all(
        self,
        df: pd.DataFrame,
        race_id_col: str = 'race_id'
    ) -> Dict[str, pd.DataFrame]:
        """全レースを処理"""
        race_ids = df[race_id_col].unique()
        
        logger.info(f"Processing {len(race_ids)} races with {self.sampler.n_samples} samples each...")
        
        # 結果格納
        pair_records = []
        trio_unordered_records = []
        trio_ordered_records = []
        mass_checks = []
        
        processed = 0
        for race_id in race_ids:
            race_df = df[df[race_id_col] == race_id].copy()
            
            if len(race_df) < 3:
                continue  # 3頭未満はスキップ
            
            # Top1確率でフィルタ
            if self.min_top1_prob > 0 and 'prob' in race_df.columns:
                max_prob = race_df['prob'].max()
                if max_prob < self.min_top1_prob:
                    continue
            
            try:
                result = self.sampler.compute_combination_probabilities(race_df)
                
                # Pair records
                for (h1, h2), prob in result['pair'].items():
                    pair_records.append({
                        'race_id': race_id,
                        'horse_1': h1,
                        'horse_2': h2,
                        'probability': prob
                    })
                
                # Trio unordered (三連複)
                for (h1, h2, h3), prob in result['trio_unordered'].items():
                    trio_unordered_records.append({
                        'race_id': race_id,
                        'horse_1': h1,
                        'horse_2': h2,
                        'horse_3': h3,
                        'probability': prob
                    })
                
                # Trio ordered (三連単)
                for (h1, h2, h3), prob in result['trio_ordered'].items():
                    trio_ordered_records.append({
                        'race_id': race_id,
                        'first': h1,
                        'second': h2,
                        'third': h3,
                        'probability': prob
                    })
                
                # Mass check
                mass_checks.append({
                    'race_id': race_id,
                    'n_horses': result['n_horses'],
                    **result['mass_check']
                })
                
                processed += 1
                if processed % 500 == 0:
                    logger.info(f"  Processed {processed}/{len(race_ids)} races")
                    
            except Exception as e:
                logger.warning(f"Error on race {race_id}: {e}")
                continue
        
        logger.info(f"Completed: {processed} races processed")
        
        # DataFrameに変換
        outputs = {
            'pair': pd.DataFrame(pair_records) if pair_records else pd.DataFrame(),
            'trio_unordered': pd.DataFrame(trio_unordered_records) if trio_unordered_records else pd.DataFrame(),
            'trio_ordered': pd.DataFrame(trio_ordered_records) if trio_ordered_records else pd.DataFrame(),
            'mass_check': pd.DataFrame(mass_checks) if mass_checks else pd.DataFrame()
        }
        
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Monte Carlo Probability Sampler"
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
        help='Plackett-Luce temperature'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=5000,
        help='Number of Monte Carlo samples per race'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=8,
        help='Top N horses to collect combinations for'
    )
    parser.add_argument(
        '--min_top1_prob',
        type=float,
        default=0.0,
        help='Minimum top1 probability to process race'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/probabilities',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    try:
        period = parse_period_args(args)
    except ValueError as e:
        logger.error(f"Period error: {e}")
        sys.exit(1)
    
    # Load config if exists
    config_path = 'config/plackett_luce_params.yaml'
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if 'temperature' in config:
            args.temperature = config['temperature']
            logger.info(f"Using temperature from config: {args.temperature}")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    df = filter_dataframe_by_period(df, period)
    
    # Filter to rows with prob
    if 'prob' in df.columns:
        df = df[df['prob'].notna()].copy()
    
    logger.info(f"Data loaded: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Generate probabilities
    generator = RaceProbabilityGenerator(
        temperature=args.temperature,
        n_samples=args.samples,
        seed=args.seed,
        top_n_collect=args.top_n,
        min_top1_prob=args.min_top1_prob
    )
    
    outputs = generator.process_all(df)
    
    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    for name, data in outputs.items():
        if len(data) > 0:
            output_path = os.path.join(args.output_dir, f'{name}_probs.parquet')
            data.to_parquet(output_path, index=False)
            logger.info(f"Saved {name}: {len(data):,} rows to {output_path}")
    
    # Mass check summary
    if 'mass_check' in outputs and len(outputs['mass_check']) > 0:
        mc = outputs['mass_check']
        logger.info("\n=== Probability Mass Check ===")
        logger.info(f"  sum_pair mean: {mc['sum_pair'].mean():.4f} (should be ~1.0)")
        logger.info(f"  sum_trio_unordered mean: {mc['sum_trio_unordered'].mean():.4f} (should be ~1.0)")
        logger.info(f"  sum_trio_ordered mean: {mc['sum_trio_ordered'].mean():.4f} (should be ~1.0)")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

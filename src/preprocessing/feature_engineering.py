import pandas as pd
import logging
from typing import List, Dict, Callable
import sys
import os

# Ensure pipeline is importable
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing.pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature Engineering Entry Point (Reboot).
    Uses FeaturePipeline to manage Feature Blocks with Caching.
    """
    def __init__(self, cache_dir: str = "data/cache/features"):
        self.pipeline = FeaturePipeline(cache_dir, dataset_name="v2_reboot")
        self.registry: Dict[str, Callable] = {}
        self._register_default_blocks()

    def _register_default_blocks(self):
        """Register available feature blocks."""
        # 1. Base Attributes (Raw data mostly)
        self.registry["base_attributes"] = self._block_base_attributes
        
        # Future blocks will be added here
        # self.registry["jockey_stats"] = self._block_jockey_stats
        
    def _block_base_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] Base Attributes
        - Keep robust raw columns
        - Drop strictly technical/leak columns
        """
        logger.info("Generating Block: base_attributes")
        
        # List of columns to KEEP as features or keys
        keep_cols = [
            # Keys
            'race_id', 'date', 'horse_id', 'jockey_id', 'trainer_id',
            # Raw Features
            'bracket_number', 'horse_number', # 枠・馬番
            'age', 'sex', # 年齢・性別
            'weight_loss', # 斤量
            'weight', 'weight_diff', # 馬体重 (Note: Cleanseで処理済み前提)
            'course_id', # コースID (Cleanseで生成済み前提)
            'distance', # 距離
            # Target (for safety, pass through, dropped later in dataset)
            'rank', 'time' 
        ]
        
        # Return only the relevant columns (plus keys for merging if needed, 
        # but Pipeline expects df -> df transformation).
        # Actually, we should probably APPEND features. 
        # But if we cache blocks, a block should ideally return *just the new features* + keys.
        # For 'base', it returns the subset of the original df.
        
        return df[df.columns.intersection(keep_cols)].copy()

    def add_features(self, df: pd.DataFrame, feature_blocks: List[str]) -> pd.DataFrame:
        """
        Apply selected feature blocks using the pipeline.
        
        Args:
            df: Initial raw/cleansed dataframe.
            feature_blocks: List of block names to apply.
        """
        logger.info(f"Feature Engineering: Applying blocks {feature_blocks}")
        
        # 1. Always start with 'raw' in pipeline?
        # The pipeline expects us to define steps.
        # Step 0: Input Trigger (We can pass 'df' as initial input to first step?)
        # FeaturePipeline.run() design:
        # It executes steps based on dependencies. 
        # We need a "Source" step that just returns the input df? 
        # Or we can modify the first block to accept the input df directly.
        
        # Let's verify 'pipeline.py'. It assumes steps read from cache or depend on previous steps.
        # We need a "root" step.
        
        # Define 'input' step that just passes/caches the base clean data
        # Be careful: Input df is huge. We might NOT want to cache the full input every time if it changes.
        # But for reproducibility, caching 'cleansed' input is good.
        
        # However, usually we load via Loader.
        # Here 'add_features' is called *after* loading.
        # Let's assume 'base' step takes the dataframe given to 'add_features'.
        # Note: FeaturePipeline.run() usually triggers the whole chain. 
        # If we pass `df` to `run`, pipeline needs to support "initial input".
        
        # Hack/Adaptation:
        # We will manually manage the "merge" of blocks.
        # The Pipeline class in `pipeline.py` is linear global chain.
        # We want "Independent Blocks" that merge at the end.
        
        # Let's use Pipeline for *each block* independently if they depend on raw data.
        # But `base_attributes` depends on `raw`.
        # `jockey_stats` depends on `raw` (and maybe `base`).
        
        # Simpler approach for REBOOT:
        # Use Pipeline for the *whole process* if possible.
        # But if we want mix-and-match in config, we need flexible dependency.
        
        # Revised Strategy:
        # 1. `clean_data` is the common parent.
        # 2. Each block is a PipelineStep depending on `clean_data`.
        # 3. Final step `merge` joins them.
        
        # Implementation:
        # We must define the pipeline structure dynamically based on `feature_blocks`.
        
        # Step 1: Register 'clean' step (The input).
        # Since 'pipeline.py' loads from cache or runs processor.
        # We can implement a dummy processor that returns the passed `df`.
        # BUT `pipeline.run` resolves order.
        
        # For now, to keep it simple and follow user advice:
        # "Feature BlockごとにParquetでキャッシュ"
        
        output_df = pd.DataFrame(index=df.index)
        # We need identifying columns to merge back (race_id, horse_number)
        key_cols = ['race_id', 'horse_number']
        
        # Ensure Keys exist
        if 'horse_number' not in df.columns or 'race_id' not in df.columns:
            logger.error("Missing key columns for merge.")
            return df
            
        final_df = df[key_cols].copy() 
        # Wait, usually we want to KEEP the raw df and ADD features.
        # So start with `df`.
        final_df = df.copy()
        
        for block in feature_blocks:
            if block not in self.registry:
                logger.warning(f"Feature block '{block}' not registered. Skipping.")
                continue
            
            # Execute block (with caching)
            # We can use the Pipeline logic for this specific block:
            # step_name = block
            # processor = self.registry[block]
            # dependency = 'raw' (implicit)
            
            # We add a step to the pipeline on the fly?
            # self.pipeline.add_step(block, self.registry[block], dependencies=[])
            # But the 'processor' needs 'input_df'. 
            
            # Direct cache management (Simplified)
            block_df = self._run_block_with_cache(block, df)
            
            # Merge
            # Check for duplicate columns
            new_cols = [c for c in block_df.columns if c not in final_df.columns and c not in key_cols]
            if new_cols:
                # Merge on index if aligned, or on keys
                # We assume row alignment if we just passed `df` and got `block_df` of same length/order.
                # If blocks sort/filter, we must merge on keys.
                if len(block_df) != len(final_df):
                    final_df = final_df.merge(block_df[key_cols + new_cols], on=key_cols, how='left')
                else:
                    final_df = pd.concat([final_df, block_df[new_cols]], axis=1)
        
        return final_df

    def _run_block_with_cache(self, block_name: str, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute a feature block with caching.
        Using a simplified caching mechanism wrapper around the registry function.
        """
        # We can reuse PipelineStep logic or just do it manually here for simplicity 
        # as Pipeline class is a bit rigid for this dynamic mix-and-match.
        
        # Actually, let's look at `pipeline.py`. It's good.
        # We can create a pipeline, add ONE step (the block), and run it.
        # We need to inject `input_df`.
        
        # If we use `pipeline.add_step(block, func, ...)`, `pipeline.run` will try to load deps.
        # If no deps, it runs `func()`. 
        # Our `func` needs `input_df`.
        
        # Hack: Pass `input_df` via `params` or wrapper? No, `params` are kwargs.
        # If we assume blocks are pure functions of `input_df`, we can cache the RESULT based on `input_df` hash?
        # Hashing a dataframe is expensive.
        
        # User said: "Feature BlockごとにParquetでキャッシュ"
        # Implicitly assuming the INPUT (clean data) is stable for a given experiment period.
        # If input data changes (new races), cache is invalid?
        # Or we use `Incremental`?
        
        # For "Reboot", let's assume we process the "Whole History" at once for now.
        # So we cache `features/base_attributes.parquet`.
        
        processor = self.registry[block_name]
        
        # Check cache
        cache_path = os.path.join(self.pipeline.cache_dir, f"{block_name}.parquet")
        
        if os.path.exists(cache_path):
             logger.info(f"Block '{block_name}': Loading from cache...")
             return pd.read_parquet(cache_path)
        
        # Run
        logger.info(f"Block '{block_name}': Computing...")
        out_df = processor(input_df)
        
        # Save
        out_df.to_parquet(cache_path)
        return out_df


"""
Hierarchical Feature Pipeline

Phase 16: 特徴量パイプラインをモジュール化し、各ステップを個別にキャッシュ。
特徴量追加時に変更箇所以降のみ再処理することで処理時間を大幅短縮。

使用例:
    pipeline = FeaturePipeline("data/cache/jra")
    pipeline.add_step("raw", loader.load, [])
    pipeline.add_step("cleanse", cleanser.cleanse, ["raw"])
    df = pipeline.run()  # 必要なステップのみ実行
"""

import os
import hashlib
import json
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any

logger = logging.getLogger(__name__)


class PipelineStep:
    """パイプラインの1ステップを表すクラス"""
    
    def __init__(self, name: str, processor: Callable, dependencies: List[str], 
                 version: str = "1.0", params: Dict = None):
        """
        Args:
            name: ステップ名（キャッシュファイル名に使用）
            processor: 処理関数（DataFrameを引数として受け取り、DataFrameを返す）
            dependencies: 依存するステップ名のリスト
            version: バージョン（変更時にキャッシュ無効化）
            params: 処理関数に渡す追加パラメータ
        """
        self.name = name
        self.processor = processor
        self.dependencies = dependencies
        self.version = version
        self.params = params or {}
        self.cache_path: Optional[str] = None
        self.meta_path: Optional[str] = None
    
    def get_cache_key(self, dep_hashes: Dict[str, str]) -> str:
        """キャッシュキーを生成（依存ステップのハッシュを含む）"""
        
        def make_serializable(obj):
            """非JSONシリアライズ可能なオブジェクトを変換"""
            if isinstance(obj, range):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            return obj
        
        key_data = {
            "version": self.version,
            "params": make_serializable(self.params),
            "dependencies": {dep: dep_hashes.get(dep, "") for dep in self.dependencies}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]


class FeaturePipeline:
    """階層的キャッシュを管理する特徴量パイプライン"""
    
    def __init__(self, cache_dir: str, dataset_name: str = "default"):
        """
        Args:
            cache_dir: キャッシュディレクトリのパス
            dataset_name: データセット識別子（JRA/NARなど）
        """
        self.cache_dir = os.path.join(cache_dir, dataset_name)
        self.steps: Dict[str, PipelineStep] = {}
        self.execution_order: List[str] = []
        self.step_hashes: Dict[str, str] = {}
        self.cached_data: Dict[str, pd.DataFrame] = {}
        
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def add_step(self, name: str, processor: Callable, dependencies: List[str] = None,
                 version: str = "1.0", params: Dict = None) -> 'FeaturePipeline':
        """
        処理ステップを追加
        
        Args:
            name: ステップ名
            processor: 処理関数
            dependencies: 依存ステップ名リスト
            version: バージョン（変更時にキャッシュ無効化）
            params: 追加パラメータ
            
        Returns:
            self（メソッドチェーン用）
        """
        step = PipelineStep(name, processor, dependencies or [], version, params)
        step.cache_path = os.path.join(self.cache_dir, f"{name}.parquet")
        step.meta_path = os.path.join(self.cache_dir, f"{name}.meta.json")
        self.steps[name] = step
        
        # 依存関係の検証
        for dep in step.dependencies:
            if dep not in self.steps:
                logger.warning(f"Step '{name}' depends on '{dep}' which is not yet defined")
        
        return self
    
    def _resolve_execution_order(self) -> List[str]:
        """依存関係を解決し、実行順序を決定（トポロジカルソート）"""
        visited = set()
        order = []
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            step = self.steps[name]
            for dep in step.dependencies:
                if dep in self.steps:
                    visit(dep)
            order.append(name)
        
        for name in self.steps:
            visit(name)
        
        return order
    
    def _is_cache_valid(self, step: PipelineStep) -> bool:
        """キャッシュが有効かどうかを確認"""
        if not os.path.exists(step.cache_path):
            return False
        if not os.path.exists(step.meta_path):
            return False
        
        try:
            with open(step.meta_path, 'r') as f:
                meta = json.load(f)
            
            # 現在のキャッシュキーと保存されたキーを比較
            current_key = step.get_cache_key(self.step_hashes)
            return meta.get("cache_key") == current_key
        except Exception as e:
            logger.warning(f"Failed to read metadata for {step.name}: {e}")
            return False
    
    def _save_metadata(self, step: PipelineStep, cache_key: str, row_count: int):
        """メタデータを保存"""
        meta = {
            "cache_key": cache_key,
            "version": step.version,
            "created_at": datetime.now().isoformat(),
            "row_count": row_count,
            "dependencies": step.dependencies
        }
        with open(step.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def run(self, force_from: str = None, target_step: str = None) -> pd.DataFrame:
        """
        パイプラインを実行
        
        Args:
            force_from: 指定したステップから強制再実行
            target_step: 実行する最終ステップ（Noneなら全ステップ）
            
        Returns:
            最終ステップの出力DataFrame
        """
        self.execution_order = self._resolve_execution_order()
        logger.info(f"Pipeline execution order: {' -> '.join(self.execution_order)}")
        
        force_rebuild = False
        
        for step_name in self.execution_order:
            step = self.steps[step_name]
            
            # force_from指定時、そのステップ以降は強制再構築
            if force_from and step_name == force_from:
                force_rebuild = True
                logger.info(f"Force rebuild from step: {step_name}")
            
            # キャッシュチェック
            if not force_rebuild and self._is_cache_valid(step):
                logger.info(f"⏭️  Step '{step_name}': Using cached data")
                
                # メタデータからキャッシュキーを読み込み
                with open(step.meta_path, 'r') as f:
                    meta = json.load(f)
                self.step_hashes[step_name] = meta.get("cache_key", "")
                
                # 必要な場合のみデータをロード
                if step_name == target_step or step_name == self.execution_order[-1]:
                    self.cached_data[step_name] = pd.read_parquet(step.cache_path)
                continue
            
            # 再構築フラグをセット（依存ステップのキャッシュが無効化されたら以降も再構築）
            force_rebuild = True
            
            # 依存ステップのデータを取得
            input_data = None
            if step.dependencies:
                # 最初の依存ステップのデータを入力として使用
                main_dep = step.dependencies[0]
                if main_dep in self.cached_data:
                    input_data = self.cached_data[main_dep]
                else:
                    input_data = pd.read_parquet(self.steps[main_dep].cache_path)
            
            # ステップを実行
            logger.info(f"🔄 Step '{step_name}': Processing...")
            start_time = datetime.now()
            
            if input_data is not None:
                output_data = step.processor(input_data, **step.params)
            else:
                output_data = step.processor(**step.params)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Step '{step_name}': Completed in {elapsed:.1f}s ({len(output_data):,} rows)")
            
            # キャッシュを保存
            cache_key = step.get_cache_key(self.step_hashes)
            output_data.to_parquet(step.cache_path)
            self._save_metadata(step, cache_key, len(output_data))
            
            self.step_hashes[step_name] = cache_key
            self.cached_data[step_name] = output_data
            
            # target_stepに到達したら終了
            if target_step and step_name == target_step:
                break
        
        # 最終出力を返す
        final_step = target_step or self.execution_order[-1]
        if final_step in self.cached_data:
            return self.cached_data[final_step]
        else:
            return pd.read_parquet(self.steps[final_step].cache_path)
    
    def invalidate(self, step_name: str):
        """指定ステップ以降のキャッシュを無効化"""
        if step_name not in self.steps:
            raise ValueError(f"Unknown step: {step_name}")
        
        self.execution_order = self._resolve_execution_order()
        start_idx = self.execution_order.index(step_name)
        
        for name in self.execution_order[start_idx:]:
            step = self.steps[name]
            if os.path.exists(step.cache_path):
                os.remove(step.cache_path)
                logger.info(f"Invalidated cache for step: {name}")
            if os.path.exists(step.meta_path):
                os.remove(step.meta_path)
    
    def get_cache_status(self) -> Dict[str, Dict]:
        """各ステップのキャッシュ状態を取得"""
        self.execution_order = self._resolve_execution_order()
        status = {}
        
        for name in self.execution_order:
            step = self.steps[name]
            if os.path.exists(step.meta_path):
                with open(step.meta_path, 'r') as f:
                    meta = json.load(f)
                status[name] = {
                    "cached": True,
                    "created_at": meta.get("created_at"),
                    "row_count": meta.get("row_count"),
                    "version": meta.get("version")
                }
            else:
                status[name] = {"cached": False}
        
        return status
    
    def print_status(self):
        """キャッシュ状態を表示"""
        status = self.get_cache_status()
        print("\n=== Pipeline Cache Status ===")
        for name, info in status.items():
            if info["cached"]:
                print(f"  ✅ {name}: {info['row_count']:,} rows (v{info['version']}, {info['created_at'][:10]})")
            else:
                print(f"  ❌ {name}: No cache")
        print()

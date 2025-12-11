from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import yaml
import os

class DataConfig(BaseModel):
    train_years: List[int] = Field(default=[2020, 2021, 2022, 2023, 2024])
    valid_year: int = Field(default=2025)
    features: str = Field(default="v5_default")
    drop_features: List[str] = Field(default_factory=list) # 特定の特徴量を除外する場合に指定
    use_cache: bool = Field(default=True)
    cache_path: Optional[str] = Field(default=None)  # 明示的なキャッシュパス（省略時は自動生成）
    jra_only: bool = Field(default=True)
    target_type: str = Field(default="ranking")  # "ranking" (v12互換) or "v13_graded" (複勝圏スコア)

class ModelConfig(BaseModel):
    type: str = Field(default="ensemble")  # "lgbm", "catboost", "tabnet", "ensemble", "ensemble_only", "roi"
    lgbm_params: Optional[Dict[str, Any]] = None
    catboost_params: Optional[Dict[str, Any]] = None
    tabnet_params: Optional[Dict[str, Any]] = None
    roi_params: Optional[Dict[str, Any]] = None  # ROI最適化モデル用パラメータ

class EvaluationConfig(BaseModel):
    metric: str = Field(default="roi")
    strategies: List[str] = Field(default=["umaren", "sanrentan"])
    jra_only: bool = Field(default=True)  # 評価時JRAのみを対象にする（NAR除外）

class StrategyConfig(BaseModel):
    enabled: bool = Field(default=True)
    target_bet_types: List[str] = Field(default=["tansho", "umaren", "sanrentan"])
    optimize_thresholds: bool = Field(default=True)
    min_roi: float = Field(default=100.0)

class ExperimentConfig(BaseModel):
    experiment_name: str
    base_dir: str = Field(default="experiments")
    description: Optional[str] = None
    
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_run_dir(self) -> str:
        """実験ごとの出力ディレクトリパス"""
        return os.path.join(self.base_dir, self.experiment_name)
    
    def setup_dirs(self):
        """必要なディレクトリを作成"""
        run_dir = self.get_run_dir()
        os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "reports"), exist_ok=True)
        return run_dir

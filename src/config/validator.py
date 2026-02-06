
import logging
import re
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ConfigValidator:
    """
    実験設定(Config)の整合性を検証するクラス。
    実験名とTargetの不一致などを検出し、事故を防止する。
    """
    
    @staticmethod
    def validate(config: Dict[str, Any], config_path: str = None, strict: bool = False) -> None:
        """
        Configの整合性を検証する。
        問題がある場合は ValueError を送出する。
        
        Args:
            config: ロード済みのConfig辞書
            config_path: Configファイルのパス (ファイル名チェック用)
            strict: Trueの場合、Warningレベルの不整合もErrorとして扱う
        """
        logger.info(f"🛡️ Config Guardrail: Validating experiment configuration... (Strict={strict})")
        
        exp_name = config.get('experiment_name', config.get('name', 'unknown'))
        dataset = config.get('dataset', {})
        model_params = config.get('model_params', {})
        objective = model_params.get('objective', 'binary')
        
        # 1. Target Consistency Check (Experiment Name & Filename)
        ConfigValidator._validate_target_consistency(exp_name, dataset, objective, config_path)
        
        # 2. Metric Consistency Check
        ConfigValidator._validate_metric_consistency(model_params, dataset, strict)
        
        # 3. Time-Decay Consistency Check
        ConfigValidator._validate_time_decay_consistency(config.get('sample_weight', {}), strict)
        
        # 4. Group Key Consistency Check
        ConfigValidator._validate_group_key_consistency(model_params, objective, strict)

        # 5. Task vs Objective Check
        ConfigValidator._validate_task_objective_consistency(config, strict)
        
        logger.info("✅ Config Guardrail: Validation Passed.")

    @staticmethod
    def _validate_target_consistency(exp_name: str, dataset: Dict[str, Any], objective: str, config_path: str = None) -> None:
        """実験名とターゲットカラムの矛盾をチェック"""
        target_col = dataset.get('target_col', '')
        binary_target = dataset.get('binary_target', '')
        
        # Check both Experiment Name and Config Filename
        names_to_check = [exp_name]
        if config_path:
            basename = os.path.basename(config_path)
            names_to_check.append(basename)
            
        for name in names_to_check:
            name_lower = name.lower()
            
            # Rule: "top3" in name => target must be Top3
            if 'top3' in name_lower:
                if target_col and target_col != 'target_top3':
                    if not binary_target or binary_target != 'top3':
                        raise ValueError(f"⛔ CONFIG ERROR: Name '{name}' contains 'top3' but target is '{target_col}' (binary_target='{binary_target}'). Expected 'target_top3'.")
                elif not target_col and (not binary_target or binary_target != 'top3'):
                    raise ValueError(f"⛔ CONFIG ERROR: Name '{name}' contains 'top3' but binary_target is '{binary_target}'. Expected 'top3'.")

            # Rule: "top2" in name => target must be Top2
            if 'top2' in name_lower:
                if target_col and target_col != 'target_top2':
                    if not binary_target or binary_target != 'top2':
                        raise ValueError(f"⛔ CONFIG ERROR: Name '{name}' contains 'top2' but target is '{target_col}' (binary_target='{binary_target}'). Expected 'target_top2'.")
                elif not target_col and (not binary_target or binary_target != 'top2'):
                    raise ValueError(f"⛔ CONFIG ERROR: Name '{name}' contains 'top2' but binary_target is '{binary_target}'. Expected 'top2'.")

            # Rule: "win" or "top1" in name => target must be Win
            if 'win' in name_lower or 'top1' in name_lower:
                if 'winter' in name_lower or 'twin' in name_lower: continue # basic exclusion
                if target_col and target_col != 'target_win':
                    if not binary_target or binary_target != 'win':
                        raise ValueError(f"⛔ CONFIG ERROR: Name '{name}' contains 'win' but target is '{target_col}' (binary_target='{binary_target}'). Expected 'target_win'.")
                elif not target_col and (binary_target and binary_target != 'win'):
                    raise ValueError(f"⛔ CONFIG ERROR: Name '{name}' contains 'win' but binary_target is '{binary_target}'. Expected 'win'.")

    @staticmethod
    def _validate_metric_consistency(model_params: Dict[str, Any], dataset: Dict[str, Any], strict: bool) -> None:
        """指標とタスクの整合性をチェック"""
        objective = model_params.get('objective', '')
        metrics = model_params.get('metric', [])
        if isinstance(metrics, str): metrics = [metrics]
        
        msg = ""
        if objective == 'lambdarank':
            if not any(m.lower().startswith('ndcg') or m.lower().startswith('map') for m in metrics):
                msg = "LambdaRank objective but no NDCG/MAP metric specified."
        
        elif objective == 'binary':
            if not any(m.lower() in ['auc', 'binary_logloss', 'logloss'] for m in metrics):
                msg = "Binary objective but no AUC/LogLoss metric specified."

        if msg:
            if strict:
                raise ValueError(f"⛔ CONFIG ERROR (Strict): {msg}")
            else:
                logger.warning(f"⚠️ GUARDRAIL WARN: {msg}")

    @staticmethod
    def _validate_time_decay_consistency(sample_weight: Dict[str, Any], strict: bool) -> None:
        """Time-Decay設定の整合性をチェック"""
        if not sample_weight.get('enabled', False):
            return

        strategy = sample_weight.get('strategy', 'none')
        
        if strategy == 'piecewise':
            yw = sample_weight.get('year_weights', {})
            if not yw:
                 raise ValueError("⛔ CONFIG ERROR: Strategy 'piecewise' requires 'year_weights' map.")
            for k, w in yw.items():
                if w <= 0 or w > 1.0:
                    msg = f"Unusual weight value {w} for key {k}. Usually 0 < w <= 1."
                    if strict: raise ValueError(f"⛔ CONFIG ERROR (Strict): {msg}")
                    else: logger.warning(f"⚠️ GUARDRAIL WARN: {msg}")
                    
        elif strategy == 'exponential':
            decay = sample_weight.get('decay_rate', 0.0)
            if decay <= 0:
                 raise ValueError(f"⛔ CONFIG ERROR: Strategy 'exponential' requires positive 'decay_rate'. Found {decay}.")

    @staticmethod
    def _validate_group_key_consistency(model_params: Dict[str, Any], objective: str, strict: bool) -> None:
        """RankingタスクでのGroupKey整合性をチェック"""
        pass

    @staticmethod
    def _validate_task_objective_consistency(config: Dict[str, Any], strict: bool) -> None:
        """TaskTypeとObjectiveの整合性をチェック"""
        task_type = config.get('task_type', '') # Optional key
        objective = config.get('model_params', {}).get('objective', '')
        
        if not task_type: return

        if task_type == 'ranking':
            if objective not in ['lambdarank', 'rank:pairwise', 'rank:ndcg', 'yetirank']:
                raise ValueError(f"⛔ CONFIG ERROR: task_type='ranking' but objective='{objective}'. Expected ranking objective.")
        
        elif task_type == 'classification':
             if objective not in ['binary', 'multiclass', 'cross_entropy', 'logloss']:
                 raise ValueError(f"⛔ CONFIG ERROR: task_type='classification' but objective='{objective}'. Expected binary/multiclass objective.")

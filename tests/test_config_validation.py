
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.config.validator import ConfigValidator

class TestConfigValidator(unittest.TestCase):
    
    def test_target_consistency_valid(self):
        """正しい設定はパスすること"""
        config = {
            'experiment_name': 'exp_top3_model',
            'dataset': {'binary_target': 'top3'}
        }
        try:
            ConfigValidator.validate(config)
        except ValueError as e:
            self.fail(f"Valid config raised ValueError: {e}")

    def test_target_consistency_invalid_m4a_incident(self):
        """M4-Aの実例: Top3名目で Top2ターゲットはエラーになること"""
        config = {
            'experiment_name': 'exp_m4_a_top3_temporal',
            'dataset': {
                'target_col': 'target_top2', # Mismatch!
                'binary_target': 'top2'
            }
        }
        with self.assertRaises(ValueError) as context:
            ConfigValidator.validate(config)
        
        self.assertIn("Name 'exp_m4_a_top3_temporal' contains 'top3' but target is 'target_top2'", str(context.exception))

    def test_target_consistency_win(self):
        """Winモデルのチェック"""
        config = {
            'experiment_name': 'phase_m_win_model',
            'dataset': {'binary_target': 'top2'} # Mismatch
        }
        with self.assertRaises(ValueError):
            ConfigValidator.validate(config)

    def test_filename_consistency(self):
        """ファイル名(Basename)もチェックされること"""
        config = {
            'experiment_name': 'exp_generic_name', # Generic name
            'dataset': {'binary_target': 'top2'}
        }
        # Filename implies Top3 -> Mismatch with Top2 target
        with self.assertRaises(ValueError) as context:
            ConfigValidator.validate(config, config_path="config/experiments/exp_m4_top3_test.yaml")
        
        self.assertIn("Name 'exp_m4_top3_test.yaml' contains 'top3'", str(context.exception))

    def test_metric_warning_strict(self):
        """StrictモードならMetric WarningがErrorになること"""
        config = {
            'experiment_name': 'exp_lambda_test',
            'model_params': {'objective': 'lambdarank', 'metric': ['rmse']}, # Bad metric
            'dataset': {}
        }
        # Non-strict: Pass (Warning only)
        try:
            ConfigValidator.validate(config, strict=False)
        except ValueError:
            self.fail("Non-strict validation raised ValueError for metric warning")

        # Strict: Fail
        with self.assertRaises(ValueError):
            ConfigValidator.validate(config, strict=True)
            
    def test_task_objective_consistency(self):
        """TaskTypeとObjectiveの不整合"""
        config = {
            'task_type': 'ranking',
            'model_params': {'objective': 'binary'} # Mismatch
        }
        with self.assertRaises(ValueError):
            ConfigValidator.validate(config)

    def test_time_decay_invalid(self):
        """Time Decay設定不備"""
        config = {
            'experiment_name': 'exp_test',
            'sample_weight': {
                'enabled': True,
                'strategy': 'exponential',
                'decay_rate': -0.1 # Invalid
            }
        }
        with self.assertRaises(ValueError):
            ConfigValidator.validate(config)

if __name__ == '__main__':
    unittest.main()

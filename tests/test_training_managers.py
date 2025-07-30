"""
Training Managers Test Suite

Tests for the consolidated training manager architecture:
- BaseManager abstract functionality
- TrainingStateManager consolidation validation  
- LossManager simplification verification
- BBUTrainer integration with new managers
"""

import inspect
import sys
import unittest
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger_utils import get_logger
from tests.fixtures import TestUtils


logger = get_logger("test_training_managers")


class TestTrainingManagerConsolidation(unittest.TestCase):
    """Test suite for training manager consolidation validation."""

    def test_manager_imports(self):
        """Test that new consolidated managers can be imported successfully."""
        logger.info("Testing Manager Imports")
        
        # Test BaseManager import
        from src.training.base_manager import BaseManager
        self.assertIsNotNone(BaseManager, "BaseManager should be importable")
        
        # Test TrainingStateManager import
        from src.training.training_state_manager import TrainingStateManager
        self.assertIsNotNone(TrainingStateManager, "TrainingStateManager should be importable")
        
        # Test simplified LossManager import
        from src.training.loss_manager import LossManager
        self.assertIsNotNone(LossManager, "LossManager should be importable")
        
        # Test that old managers are eliminated
        old_managers = [
            'dataloader_manager',
            'evaluation_manager', 
            'metrics_manager',
            'parameter_manager'
        ]
        
        for manager_name in old_managers:
            with self.assertRaises(ImportError, msg=f"{manager_name} should be eliminated"):
                __import__(f'src.training.{manager_name}', fromlist=[manager_name])

    def test_base_manager_abstraction(self):
        """Test BaseManager abstract class functionality."""
        logger.info("Testing BaseManager Abstraction")
        
        from src.training.base_manager import BaseManager
        
        # Test that BaseManager cannot be instantiated directly
        with self.assertRaises(TypeError, msg="BaseManager should be abstract"):
            BaseManager(config={"test": "config"})
        
        # Test abstract methods are defined
        abstract_methods = BaseManager.__abstractmethods__
        self.assertTrue(len(abstract_methods) > 0, "BaseManager should have abstract methods")

    def test_training_state_manager_consolidation(self):
        """Test that TrainingStateManager consolidates multiple manager functionalities."""
        logger.info("Testing TrainingStateManager Consolidation")
        
        from src.training.training_state_manager import TrainingStateManager
        
        # Check that TrainingStateManager has methods from consolidated managers
        expected_methods = [
            # From MetricsManager
            'log_training_metrics',
            'log_metrics_batch', 
            'reset_metrics_state',
            'cache_norm_metrics',
            
            # From EvaluationManager
            'run_evaluation',
            'predict_batch',
            
            # From ParameterManager
            'create_optimizer_groups',
        ]
        
        tsm_methods = [method for method in dir(TrainingStateManager) 
                      if not method.startswith('_')]
        
        missing_methods = []
        for method in expected_methods:
            if method not in tsm_methods:
                missing_methods.append(method)
        
        self.assertEqual(len(missing_methods), 0, 
                        f"TrainingStateManager missing methods: {missing_methods}")
        
        # Check constructor parameters
        sig = inspect.signature(TrainingStateManager.__init__)
        expected_params = ['config', 'model', 'trainer', 'training_coordinator']
        
        for param in expected_params:
            self.assertIn(param, sig.parameters, 
                         f"TrainingStateManager missing constructor parameter: {param}")

    def test_loss_manager_simplification(self):
        """Test that LossManager is simplified and focused."""
        logger.info("Testing LossManager Simplification")
        
        from src.training.loss_manager import LossManager
        
        # Check that LossManager has core loss computation methods
        expected_methods = [
            'compute_total_loss',
            '_extract_total_loss',
            '_extract_loss_components',
        ]
        
        lm_methods = [method for method in dir(LossManager) 
                     if not method.startswith('__')]
        
        for method in expected_methods:
            self.assertIn(method, lm_methods, 
                         f"LossManager missing core method: {method}")
        
        # Check constructor is simplified (reasonable parameter count)
        sig = inspect.signature(LossManager.__init__)
        param_count = len([p for p in sig.parameters.values() 
                          if p.name != 'self'])
        
        self.assertLessEqual(param_count, 10, 
                           f"LossManager constructor has too many parameters ({param_count})")

    def test_trainer_integration(self):
        """Test that BBUTrainer integrates with new managers."""
        logger.info("Testing BBUTrainer Integration")
        
        from src.training.trainer import BBUTrainer
        
        # Check that BBUTrainer imports work
        source = inspect.getsourcefile(BBUTrainer)
        with open(source, 'r') as f:
            trainer_source = f.read()
        
        # Check for new manager imports
        self.assertIn('from src.training.training_state_manager import TrainingStateManager', 
                     trainer_source, "BBUTrainer should import TrainingStateManager")
        
        # Check that old manager references are gone
        old_manager_refs = [
            'self.metrics_manager',
            'self.evaluation_manager', 
            'self.parameter_manager',
            'self.dataloader_manager'
        ]
        
        for ref in old_manager_refs:
            self.assertNotIn(ref, trainer_source, 
                           f"BBUTrainer should not have old manager reference: {ref}")
        
        # Check for new manager usage
        self.assertIn('self.training_state_manager', trainer_source,
                     "BBUTrainer should use TrainingStateManager")


class TestManagerArchitecture(unittest.TestCase):
    """Test overall manager architecture consistency."""

    def test_manager_hierarchy(self):
        """Test that manager hierarchy is correctly implemented."""
        from src.training.base_manager import BaseManager
        from src.training.training_state_manager import TrainingStateManager
        from src.training.loss_manager import LossManager
        
        # Test inheritance relationships
        self.assertTrue(issubclass(TrainingStateManager, BaseManager),
                       "TrainingStateManager should inherit from BaseManager")
        self.assertTrue(issubclass(LossManager, BaseManager),
                       "LossManager should inherit from BaseManager")

    def test_manager_initialization_contracts(self):
        """Test that managers follow consistent initialization contracts."""
        from src.training.training_state_manager import TrainingStateManager
        from src.training.loss_manager import LossManager
        
        # Both managers should accept config parameter
        tsm_sig = inspect.signature(TrainingStateManager.__init__)
        lm_sig = inspect.signature(LossManager.__init__)
        
        self.assertIn('config', tsm_sig.parameters, 
                     "TrainingStateManager should accept config parameter")
        self.assertIn('config', lm_sig.parameters, 
                     "LossManager should accept config parameter")


if __name__ == "__main__":
    unittest.main()
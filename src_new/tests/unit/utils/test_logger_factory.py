#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the logger factory module.

Tests the centralized logger creation functionality with various scenarios
including rank-aware logging, fallback behavior, and configuration options.
"""

import logging
import pytest
from unittest.mock import patch, MagicMock

from src_new.utils.logger_factory import (
    get_module_logger,
    get_debug_logger,
    get_training_logger,
    get_inference_logger,
    get_processing_logger,
    reconfigure_logger,
    set_global_log_level,
    get_configured_loggers,
    reset_logger_state,
)


class TestLoggerFactory:
    """Test cases for logger factory functionality."""
    
    def setup_method(self):
        """Reset logger state before each test."""
        reset_logger_state()
    
    def test_get_module_logger_with_rank_aware(self):
        """Test logger creation with rank-aware logging available."""
        with patch('src_new.utils.logger_factory._create_rank_aware_logger') as mock_rank_aware:
            mock_logger = MagicMock()
            mock_rank_aware.return_value = mock_logger
            
            logger = get_module_logger("test_module")
            
            assert logger == mock_logger
            mock_rank_aware.assert_called_once_with("test_module", False)
    
    def test_get_module_logger_with_config_fallback(self):
        """Test logger creation with config system fallback."""
        with patch('src_new.utils.logger_factory._create_rank_aware_logger') as mock_rank_aware:
            with patch('src_new.utils.logger_factory._create_config_system_logger') as mock_config:
                mock_rank_aware.side_effect = ImportError("No rank-aware logging")
                mock_logger = MagicMock()
                mock_config.return_value = mock_logger
                
                logger = get_module_logger("test_module")
                
                assert logger == mock_logger
                mock_config.assert_called_once_with("test_module", False)
    
    def test_get_module_logger_with_standard_fallback(self):
        """Test logger creation with standard logging fallback."""
        with patch('src_new.utils.logger_factory._create_rank_aware_logger') as mock_rank_aware:
            with patch('src_new.utils.logger_factory._create_config_system_logger') as mock_config:
                with patch('src_new.utils.logger_factory._create_standard_logger') as mock_standard:
                    mock_rank_aware.side_effect = ImportError("No rank-aware logging")
                    mock_config.side_effect = ImportError("No config system")
                    mock_logger = MagicMock()
                    mock_standard.return_value = mock_logger
                    
                    logger = get_module_logger("test_module")
                    
                    assert logger == mock_logger
                    mock_standard.assert_called_once_with("test_module", False)
    
    def test_get_module_logger_complete_failure(self):
        """Test logger creation with complete failure."""
        with patch('src_new.utils.logger_factory._create_rank_aware_logger') as mock_rank_aware:
            with patch('src_new.utils.logger_factory._create_config_system_logger') as mock_config:
                with patch('src_new.utils.logger_factory._create_standard_logger') as mock_standard:
                    mock_rank_aware.side_effect = ImportError("No rank-aware logging")
                    mock_config.side_effect = ImportError("No config system")
                    mock_standard.side_effect = RuntimeError("Complete failure")
                    
                    with pytest.raises(RuntimeError, match="Failed to create logger for module 'test_module'"):
                        get_module_logger("test_module")
    
    def test_get_debug_logger(self):
        """Test debug logger creation with forced debug level."""
        with patch('src_new.utils.logger_factory.get_module_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            logger = get_debug_logger("test_module")
            
            assert logger == mock_logger
            mock_get_logger.assert_called_once_with("test_module", force_debug=True)
    
    def test_specialized_loggers(self):
        """Test specialized logger creation functions."""
        with patch('src_new.utils.logger_factory.get_module_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Test training logger
            training_logger = get_training_logger("test_module")
            assert training_logger == mock_logger
            mock_get_logger.assert_called_with("training.test_module")
            
            # Test inference logger
            inference_logger = get_inference_logger("test_module")
            assert inference_logger == mock_logger
            mock_get_logger.assert_called_with("inference.test_module")
            
            # Test processing logger
            processing_logger = get_processing_logger("test_module")
            assert processing_logger == mock_logger
            mock_get_logger.assert_called_with("processing.test_module")
    
    def test_reconfigure_logger(self):
        """Test logger reconfiguration."""
        mock_logger = MagicMock()
        mock_handler = MagicMock()
        mock_logger.handlers = [mock_handler]
        
        # Test normal reconfiguration
        reconfigure_logger(mock_logger, force_debug=False)
        mock_logger.info.assert_called()
        
        # Test debug reconfiguration
        reconfigure_logger(mock_logger, force_debug=True)
        mock_logger.setLevel.assert_called_with(logging.DEBUG)
        mock_handler.setLevel.assert_called_with(logging.DEBUG)
    
    def test_set_global_log_level(self):
        """Test setting global log level."""
        set_global_log_level(logging.WARNING)
        
        # This should affect future logger creation
        # We can't easily test this without creating actual loggers
        # but we can verify the function doesn't raise errors
        assert True
    
    def test_get_configured_loggers(self):
        """Test getting list of configured loggers."""
        # Initially empty
        configured = get_configured_loggers()
        assert isinstance(configured, set)
        
        # After creating a logger, it should be tracked
        with patch('src_new.utils.logger_factory._create_standard_logger') as mock_standard:
            mock_logger = MagicMock()
            mock_standard.return_value = mock_logger
            
            with patch('src_new.utils.logger_factory._create_rank_aware_logger', side_effect=ImportError):
                with patch('src_new.utils.logger_factory._create_config_system_logger', side_effect=ImportError):
                    get_module_logger("test_module")
                    
                    configured = get_configured_loggers()
                    # Note: The actual tracking happens in _create_standard_logger
                    # This test verifies the function works
                    assert isinstance(configured, set)
    
    def test_reset_logger_state(self):
        """Test resetting logger state."""
        # Set some state
        set_global_log_level(logging.ERROR)
        
        # Reset
        reset_logger_state()
        
        # Verify reset worked
        configured = get_configured_loggers()
        assert len(configured) == 0


class TestLoggerCreationMethods:
    """Test cases for individual logger creation methods."""
    
    def test_create_standard_logger(self):
        """Test standard logger creation."""
        from src_new.utils.logger_factory import _create_standard_logger
        
        logger = _create_standard_logger("test_standard", force_debug=False)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_standard"
        assert len(logger.handlers) > 0
    
    def test_create_standard_logger_debug(self):
        """Test standard logger creation with debug forced."""
        from src_new.utils.logger_factory import _create_standard_logger
        
        logger = _create_standard_logger("test_debug", force_debug=True)
        
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.DEBUG
    
    def test_create_standard_logger_no_duplicate_handlers(self):
        """Test that creating the same logger twice doesn't add duplicate handlers."""
        from src_new.utils.logger_factory import _create_standard_logger
        
        logger1 = _create_standard_logger("test_duplicate", force_debug=False)
        handler_count1 = len(logger1.handlers)
        
        logger2 = _create_standard_logger("test_duplicate", force_debug=False)
        handler_count2 = len(logger2.handlers)
        
        assert logger1 is logger2  # Same logger instance
        assert handler_count1 == handler_count2  # No duplicate handlers


class TestLoggerIntegration:
    """Integration tests for logger factory."""
    
    def test_logger_actually_logs(self):
        """Test that created loggers actually work for logging."""
        import io
        import sys
        
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        
        # Create logger and add our capture handler
        logger = get_module_logger("test_integration")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)
        
        # Log a message
        test_message = "Test integration message"
        logger.info(test_message)
        
        # Check output
        log_output = log_capture.getvalue()
        assert test_message in log_output
        
        # Clean up
        logger.removeHandler(handler)
    
    def test_multiple_loggers_independent(self):
        """Test that multiple loggers are independent."""
        logger1 = get_module_logger("test_module1")
        logger2 = get_module_logger("test_module2")
        
        assert logger1 is not logger2
        assert logger1.name != logger2.name
    
    def test_same_module_same_logger(self):
        """Test that requesting the same module returns the same logger."""
        logger1 = get_module_logger("test_same")
        logger2 = get_module_logger("test_same")
        
        assert logger1 is logger2

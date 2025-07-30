"""
Integration Tests for Multi-Geometry Token Parser in Inference Engine

Tests the integration of multi-geometry token parsing with the inference engine,
including configuration detection and end-to-end response processing.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger_utils import configure_global_logging, get_logger


logger = get_logger("test_inference_multi_geometry")


class TestInferenceMultiGeometry:
    """Integration tests for multi-geometry token parsing in inference engine."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        configure_global_logging(rank=0, world_size=1)

    def create_mock_inference_engine(
        self, coordinate_tokens_enabled=False, has_coord_tokens_in_vocab=False
    ):
        """Create a mock inference engine for testing."""
        # Mock the inference engine
        mock_engine = Mock()
        mock_engine.coordinate_tokens_enabled = coordinate_tokens_enabled

        # Mock model
        mock_model = Mock()
        mock_model.coordinate_tokens_enabled = coordinate_tokens_enabled
        mock_model.config = Mock()
        mock_model.config.coordinate_tokens_enabled = coordinate_tokens_enabled
        mock_engine.model = mock_model

        # Mock processor and tokenizer
        mock_processor = Mock()
        mock_tokenizer = Mock()
        vocab = {"<|coord_0|>": 12345} if has_coord_tokens_in_vocab else {}
        mock_tokenizer.get_vocab.return_value = vocab
        mock_processor.tokenizer = mock_tokenizer
        mock_engine.processor = mock_processor

        # Import the actual methods we want to test
        from src.inference import InferenceEngine

        mock_engine._detect_coordinate_tokens_enabled = (
            InferenceEngine._detect_coordinate_tokens_enabled.__get__(mock_engine)
        )
        mock_engine._parse_multi_geometry_response = (
            InferenceEngine._parse_multi_geometry_response.__get__(mock_engine)
        )
        mock_engine._process_model_response = (
            InferenceEngine._process_model_response.__get__(mock_engine)
        )

        return mock_engine

    def test_detect_coordinate_tokens_enabled_via_attribute(self):
        """Test coordinate token detection via engine attribute."""
        engine = self.create_mock_inference_engine(coordinate_tokens_enabled=True)

        result = engine._detect_coordinate_tokens_enabled()
        assert result is True

    def test_detect_coordinate_tokens_enabled_via_model_attribute(self):
        """Test coordinate token detection via model attribute."""
        engine = self.create_mock_inference_engine()
        # Remove engine attribute but keep model attribute
        del engine.coordinate_tokens_enabled
        engine.model.coordinate_tokens_enabled = True

        result = engine._detect_coordinate_tokens_enabled()
        assert result is True

    def test_detect_coordinate_tokens_enabled_via_model_config(self):
        """Test coordinate token detection via model config."""
        engine = self.create_mock_inference_engine()
        # Remove engine and model attributes but keep config
        del engine.coordinate_tokens_enabled
        del engine.model.coordinate_tokens_enabled
        engine.model.config.coordinate_tokens_enabled = True

        result = engine._detect_coordinate_tokens_enabled()
        assert result is True

    def test_detect_coordinate_tokens_enabled_via_tokenizer_vocab(self):
        """Test coordinate token detection via tokenizer vocabulary."""
        engine = self.create_mock_inference_engine(has_coord_tokens_in_vocab=True)
        # Remove all attributes, rely on vocab detection
        del engine.coordinate_tokens_enabled
        del engine.model.coordinate_tokens_enabled
        del engine.model.config.coordinate_tokens_enabled

        result = engine._detect_coordinate_tokens_enabled()
        assert result is True

    def test_detect_coordinate_tokens_enabled_via_global_config(self):
        """Test coordinate token detection via global config."""
        engine = self.create_mock_inference_engine()
        # Remove all attributes
        del engine.coordinate_tokens_enabled
        del engine.model.coordinate_tokens_enabled
        del engine.model.config.coordinate_tokens_enabled
        engine.processor.tokenizer.get_vocab.return_value = {}  # No coord tokens in vocab

        # Mock the import and config by patching the import statement
        mock_config = Mock()
        mock_config.coordinate_tokens_enabled = True

        # Patch the import at the module level where it's used
        with patch.dict("sys.modules", {"src.config": mock_config}):
            result = engine._detect_coordinate_tokens_enabled()
            assert result is True

    def test_detect_coordinate_tokens_enabled_default_false(self):
        """Test coordinate token detection defaults to False when not found."""
        engine = self.create_mock_inference_engine()
        # Remove all attributes and ensure no coord tokens in vocab
        del engine.coordinate_tokens_enabled
        del engine.model.coordinate_tokens_enabled
        del engine.model.config.coordinate_tokens_enabled
        engine.processor.tokenizer.get_vocab.return_value = {}

        # Mock import error for config
        with patch("builtins.__import__", side_effect=ImportError):
            result = engine._detect_coordinate_tokens_enabled()
            assert result is False

    def test_parse_multi_geometry_response_with_coordinates(self):
        """Test parsing multi-geometry response with coordinate tokens enabled."""
        engine = self.create_mock_inference_engine(coordinate_tokens_enabled=True)
        response = "<obj_ref_start><bbox_2d_start>BBU设备<|coord_150|><|coord_10|><|coord_211|><|coord_35|><bbox_2d_end><obj_ref_end>"

        objects = engine._parse_multi_geometry_response(response)

        assert len(objects) == 1
        obj = objects[0]
        assert obj["geometry_type"] == "bbox_2d"
        assert obj["caption"] == "BBU设备"
        assert obj["coordinates"] == [150, 10, 211, 35]
        assert obj["formatted_output"] == "bbox_2d:BBU设备[150,10,211,35]"

    def test_parse_multi_geometry_response_without_coordinates(self):
        """Test parsing multi-geometry response with coordinate tokens disabled."""
        engine = self.create_mock_inference_engine(coordinate_tokens_enabled=False)
        response = "<obj_ref_start><bbox_2d_start>BBU设备<bbox_2d_end><obj_ref_end>"

        objects = engine._parse_multi_geometry_response(response)

        assert len(objects) == 1
        obj = objects[0]
        assert obj["geometry_type"] == "bbox_2d"
        assert obj["caption"] == "BBU设备"
        assert obj["coordinates"] == []
        assert obj["formatted_output"] == "bbox_2d:BBU设备"

    def test_parse_multi_geometry_response_no_tokens(self):
        """Test parsing response with no multi-geometry tokens."""
        engine = self.create_mock_inference_engine()
        response = "This is just regular text without any special tokens"

        objects = engine._parse_multi_geometry_response(response)

        assert len(objects) == 0

    def test_parse_multi_geometry_response_error_handling(self):
        """Test error handling in multi-geometry response parsing."""
        engine = self.create_mock_inference_engine()

        # Test with None response
        objects = engine._parse_multi_geometry_response(None)
        assert len(objects) == 0

        # Test with empty response
        objects = engine._parse_multi_geometry_response("")
        assert len(objects) == 0

    def test_process_model_response_single_object(self):
        """Test processing model response with single multi-geometry object."""
        engine = self.create_mock_inference_engine(coordinate_tokens_enabled=True)
        response = "<obj_ref_start><bbox_2d_start>BBU设备<|coord_150|><|coord_10|><|coord_211|><|coord_35|><bbox_2d_end><obj_ref_end>"

        result = engine._process_model_response(response)

        assert result == "bbox_2d:BBU设备[150,10,211,35]"

    def test_process_model_response_multiple_objects(self):
        """Test processing model response with multiple multi-geometry objects."""
        engine = self.create_mock_inference_engine(coordinate_tokens_enabled=True)
        response = (
            "<obj_ref_start><bbox_2d_start>BBU设备<|coord_150|><|coord_10|><|coord_211|><|coord_35|><bbox_2d_end><obj_ref_end>"
            "<obj_ref_start><line_start>光纤<|coord_100|><|coord_200|><|coord_150|><|coord_250|><line_end><obj_ref_end>"
        )

        result = engine._process_model_response(response)

        expected = "bbox_2d:BBU设备[150,10,211,35] | line:光纤[100,200,150,250]"
        assert result == expected

    def test_process_model_response_fallback_to_raw(self):
        """Test fallback to raw response when no multi-geometry tokens found."""
        engine = self.create_mock_inference_engine()
        response = "This is just regular text"

        result = engine._process_model_response(response)

        assert result == response  # Should return raw response

    def test_process_model_response_no_coordinate_detection(self):
        """Test processing when coordinate token detection is not available."""
        engine = self.create_mock_inference_engine()
        # Remove coordinate_tokens_enabled attribute to simulate detection failure
        del engine.coordinate_tokens_enabled

        response = "Some response text"

        with patch.object(
            engine,
            "_detect_coordinate_tokens_enabled",
            side_effect=Exception("Detection failed"),
        ):
            result = engine._process_model_response(response)
            assert result == response  # Should fallback to raw response

    @pytest.mark.parametrize(
        "coordinate_enabled,expected_format",
        [
            (True, "bbox_2d:BBU设备[150,10,211,35]"),
            (False, "bbox_2d:BBU设备"),
        ],
    )
    def test_end_to_end_processing(self, coordinate_enabled, expected_format):
        """Test end-to-end processing with different coordinate token configurations."""
        engine = self.create_mock_inference_engine(
            coordinate_tokens_enabled=coordinate_enabled
        )

        if coordinate_enabled:
            response = "<obj_ref_start><bbox_2d_start>BBU设备<|coord_150|><|coord_10|><|coord_211|><|coord_35|><bbox_2d_end><obj_ref_end>"
        else:
            response = "<obj_ref_start><bbox_2d_start>BBU设备<bbox_2d_end><obj_ref_end>"

        result = engine._process_model_response(response)

        assert result == expected_format

    def test_mixed_coordinate_and_caption_only_objects(self):
        """Test processing response with mixed coordinate and caption-only objects."""
        engine = self.create_mock_inference_engine(coordinate_tokens_enabled=True)

        # Mix of objects with and without coordinates
        response = (
            "<obj_ref_start><bbox_2d_start>BBU设备<|coord_150|><|coord_10|><|coord_211|><|coord_35|><bbox_2d_end><obj_ref_end>"
            "<obj_ref_start><line_start>光纤<line_end><obj_ref_end>"  # No coordinates
        )

        result = engine._process_model_response(response)

        expected = "bbox_2d:BBU设备[150,10,211,35] | line:光纤"
        assert result == expected

    def test_parser_configuration_propagation(self):
        """Test that coordinate token configuration is properly propagated to parser."""
        engine = self.create_mock_inference_engine(coordinate_tokens_enabled=True)
        response = "<obj_ref_start><bbox_2d_start>test<bbox_2d_end><obj_ref_end>"

        with patch("src.utils.response_parser.ResponseParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser._parse_multi_geometry_tokens.return_value = []
            mock_parser_class.return_value = mock_parser

            engine._parse_multi_geometry_response(response)

            # Verify parser was configured with coordinate tokens enabled
            assert hasattr(mock_parser, "_coordinate_tokens_enabled")
            assert mock_parser._coordinate_tokens_enabled is True

            # Verify the parsing method was called with correct parameters
            mock_parser._parse_multi_geometry_tokens.assert_called_once_with(
                response, True
            )

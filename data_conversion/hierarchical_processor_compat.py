#!/usr/bin/env python3
"""
Hierarchical Processor Compatibility Layer

Clean geometry-native processing for V2 data without backward compatibility bloat.
"""

import logging
import sys
from typing import Dict, List

from data_conversion.flexible_taxonomy_processor import FlexibleTaxonomyProcessor

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class HierarchicalProcessor:
    """Clean processor for V2 data with native geometry output."""
    
    def __init__(self, response_types = None, label_hierarchy: Dict = None):
        """Initialize processor for Chinese-only processing."""
        self.response_types = response_types or {"object_type", "property", "extra_info"}
        self.label_hierarchy = label_hierarchy or {}
        
        # Create flexible processor
        self.flexible_processor = FlexibleTaxonomyProcessor()
        
        logger.info("Initialized HierarchicalProcessor for Chinese-only processing")
    
    def extract_objects_from_markresult(self, features: List[Dict]) -> List[Dict]:
        """
        Extract objects from markResult features with native geometry types.
        
        Returns objects in format:
        [
            {'bbox_2d': [x1,y1,x2,y2], 'desc': '...'}, 
            {'square': [x1,y1,x2,y2,x3,y3,x4,y4], 'desc': '...'},
            {'line': [x1,y1,x2,y2,...], 'desc': '...'}
        ]
        """
        objects = []
        
        for feature in features:
            # Process with flexible processor
            sample = self.flexible_processor.process_v2_feature(feature)
            if not sample:
                continue
            
            # Convert to clean training format (native geometry only)
            training_obj = sample.to_training_format()
            objects.append(training_obj)
        
        return objects


# For import compatibility
def create_hierarchical_processor(*args, **kwargs):
    """Factory function for compatibility."""
    return HierarchicalProcessor(*args, **kwargs)
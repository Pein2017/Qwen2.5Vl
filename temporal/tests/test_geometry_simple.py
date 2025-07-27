#!/root/miniconda3/envs/ms/bin/python
"""
Simple test to verify the geometry token fix works
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

def test_span_detection_logic():
    """Test the geometry span detection logic without full model loading."""
    
    print("🔍 Testing geometry span detection logic...")
    
    # Mock coordinate token manager with the fixed detection logic
    class MockCoordinateManager:
        def __init__(self):
            # Simulate the geometry token IDs after SimpleTokenManager update
            self.geometry_token_ids = {
                "box_start": 151648,
                "box_end": 151649, 
                "square_start": 151665 + 2048,  # After coordinate tokens
                "square_end": 151665 + 2048 + 1,
                "line_start": 151665 + 2048 + 2,
                "line_end": 151665 + 2048 + 3,
            }
            
        def is_coordinate_token(self, token_id):
            """Check if token is a coordinate token."""
            # Coordinate tokens are in range [151665, 151665+2048)
            return 151665 <= token_id < (151665 + 2048)
            
        def detect_geometry_spans(self, input_ids):
            """The fixed geometry span detection (copy from updated code)."""
            import torch
            
            geometry_spans = []
            
            for batch_idx in range(input_ids.shape[0]):
                batch_spans = []
                seq = input_ids[batch_idx]

                i = 0
                while i < len(seq):
                    # Look for any geometry start tokens
                    detected_geometry = None
                    start_token_id = seq[i].item() if hasattr(seq[i], 'item') else seq[i]
                    
                    # Check all geometry start tokens
                    for geom_name, token_id in self.geometry_token_ids.items():
                        if geom_name.endswith("_start") and start_token_id == token_id:
                            detected_geometry = geom_name.replace("_start", "")
                            break
                    
                    if detected_geometry:
                        start_idx = i
                        # Get corresponding end token
                        end_token_id = self.geometry_token_ids[f"{detected_geometry}_end"]

                        # Look for matching end token
                        j = i + 1
                        coord_count = 0
                        while j < len(seq):
                            token_id = seq[j].item() if hasattr(seq[j], 'item') else seq[j]
                            if token_id == end_token_id:
                                break
                            # Count coordinate tokens within the span
                            if self.is_coordinate_token(seq[j]):
                                coord_count += 1
                            j += 1

                        if j < len(seq):  # Found matching end token
                            # Validate coordinate count matches geometry type
                            geometry_type = detected_geometry.replace("box", "bbox")  # normalize naming
                            
                            if detected_geometry == "box":
                                geometry_type = "bbox"
                            elif detected_geometry == "square" and coord_count == 8:
                                geometry_type = "square"
                            elif detected_geometry == "line" and coord_count >= 6 and coord_count % 2 == 0:
                                geometry_type = "line"
                            else:
                                # Fallback: determine by coordinate count
                                if coord_count == 4:
                                    geometry_type = "bbox"
                                elif coord_count == 8:
                                    geometry_type = "square"
                                elif coord_count >= 6 and coord_count % 2 == 0:
                                    geometry_type = "line"
                                else:
                                    print(f"⚠️ Invalid coordinate count {coord_count} for {detected_geometry}, treating as bbox")
                                    geometry_type = "bbox"

                            batch_spans.append((start_idx, j + 1, geometry_type))
                            print(f"   ✅ Detected {geometry_type} span: [{start_idx}:{j + 1}] with {coord_count} coords")
                            i = j + 1
                        else:
                            i += 1
                    else:
                        i += 1

                geometry_spans.append(batch_spans)

            return geometry_spans
    
    # Test with mock sequences
    import torch
    
    manager = MockCoordinateManager()
    
    print(f"\n📋 Mock geometry token IDs:")
    for name, token_id in manager.geometry_token_ids.items():
        print(f"   {name}: {token_id}")
    
    # Test different geometry sequences
    print(f"\n🧪 Testing geometry span detection:")
    
    # 1. Square sequence (8 coordinates)
    square_seq = torch.tensor([[
        manager.geometry_token_ids["square_start"],
        151665, 151666, 151667, 151668, 151669, 151670, 151671, 151672,  # 8 coord tokens
        manager.geometry_token_ids["square_end"]
    ]])
    
    print(f"\n🟩 Testing square sequence...")
    square_spans = manager.detect_geometry_spans(square_seq)
    print(f"   Result: {square_spans}")
    
    # 2. Line sequence (6 coordinates)
    line_seq = torch.tensor([[
        manager.geometry_token_ids["line_start"],
        151665, 151666, 151667, 151668, 151669, 151670,  # 6 coord tokens
        manager.geometry_token_ids["line_end"]
    ]])
    
    print(f"\n🟦 Testing line sequence...")
    line_spans = manager.detect_geometry_spans(line_seq)
    print(f"   Result: {line_spans}")
    
    # 3. Box sequence (4 coordinates)
    box_seq = torch.tensor([[
        manager.geometry_token_ids["box_start"],
        151665, 151666, 151667, 151668,  # 4 coord tokens
        manager.geometry_token_ids["box_end"]
    ]])
    
    print(f"\n🟨 Testing box sequence...")
    box_spans = manager.detect_geometry_spans(box_seq)
    print(f"   Result: {box_spans}")
    
    # Verify results
    print(f"\n📊 Test Results:")
    
    success_count = 0
    
    if square_spans and len(square_spans[0]) > 0 and square_spans[0][0][2] == "square":
        print(f"   ✅ Square detection: PASS")
        success_count += 1
    else:
        print(f"   ❌ Square detection: FAIL")
    
    if line_spans and len(line_spans[0]) > 0 and line_spans[0][0][2] == "line":
        print(f"   ✅ Line detection: PASS")
        success_count += 1
    else:
        print(f"   ❌ Line detection: FAIL")
        
    if box_spans and len(box_spans[0]) > 0 and box_spans[0][0][2] == "bbox":
        print(f"   ✅ Box detection: PASS") 
        success_count += 1
    else:
        print(f"   ❌ Box detection: FAIL")
    
    print(f"\n🎯 Overall: {success_count}/3 tests passed")
    
    if success_count == 3:
        print(f"✅ All geometry types can now be detected! Square and line losses should work.")
    else:
        print(f"❌ Some geometry detection failed. Need to debug further.")

if __name__ == "__main__":
    test_span_detection_logic()
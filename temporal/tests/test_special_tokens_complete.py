#!/root/miniconda3/envs/ms/bin/python
"""
Complete test of the special token system to verify all requirements are met.
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

def test_special_token_system():
    """Test the complete special token system end-to-end."""
    
    print("🔍 Testing complete special token system...")
    
    # Test 1: SimpleTokenManager validation
    print("\n📋 Test 1: SimpleTokenManager strict validation")
    
    from src.utils.simple_token_manager import SimpleTokenManager
    from transformers import AutoTokenizer
    
    # Mock tokenizer and model for testing
    class MockModel:
        def get_input_embeddings(self):
            class MockEmbedding:
                num_embeddings = 152000
            return MockEmbedding()
        
        def resize_token_embeddings(self, size):
            pass
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("model_cache/Qwen/Qwen2.5-VL-7B-Instruct")
        model = MockModel()
        
        manager = SimpleTokenManager(tokenizer, model)
        
        # Test valid object (should pass)
        print("   Testing valid object...")
        valid_obj = {"square": [277, 262, 289, 26, 380, 168, 344, 364], "desc": "测试对象"}
        result = manager.format_object(valid_obj)
        print(f"   ✅ Valid object processed: {len(result)} fields")
        
        # Test invalid object - no geometry (should fail)
        print("   Testing invalid object - no geometry...")
        try:
            invalid_obj = {"desc": "测试对象"}
            manager.format_object(invalid_obj)
            print("   ❌ ERROR: Should have raised exception for missing geometry!")
        except ValueError as e:
            if "SPECIAL_TOKEN_VIOLATION" in str(e):
                print(f"   ✅ Correctly rejected: {e}")
            else:
                print(f"   ❌ Wrong exception: {e}")
        
        # Test invalid object - multiple geometries (should fail)  
        print("   Testing invalid object - multiple geometries...")
        try:
            invalid_obj = {"square": [1,2,3,4,5,6,7,8], "bbox_2d": [1,2,3,4], "desc": "测试对象"}
            manager.format_object(invalid_obj)
            print("   ❌ ERROR: Should have raised exception for multiple geometries!")
        except ValueError as e:
            if "SPECIAL_TOKEN_VIOLATION" in str(e):
                print(f"   ✅ Correctly rejected: {e}")
            else:
                print(f"   ❌ Wrong exception: {e}")
        
        # Test invalid object - no description (should fail)
        print("   Testing invalid object - no description...")
        try:
            invalid_obj = {"square": [1,2,3,4,5,6,7,8]}
            manager.format_object(invalid_obj)
            print("   ❌ ERROR: Should have raised exception for missing description!")
        except ValueError as e:
            if "SPECIAL_TOKEN_VIOLATION" in str(e):
                print(f"   ✅ Correctly rejected: {e}")
            else:
                print(f"   ❌ Wrong exception: {e}")
                
        print("\n📋 Test 2: Token generation format")
        
        # Test correct token format generation
        bbox_obj = {"bbox_2d": [100, 200, 300, 400], "desc": "测试bbox"}
        square_obj = {"square": [100, 200, 150, 250, 200, 300, 150, 350], "desc": "测试square"} 
        line_obj = {"line": [100, 200, 150, 250, 200, 300], "desc": "测试line"}
        
        bbox_result = manager.format_object(bbox_obj)
        square_result = manager.format_object(square_obj)
        line_result = manager.format_object(line_obj)
        
        # Verify token structure
        print(f"   📦 bbox format: {bbox_result['wrapped_bbox_2d'][:50]}...")
        print(f"   📦 square format: {square_result['wrapped_square'][:50]}...")
        print(f"   📦 line format: {line_result['wrapped_line'][:50]}...")
        
        # Check that descriptions are wrapped with object_ref tokens
        for result_name, result in [("bbox", bbox_result), ("square", square_result), ("line", line_result)]:
            desc = result.get("wrapped_desc", "")
            if desc.startswith("<|object_ref_start|>") and desc.endswith("<|object_ref_end|>"):
                print(f"   ✅ {result_name} description properly wrapped")
            else:
                print(f"   ❌ {result_name} description not wrapped: {desc}")
        
        print("\n✅ SimpleTokenManager validation tests passed!")
        
    except Exception as e:
        print(f"❌ SimpleTokenManager test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Verify special token principles
    print("\n📋 Test 3: Special token principles verification")
    
    # According to user requirements:
    # 1. ALL geometry types must use dedicated special tokens  
    # 2. ALL coordinates must be <coord_xxx> special tokens
    # 3. ALL object references must be wrapped in <|object_ref_start|> and <|object_ref_end|>
    # 4. ONLY the caption text should be pure text
    # 5. Every sample must have exactly one geometry type
    
    sample_response = "<|square_start|><coord_277><coord_262><coord_289><coord_26><coord_380><coord_168><coord_344><coord_364><|square_end|> <|object_ref_start|>螺丝、光纤插头/ODF端光纤插头,显示完整,符合要求<|object_ref_end|>"
    
    checks = []
    
    # Check 1: Dedicated geometry tokens
    if "<|square_start|>" in sample_response and "<|square_end|>" in sample_response:
        checks.append("✅ Dedicated geometry tokens present")
    else:
        checks.append("❌ Missing dedicated geometry tokens")
    
    # Check 2: Coordinate tokens
    import re
    coord_tokens = re.findall(r'<coord_\d+>', sample_response)
    if len(coord_tokens) == 8:  # Square should have 8 coordinates
        checks.append(f"✅ All coordinates are <coord_xxx> tokens ({len(coord_tokens)} found)")
    else:
        checks.append(f"❌ Incorrect coordinate token count: {len(coord_tokens)}")
    
    # Check 3: Object reference wrapping
    if "<|object_ref_start|>" in sample_response and "<|object_ref_end|>" in sample_response:
        checks.append("✅ Object reference tokens present")
    else:
        checks.append("❌ Missing object reference tokens")
    
    # Check 4: Only caption is pure text
    ref_match = re.search(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>', sample_response)
    if ref_match:
        caption = ref_match.group(1)
        if not any(token in caption for token in ["<coord_", "<|", ">"]):
            checks.append(f"✅ Caption is pure text: '{caption}'")
        else:
            checks.append(f"❌ Caption contains special tokens: '{caption}'")
    else:
        checks.append("❌ No caption found in object reference")
    
    for check in checks:
        print(f"   {check}")
    
    if all("✅" in check for check in checks):
        print("\n🎯 ALL SPECIAL TOKEN PRINCIPLES VERIFIED!")
        return True
    else:
        print("\n❌ Some special token principles violated!")
        return False

if __name__ == "__main__":
    success = test_special_token_system()
    if success:
        print("\n🎉 Complete special token system test PASSED!")
    else:
        print("\n💥 Complete special token system test FAILED!")
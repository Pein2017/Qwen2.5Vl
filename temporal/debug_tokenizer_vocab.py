#!/usr/bin/env python3
"""
Debug script to examine the tokenizer vocabulary and find coordinate tokens.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer


def debug_tokenizer_vocab():
    """Debug tokenizer vocabulary to find coordinate tokens."""
    print("🔍 Loading tokenizer and examining vocabulary...")
    
    # Load tokenizer
    model_path = "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    vocab = tokenizer.get_vocab()
    print(f"📊 Tokenizer vocabulary size: {len(vocab)}")
    
    # Find coordinate tokens
    coord_tokens = {}
    for token, token_id in vocab.items():
        if token.startswith("<coord_") and token.endswith(">"):
            try:
                coord_value = int(token[7:-1])  # Extract number from <coord_123>
                coord_tokens[coord_value] = token_id
            except ValueError:
                continue
    
    if coord_tokens:
        print(f"\n🎯 Found {len(coord_tokens)} coordinate tokens:")
        
        # Sort by coordinate value
        sorted_coords = sorted(coord_tokens.items())
        
        print(f"   First 10 coordinate tokens:")
        for i, (coord_val, token_id) in enumerate(sorted_coords[:10]):
            print(f"     <coord_{coord_val}> -> ID {token_id}")
        
        print(f"   Last 10 coordinate tokens:")
        for i, (coord_val, token_id) in enumerate(sorted_coords[-10:]):
            print(f"     <coord_{coord_val}> -> ID {token_id}")
        
        # Analyze the range
        coord_values = list(coord_tokens.keys())
        token_ids = list(coord_tokens.values())
        
        min_coord_val = min(coord_values)
        max_coord_val = max(coord_values)
        min_token_id = min(token_ids)
        max_token_id = max(token_ids)
        
        print(f"\n📊 Coordinate Token Analysis:")
        print(f"   Coordinate value range: [{min_coord_val}, {max_coord_val}]")
        print(f"   Token ID range: [{min_token_id}, {max_token_id}]")
        print(f"   Total coordinate tokens: {len(coord_tokens)}")
        
        # Check if the range is consecutive
        expected_token_ids = list(range(min_token_id, max_token_id + 1))
        actual_token_ids = sorted(token_ids)
        
        if expected_token_ids == actual_token_ids:
            print(f"   ✅ Token IDs are consecutive")
        else:
            print(f"   ❌ Token IDs are NOT consecutive")
            missing_ids = set(expected_token_ids) - set(actual_token_ids)
            if missing_ids:
                print(f"   Missing token IDs: {sorted(list(missing_ids))[:10]}...")
        
        # Check coordinate values
        expected_coord_vals = list(range(min_coord_val, max_coord_val + 1))
        actual_coord_vals = sorted(coord_values)
        
        if expected_coord_vals == actual_coord_vals:
            print(f"   ✅ Coordinate values are consecutive")
        else:
            print(f"   ❌ Coordinate values are NOT consecutive")
            missing_vals = set(expected_coord_vals) - set(actual_coord_vals)
            if missing_vals:
                print(f"   Missing coordinate values: {sorted(list(missing_vals))[:10]}...")
        
        print(f"\n🔧 Recommended coordinate manager settings:")
        print(f"   coord_start_id: {min_token_id}")
        print(f"   coord_end_id: {max_token_id + 1}")
        print(f"   max_coord_value: {max_coord_val + 1}")
        
        # Check if this matches the error log coordinate positions
        error_coord_positions = [
            2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 3631, 3632, 3633, 3634
        ]
        
        print(f"\n🧪 Checking against error log coordinate positions:")
        error_set = set(error_coord_positions)
        vocab_set = set(token_ids)
        
        matching_positions = error_set & vocab_set
        missing_in_vocab = error_set - vocab_set
        extra_in_vocab = vocab_set - error_set
        
        print(f"   Error log positions: {len(error_coord_positions)}")
        print(f"   Vocab coordinate token IDs: {len(token_ids)}")
        print(f"   Matching positions: {len(matching_positions)}")
        print(f"   Missing in vocab: {len(missing_in_vocab)}")
        print(f"   Extra in vocab: {len(extra_in_vocab)}")
        
        if len(matching_positions) > 0:
            print(f"   ✅ Some coordinate tokens match!")
            print(f"   Sample matching: {sorted(list(matching_positions))[:10]}")
        else:
            print(f"   ❌ No coordinate tokens match error log positions!")
        
        if len(missing_in_vocab) > 0:
            print(f"   Missing from vocab: {sorted(list(missing_in_vocab))[:10]}...")
        
    else:
        print(f"\n❌ No coordinate tokens found in vocabulary!")
        
        # Check for any tokens that might be coordinate-related
        coord_like_tokens = {}
        for token, token_id in vocab.items():
            if "coord" in token.lower():
                coord_like_tokens[token] = token_id
        
        if coord_like_tokens:
            print(f"   Found {len(coord_like_tokens)} coordinate-like tokens:")
            for token, token_id in list(coord_like_tokens.items())[:10]:
                print(f"     {token} -> ID {token_id}")
        else:
            print(f"   No coordinate-like tokens found either!")
    
    # Also check for geometry tokens
    print(f"\n🎯 Checking geometry tokens:")
    geometry_tokens = {
        "box_start": "<|box_start|>",
        "box_end": "<|box_end|>",
        "square_start": "<|square_start|>",
        "square_end": "<|square_end|>",
        "line_start": "<|line_start|>",
        "line_end": "<|line_end|>",
    }
    
    for name, token in geometry_tokens.items():
        token_id = vocab.get(token)
        if token_id is not None:
            print(f"   ✅ {name}: {token} -> ID {token_id}")
        else:
            print(f"   ❌ {name}: {token} -> NOT FOUND")
    
    return coord_tokens


if __name__ == "__main__":
    try:
        coord_tokens = debug_tokenizer_vocab()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

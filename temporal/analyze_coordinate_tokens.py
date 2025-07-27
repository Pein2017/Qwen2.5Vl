#!/usr/bin/env python3
"""
Analyze coordinate token positions from the error log to determine the correct range.
"""

def analyze_coordinate_positions():
    """Analyze the coordinate token positions from the error log."""
    
    # Coordinate token positions from the error log
    coord_positions = [
        2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 3631, 3632, 3633, 3634
    ]
    
    print("🔍 Analyzing coordinate token positions from error log...")
    print(f"   Total coordinate tokens found: {len(coord_positions)}")
    print(f"   Min coordinate token ID: {min(coord_positions)}")
    print(f"   Max coordinate token ID: {max(coord_positions)}")
    
    # Check for patterns
    print(f"\n📊 Pattern Analysis:")
    
    # Group consecutive ranges
    ranges = []
    current_start = coord_positions[0]
    current_end = coord_positions[0]
    
    for i in range(1, len(coord_positions)):
        if coord_positions[i] == current_end + 1:
            current_end = coord_positions[i]
        else:
            ranges.append((current_start, current_end))
            current_start = coord_positions[i]
            current_end = coord_positions[i]
    ranges.append((current_start, current_end))
    
    print(f"   Consecutive ranges found: {len(ranges)}")
    for i, (start, end) in enumerate(ranges):
        length = end - start + 1
        print(f"     Range {i+1}: [{start}, {end}] (length: {length})")
    
    # Check if this looks like coordinate tokens [0, 2047]
    print(f"\n🎯 Coordinate Token Analysis:")
    
    # The coordinate tokens should be in format <coord_0>, <coord_1>, ..., <coord_2047>
    # Let's see if we can map these IDs back to coordinate values
    
    # From the log, we know the expected coordinate range is [151669, 153717)
    # But actual tokens are in different ranges
    
    expected_start = 151669
    expected_end = 153717
    actual_min = min(coord_positions)
    actual_max = max(coord_positions)
    
    print(f"   Expected coordinate range: [{expected_start}, {expected_end})")
    print(f"   Actual coordinate range: [{actual_min}, {actual_max}]")
    print(f"   Range mismatch: {actual_min - expected_start} offset")
    
    # Check if the ranges make sense for 2048 coordinate tokens
    total_expected = expected_end - expected_start  # Should be 2048
    total_actual = len(coord_positions)
    
    print(f"   Expected total coordinate tokens: {total_expected}")
    print(f"   Actual coordinate tokens found: {total_actual}")
    
    # Determine the likely coordinate token start ID
    # If we assume the coordinate tokens are contiguous, we can find the base
    if len(ranges) > 0:
        # Find the largest consecutive range (likely the main coordinate token range)
        largest_range = max(ranges, key=lambda r: r[1] - r[0])
        range_start, range_end = largest_range
        range_length = range_end - range_start + 1
        
        print(f"\n🔧 Suggested Fix:")
        print(f"   Largest consecutive range: [{range_start}, {range_end}] (length: {range_length})")
        
        if range_length >= 1000:  # Likely the main coordinate token range
            print(f"   Suggested coord_start_id: {range_start}")
            print(f"   Suggested coord_end_id: {range_end + 1}")
        else:
            print(f"   No large consecutive range found. Data may be fragmented.")
            print(f"   Consider using min/max approach:")
            print(f"   Suggested coord_start_id: {actual_min}")
            print(f"   Suggested coord_end_id: {actual_max + 1}")
    
    return coord_positions


if __name__ == "__main__":
    analyze_coordinate_positions()

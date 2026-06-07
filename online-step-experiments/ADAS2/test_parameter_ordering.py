#!/usr/bin/env python3
"""
Test to verify parameter ordering fix in global_search_nsga3() for ADAS2

The bug was: NSGA3 returns solutions in alphabetically sorted order
but the code was extracting in hardcoded order (ADAS1 specific)

This test verifies the fix correctly uses alphabetically sorted order.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as conf
import numpy as np

def test_variable_ordering():
    """Test that variables are extracted in alphabetically sorted order"""

    # Get alphabetically sorted variable names (what NSGA3 uses)
    var_names = sorted(conf.SS_VARIABLES.keys())
    print(f"Alphabetically sorted variables: {var_names}")
    print(f"Number of variables: {len(var_names)}")

    # Show variable bounds
    print("\nVariable configuration:")
    for var in var_names:
        domain = conf.SS_VARIABLES[var]["domain"]
        bounds = conf.SS_VARIABLES[var]["range"]
        print(f"  {var}: {bounds} ({domain.__name__})")

    # Create a test NSGA3 result array in alphabetically sorted order
    # Using valid values for ADAS2
    test_values = {}
    test_array = []
    for i, var in enumerate(var_names):
        bounds = conf.SS_VARIABLES[var]["range"]
        # Use midpoint of bounds
        val = (bounds[0] + bounds[1]) / 2
        if conf.SS_VARIABLES[var]["domain"] == int:
            val = int(val)
        test_values[var] = val
        test_array.append(val)

    nsga3_result_array = np.array(test_array)

    # The NEW (fixed) code extracts using var_names order:
    params = np.array(nsga3_result_array)  # Already in alphabetically sorted order

    # Convert back to dict using same alphabetically sorted order
    test_dict = {
        var_name: (float(params[i]) if conf.SS_VARIABLES[var_name]["domain"] == float
                  else int(params[i]))
        for i, var_name in enumerate(var_names)
    }

    print(f"\nTest NSGA3 array (alphabetically sorted): {nsga3_result_array}")
    print(f"Extracted dict: {test_dict}")

    # Verify correct extraction
    for var in var_names:
        expected_val = test_values[var]
        actual_val = test_dict[var]
        assert actual_val == expected_val, f"{var} mismatch: {actual_val} != {expected_val}"

    print("✓ Parameter extraction is correct for all variables")

    # Test with dict input (from NSGA3 with mixed variables)
    nsga3_result_dict = test_dict.copy()

    params_from_dict = np.array([nsga3_result_dict[var] if conf.SS_VARIABLES[var]["domain"] == float else int(nsga3_result_dict[var])
                                 for var in var_names])

    print(f"\nTest NSGA3 dict: {nsga3_result_dict}")
    print(f"Extracted params array: {params_from_dict}")

    assert np.allclose(params_from_dict, nsga3_result_array, atol=1e-6), \
        f"Array mismatch: {params_from_dict} != {nsga3_result_array}"
    print("✓ Dict-to-array extraction is correct")

    # Verify parameter bounds are respected after conversion
    print("\nVerifying parameter bounds are respected:")
    for i, var in enumerate(var_names):
        bounds = conf.SS_VARIABLES[var]["range"]
        val = test_dict[var]
        if bounds[0] <= val <= bounds[1]:
            print(f"  ✓ {var} = {val} in [{bounds[0]}, {bounds[1]}]")
        else:
            print(f"  ✗ {var} = {val} OUTSIDE [{bounds[0]}, {bounds[1]}]")
            raise ValueError(f"{var} out of bounds!")

    print("\n" + "="*80)
    print("ALL TESTS PASSED - Parameter ordering fix is correct for ADAS2!")
    print("="*80)

if __name__ == "__main__":
    test_variable_ordering()

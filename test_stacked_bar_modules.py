#!/usr/bin/env python
"""
Test script to verify the stacked bar modules work correctly without matplotlib.
"""

# Test importing the common module
print("Testing tool_stacked_bar_common module...")
try:
    from tool_stacked_bar_common import (
        translate_modes,
        noise_modes,
        error_categories,
        category_colors,
        pascal_to_readable,
        load_model_statistics,
        load_multi_model_statistics,
    )
    print("✓ Successfully imported all functions from tool_stacked_bar_common")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    exit(1)

# Test pascal_to_readable function
print("\nTesting pascal_to_readable function...")
test_cases = [
    ("SyntaxError", "syntax error"),
    ("LanguageMismatchWrongValue", "language mismatch wrong value"),
    ("ExactlySameMeaning", "exactly same meaning"),
]
for input_str, expected in test_cases:
    result = pascal_to_readable(input_str)
    if result == expected:
        print(f"✓ pascal_to_readable('{input_str}') = '{result}'")
    else:
        print(f"✗ pascal_to_readable('{input_str}') = '{result}', expected '{expected}'")

# Test loading model statistics
print("\nTesting load_model_statistics function...")
try:
    data = load_model_statistics("gpt-5", "tool/result", "Hindi")
    print(f"✓ Successfully loaded data for gpt-5, Hindi")

    # Check structure
    if isinstance(data, dict):
        print(f"✓ Data is a dictionary")

        # Check for translate modes
        for tm in translate_modes:
            if tm in data:
                print(f"✓ Found translate mode: {tm}")
                for nm in noise_modes:
                    if nm in data[tm]:
                        total = sum(data[tm][nm].values())
                        print(f"  ✓ {tm} + {nm}: {total} total errors")
                    else:
                        print(f"  ✗ Missing noise mode: {nm}")
            else:
                print(f"✗ Missing translate mode: {tm}")
    else:
        print(f"✗ Data is not a dictionary: {type(data)}")

except Exception as e:
    print(f"✗ Failed to load data: {e}")
    import traceback
    traceback.print_exc()

# Test loading multi-model statistics
print("\nTesting load_multi_model_statistics function...")
try:
    model_list = ["gpt-5", "gpt-5-mini"]
    data = load_multi_model_statistics(model_list, "tool/result", "Hindi", "FT")
    print(f"✓ Successfully loaded data for {model_list}, Hindi, FT")

    # Check structure
    if isinstance(data, dict):
        print(f"✓ Data is a dictionary")

        # Check for models
        for model_name in model_list:
            if model_name in data:
                print(f"✓ Found model: {model_name}")
                for nm in noise_modes:
                    if nm in data[model_name]:
                        total = sum(data[model_name][nm].values())
                        print(f"  ✓ {model_name} + {nm}: {total} total errors")
                    else:
                        print(f"  ✗ Missing noise mode: {nm}")
            else:
                print(f"✗ Missing model: {model_name}")
    else:
        print(f"✗ Data is not a dictionary: {type(data)}")

except Exception as e:
    print(f"✗ Failed to load multi-model data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All tests completed!")
print("="*60)

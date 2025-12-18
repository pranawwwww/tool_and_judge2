from codebase_rs import *

# Test 1: Single item in list (from test_cmp.py)
result1 = LocalModel.Llama3_3_70B in [LocalModel.Llama3_3_70B]
print(f"Test 1 - Single item in list: {result1}")

# Test 2: Multiple items in list (like in judge.py line 106)
result2 = LocalModel.Qwen3_8B in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b]
print(f"Test 2 - Qwen3_8B in list of Qwen models: {result2}")

# Test 3: Direct equality
result3 = LocalModel.Qwen3_8B == LocalModel.Qwen3_8B
print(f"Test 3 - Direct equality (Qwen3_8B == Qwen3_8B): {result3}")

# Test 4: Different models
result4 = LocalModel.Qwen3_8B == LocalModel.Llama3_3_70B
print(f"Test 4 - Different models (Qwen3_8B == Llama3_3_70B): {result4}")

# Test 5: Identity check
a = LocalModel.Qwen3_8B
b = LocalModel.Qwen3_8B
print(f"Test 5 - Identity (a is b): {a is b}")
print(f"Test 5 - Equality (a == b): {a == b}")

# Test 6: Type and repr
print(f"\nType of LocalModel.Qwen3_8B: {type(LocalModel.Qwen3_8B)}")
print(f"Repr of LocalModel.Qwen3_8B: {repr(LocalModel.Qwen3_8B)}")
print(f"Str of LocalModel.Qwen3_8B: {str(LocalModel.Qwen3_8B)}")

from codebase_rs import *

# Simulate what happens in judge_config_slurm2.py
config = JudgeConfig(LocalModel.Qwen3_8B, JudgeExperiment.PreferenceDirect(lang1="en", lang2="zh_cn"))

print(f"Config model: {config.model}")
print(f"Config model type: {type(config.model)}")
print(f"Config model repr: {repr(config.model)}")

# Test the exact conditions from judge.py lines 104-107
print("\n--- Testing conditions from judge.py ---")

# Line 104
test1 = config.model == LocalModel.Llama3_3_70B
print(f"Line 104: config.model == LocalModel.Llama3_3_70B: {test1}")

# Line 106
test2 = config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b]
print(f"Line 106: config.model in [Qwen3_8B, Qwen3_14B, Qwen3_30bA3b, Qwen3Next80bA3b]: {test2}")

# Additional identity checks
print("\n--- Identity checks ---")
print(f"config.model is LocalModel.Qwen3_8B: {config.model is LocalModel.Qwen3_8B}")
print(f"id(config.model): {id(config.model)}")
print(f"id(LocalModel.Qwen3_8B): {id(LocalModel.Qwen3_8B)}")

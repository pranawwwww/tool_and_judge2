from codebase_rs import *

config = JudgeConfig(LocalModel.Qwen3_8B, JudgeExperiment.PreferenceDirect(lang1="en", lang2="zh_cn"))

# Direct equality test
a = config.model
b = LocalModel.Qwen3_8B

print(f"a = config.model: {a}")
print(f"b = LocalModel.Qwen3_8B: {b}")
print(f"a == b: {a == b}")
print(f"a is b: {a is b}")
print(f"id(a): {id(a)}, id(b): {id(b)}")

# Test if __richcmp__ is implemented
try:
    result = a.__eq__(b)
    print(f"a.__eq__(b): {result}")
except AttributeError as e:
    print(f"No __eq__ method: {e}")

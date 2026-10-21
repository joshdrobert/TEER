#!/usr/bin/env python3

import subprocess
import sys

# Run pytest on the transform repeatability test file
result = subprocess.run([
    sys.executable, "-m", "pytest", 
    "tests/test_transform_repeatability.py", 
    "-v"
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)
print("Return code:", result.returncode)
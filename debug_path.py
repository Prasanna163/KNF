import sys
import os
import inspect

# Ensure current directory is in path
sys.path.insert(0, os.getcwd())

import knf_core
from knf_core import xtb

print(f"knf_core path: {knf_core.__file__}")
print(f"xtb module path: {xtb.__file__}")

source = inspect.getsource(xtb.run_xtb_single_point)
print("\nSource of run_xtb_single_point:\n")
print(source)

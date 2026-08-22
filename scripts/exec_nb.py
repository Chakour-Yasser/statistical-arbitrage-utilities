import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
p = Path(sys.argv[1])
nb = nbformat.read(p, as_version=4)
try:
    ExecutePreprocessor(timeout=2400, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(Path.cwd())}})
    print("EXECUTION OK")
except Exception as e:
    print("FAILED:", type(e).__name__); import re; msg=str(e); print(msg[-2500:])
nbformat.write(nb, p)

#!/usr/bin/env python3
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
    print("Matplotlib import successful!")
except ImportError as e:
    print(f"Matplotlib import failed: {e}")
    print("Trying to install matplotlib...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib"])
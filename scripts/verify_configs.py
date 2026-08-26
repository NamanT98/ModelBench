import sys
from pathlib import Path
from modelbench.config import load_config

configs = list(Path("configs").glob("*.yaml")) + list(Path("configs/experimental").glob("*.yaml"))
success = True
for c in configs:
    try:
        load_config(str(c))
        print(f"OK: {c}")
    except Exception as e:
        print(f"FAIL: {c} - {e}")
        success = False

sys.exit(0 if success else 1)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.seed import run_seed  # noqa: E402

if __name__ == "__main__":
    print(run_seed(reset="--reset" in sys.argv))

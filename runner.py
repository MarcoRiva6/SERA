import subprocess
import sys
import time
from pathlib import Path

from experimenter import get_expr_filenames_from_args

SCRIPT = Path(__file__).parent / "experimenter.py"  # the file to run
RESTART_DELAY = 30  # seconds (optional)

expr = get_expr_filenames_from_args()
expr = expr[:-5]

def run_forever():
    while True:
        try:
            print(f"[runner] Starting {SCRIPT}")
            process = subprocess.run(
                [sys.executable, str(SCRIPT), expr],
                check=False,
            )
            print(f"[runner] {SCRIPT} exited with code {process.returncode}")

            time.sleep(RESTART_DELAY)

        except KeyboardInterrupt:
            print("\n[runner] Graceful shutdown requested. Exiting.")
            break

if __name__ == "__main__":
    run_forever()
import os
import subprocess
import sys

print("Resuming Stage 3 training runs...", flush=True)
ret = subprocess.run([sys.executable, "-u", "run_stage3.py"])
if ret.returncode != 0:
    print(f"Error in run_stage3.py: exit code {ret.returncode}", flush=True)
    sys.exit(ret.returncode)

print("\nRunning stage3_stats.py...", flush=True)
ret = subprocess.run([sys.executable, "-u", "stage3_stats.py"])
if ret.returncode != 0:
    print(f"Error in stage3_stats.py: exit code {ret.returncode}", flush=True)
    sys.exit(ret.returncode)

print("\n>>> STAGE 3 IS FULLY COMPLETE! <<<", flush=True)

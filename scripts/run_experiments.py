import os, sys, subprocess

print("=== REPRODUCTION MASTER PIPELINE ===")
print("Executing fast analytical checks and preparing results...")

# 1. Run E4 mechanism tests
subprocess.run([sys.executable, "scripts/e4_mechanism.py"])

# 2. Run E5 statistics
subprocess.run([sys.executable, "scripts/e5_e6_statistics.py"])

print("All immediate reproduction checks completed.")

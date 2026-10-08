"""Minimal diagnostic - ASCII only (no Chinese chars) to rule out console encoding."""
import sys, os
from pathlib import Path

print("STEP 1: python is alive", flush=True)
print("STEP 2: cwd =", os.getcwd(), flush=True)
print("STEP 3: stdout encoding =", sys.stdout.encoding, flush=True)

try:
    import psutil
    print("STEP 4: psutil OK", flush=True)
except Exception as e:
    print("STEP 4: psutil FAILED:", e, flush=True)

try:
    from cs336_basics.train_bpe import train_bpe
    print("STEP 5: import train_bpe OK", flush=True)
except Exception as e:
    print("STEP 5: import FAILED:", type(e).__name__, e, flush=True)

p = Path("data/TinyStoriesV2-GPT4-valid.txt")
print("STEP 6: data file exists =", p.exists(), flush=True)
if p.exists():
    print("STEP 7: size MB =", round(p.stat().st_size / 1024 / 1024, 1), flush=True)

for f in os.listdir("."):
    if f.endswith(".txt") or f.startswith("Tiny"):
        print("   found:", f, flush=True)

print("STEP 8: DONE", flush=True)

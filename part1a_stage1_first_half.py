"""
PART 1a / 5 -- Run this first.

Stage 1 (main sweep), FIRST HALF of scenarios only: default, K2, K10,
N100, N3000. Same algorithm/logic as the original script -- only the
orchestration is split so each run stays under ~3 hours.

Appends to: results/raw/main_sweep_raw.csv
"""
import os
import time
import random
import numpy as np

from fedpc_core import MASTER_SEED, run_main_sweep_partial

if __name__ == "__main__":
    np.random.seed(MASTER_SEED)
    random.seed(MASTER_SEED)

    REPS_MAIN = 15
    OUT_DIR = "results"
    for sub in ["raw", "figures", "tables", "statistics"]:
        os.makedirs(f"{OUT_DIR}/{sub}", exist_ok=True)

    t0 = time.time()
    print("=" * 70)
    print("STAGE 1a/5: Main sweep, scenarios 1-5 of 10")
    print("=" * 70)
    run_main_sweep_partial(REPS_MAIN, OUT_DIR, row_start=0, row_end=5)
    print(f"\nPART 1a DONE in {(time.time()-t0)/60:.1f} minutes.")
    print("Next: run part1b_stage1_second_half.py")

"""
PART 1b / 5 -- Run this after part1a_stage1_first_half.py.

Stage 1 (main sweep), SECOND HALF of scenarios: phi0.1, noise_none,
noise_strong, tau0.3, ell1. Appends to the same CSV part1a wrote to,
so after this finishes, results/raw/main_sweep_raw.csv contains all
10 scenarios -- identical to running Stage 1 in one shot.
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

    csv_path = f"{OUT_DIR}/raw/main_sweep_raw.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected {csv_path} from part1a. Run part1a_stage1_first_half.py first.")

    t0 = time.time()
    print("=" * 70)
    print("STAGE 1b/5: Main sweep, scenarios 6-10 of 10")
    print("=" * 70)
    run_main_sweep_partial(REPS_MAIN, OUT_DIR, row_start=5, row_end=10)
    print(f"\nPART 1b DONE in {(time.time()-t0)/60:.1f} minutes.")
    print("Stage 1 fully complete -> results/raw/main_sweep_raw.csv (10/10 scenarios).")
    print("Next: run part2a_stage2_first_half.py")

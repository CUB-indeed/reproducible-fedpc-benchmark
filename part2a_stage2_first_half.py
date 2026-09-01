"""
PART 2a / 5 -- Run this after part1b_stage1_second_half.py.

Stage 2 (mechanism-shift sweep), FIRST 3 of 5 scenarios: delprob 0.0,
0.03, 0.07. Each scenario is ~35-40 min at 15 reps, so 3 of them stays
safely under 3 hours. Appends to results/raw/mechshift_sweep_raw.csv.
"""
import os
import time
import random
import numpy as np

from fedpc_core import MASTER_SEED, run_mechshift_sweep_partial

if __name__ == "__main__":
    np.random.seed(MASTER_SEED)
    random.seed(MASTER_SEED)

    REPS_MECHSHIFT = 15
    OUT_DIR = "results"
    for sub in ["raw", "figures", "tables", "statistics"]:
        os.makedirs(f"{OUT_DIR}/{sub}", exist_ok=True)

    t0 = time.time()
    print("=" * 70)
    print("STAGE 2a/5: Mechanism-shift sweep, scenarios 1-3 of 5")
    print("=" * 70)
    run_mechshift_sweep_partial(REPS_MECHSHIFT, OUT_DIR, row_start=0, row_end=3)
    print(f"\nPART 2a DONE in {(time.time()-t0)/60:.1f} minutes.")
    print("Next: run part2b_stage2_second_half.py")

"""
PART 2b / 5 -- Run this after part2a_stage2_first_half.py.

Stage 2 (mechanism-shift sweep), LAST 2 of 5 scenarios: delprob 0.15,
0.25. Appends to the same CSV part2a wrote to, so after this finishes,
results/raw/mechshift_sweep_raw.csv contains all 5 scenarios --
identical to running Stage 2 in one shot.
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

    csv_path = f"{OUT_DIR}/raw/mechshift_sweep_raw.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected {csv_path} from part2a. Run part2a_stage2_first_half.py first.")

    t0 = time.time()
    print("=" * 70)
    print("STAGE 2b/5: Mechanism-shift sweep, scenarios 4-5 of 5")
    print("=" * 70)
    run_mechshift_sweep_partial(REPS_MECHSHIFT, OUT_DIR, row_start=3, row_end=5)
    print(f"\nPART 2b DONE in {(time.time()-t0)/60:.1f} minutes.")
    print("Stage 2 fully complete -> results/raw/mechshift_sweep_raw.csv (5/5 scenarios).")
    print("Next: run part3_stage3_6.py")

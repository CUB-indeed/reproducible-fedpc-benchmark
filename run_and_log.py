import sys
import os
import time
import runpy

# Configuration
TARGET_SCRIPT = "/home/coder/project/FedPC-style.py"
LOG_DIR = "/home/coder/project/results"

os.makedirs(LOG_DIR, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"run_log_{timestamp}.txt")


class Tee:
    """Write output to both the terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    if not os.path.exists(TARGET_SCRIPT):
        print(
            f"[ERROR] Target script '{TARGET_SCRIPT}' not found. "
            "Please check the path or update TARGET_SCRIPT."
        )
        sys.exit(1)

    with open(log_path, "w", encoding="utf-8") as f:
        tee_out = Tee(sys.stdout, f)
        tee_err = Tee(sys.stderr, f)

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_out, tee_err

        start = time.time()
        print(f"=== Running {TARGET_SCRIPT} ===")
        print(f"Logging output to: {log_path}\n")

        try:
            runpy.run_path(TARGET_SCRIPT, run_name="__main__")
        except SystemExit:
            pass
        except Exception as e:
            print(f"\n[ERROR] Script terminated with an exception: {e}")
            raise
        finally:
            elapsed = time.time() - start
            print(f"\n=== Finished in {elapsed:.1f} seconds ===")
            sys.stdout, sys.stderr = old_stdout, old_stderr

    print(f"\nAll output has been saved to: {log_path}")


if __name__ == "__main__":
    main()
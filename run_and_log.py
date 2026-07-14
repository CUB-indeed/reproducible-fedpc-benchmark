"""
Wrapper runner: menjalankan fedpc_experiment.py TANPA mengubah file aslinya,
sambil menyimpan seluruh output (print, warning, error) ke file .txt.

Cara pakai:
    1. Pastikan script utama kamu bernama fedpc_experiment.py
       (atau ganti nama TARGET_SCRIPT di bawah sesuai nama file kamu).
    2. Jalankan file ini, bukan file aslinya:
           python run_and_log.py
    3. Output akan tetap muncul di terminal SEKALIGUS tersimpan
       otomatis ke results/run_log_<timestamp>.txt
"""

import sys
import os
import time
import runpy

# ---- KONFIGURASI ----
TARGET_SCRIPT = "/home/coder/project/FedPC-style.py"   # path file script asli kamu
LOG_DIR = "/home/coder/project/results"                 # folder log disimpan di sini
# ----------------------

os.makedirs(LOG_DIR, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"run_log_{timestamp}.txt")


class Tee:
    """Menulis output ke terminal DAN ke file secara bersamaan."""
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
        print(f"[ERROR] Tidak menemukan '{TARGET_SCRIPT}'. "
              f"Pastikan nama file sesuai atau ubah TARGET_SCRIPT di run_and_log.py")
        sys.exit(1)

    with open(log_path, "w", encoding="utf-8") as f:
        tee_out = Tee(sys.stdout, f)
        tee_err = Tee(sys.stderr, f)

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_out, tee_err

        start = time.time()
        print(f"=== Mulai menjalankan {TARGET_SCRIPT} ===")
        print(f"Log disimpan otomatis ke: {log_path}\n")

        try:
            # Menjalankan script asli persis seperti `python fedpc_experiment.py`
            runpy.run_path(TARGET_SCRIPT, run_name="__main__")
        except SystemExit:
            pass
        except Exception as e:
            print(f"\n[ERROR] Script berhenti dengan error: {e}")
            raise
        finally:
            elapsed = time.time() - start
            print(f"\n=== Selesai dalam {elapsed:.1f} detik ===")
            sys.stdout, sys.stderr = old_stdout, old_stderr

    print(f"\nSemua output sudah tersimpan di: {log_path}")


if __name__ == "__main__":
    main()

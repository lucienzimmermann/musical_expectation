"""
TRIGGER TIMING TEST  —  PTB clock edition
==========================================
Sends a trigger byte every 3 s via serial port, timed with the PTB
high-resolution clock and a 50 ms spin-wait budget.
Jitter should be < 0.5 ms.

Requirements
------------
    pip install psychtoolbox pyserial numpy psutil
"""

import sys
import csv
import time

import numpy as np
from psychopy import core, prefs

prefs.hardware['audioLib'] = ['PTB']
import psychtoolbox as ptb

try:
    import serial
except ImportError:
    raise SystemExit("pyserial not installed — run: pip install pyserial")

# ── raise process priority ─────────────────────────────────────────────────────
try:
    import psutil, os
    p = psutil.Process(os.getpid())
    if sys.platform == "win32":
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("Process priority → HIGH")
    else:
        p.nice(-10)
        print("Process nice → -10")
except Exception as e:
    print(f"Could not raise priority ({e}) — continuing.")

# ==============================
# PARAMETERS
# ==============================
TRIGGER_PORT    = "COM3"
BAUD_RATE       = 115200
TRIGGER_CODE    = 99
RESET_CODE      = 0
PULSE_MS        = 10            # trigger pulse width
INTERVAL        = 3.0           # seconds between trigger onsets
N_REPS          = 100
OUTPUT_CSV      = "trigger_timing_log.csv"
SPIN_MARGIN     = 0.050         # switch from sleep → spin 50 ms before target
                                # (matches the SCHEDULED_LEAD in the audio script
                                #  so both have the same timing architecture)

# ==============================
# OPEN SERIAL PORT
# ==============================
print(f"\nOpening {TRIGGER_PORT} @ {BAUD_RATE} baud …")
port = serial.Serial(port=TRIGGER_PORT, baudrate=BAUD_RATE,
                     timeout=0, write_timeout=0)
port.write(bytes([RESET_CODE]))
time.sleep(0.05)
print("Serial port open.\n")

# ==============================
# RUN LOOP
# ==============================
rows = []
pulse_s = PULSE_MS / 1000.0

print(f"Sending {N_REPS} triggers every {INTERVAL} s.  Ctrl+C to abort.\n")

t0 = ptb.GetSecs() + 0.5   # small lead-in

try:
    for rep in range(N_REPS):
        target_t = t0 + rep * INTERVAL

        # --- coarse sleep to SPIN_MARGIN before target ---
        sleep_until = target_t - SPIN_MARGIN
        now = ptb.GetSecs()
        if sleep_until > now:
            time.sleep(sleep_until - now)

        # --- spin-wait for exact target ---
        while ptb.GetSecs() < target_t:
            pass

        # ---- SEND TRIGGER ----
        t_send = ptb.GetSecs()
        port.write(bytes([TRIGGER_CODE]))
        t_after_write = ptb.GetSecs()

        # hold pulse: spin-wait for exact pulse duration (no OS sleep)
        pulse_target = t_send + pulse_s
        while ptb.GetSecs() < pulse_target:
            pass

        port.write(bytes([RESET_CODE]))
        t_reset = ptb.GetSecs()

        delta_ms       = (t_send - target_t) * 1000
        write_call_ms  = (t_after_write - t_send) * 1000
        pulse_actual_ms = (t_reset - t_send) * 1000

        rows.append({
            "rep":             rep + 1,
            "target_t":        f"{target_t:.6f}",
            "actual_t":        f"{t_send:.6f}",
            "delta_ms":        f"{delta_ms:.3f}",
            "write_call_ms":   f"{write_call_ms:.3f}",
            "pulse_actual_ms": f"{pulse_actual_ms:.3f}",
        })
        print(f"  Rep {rep+1:3d} | target={target_t:.4f} s"
              f" | actual={t_send:.4f} s"
              f" | Δ={delta_ms:+.3f} ms"
              f" | write={write_call_ms:.3f} ms"
              f" | pulse={pulse_actual_ms:.3f} ms")

except KeyboardInterrupt:
    print("\nAborted by user.")

# ==============================
# CLEANUP & SAVE
# ==============================
port.write(bytes([RESET_CODE]))
port.close()
print("\nSerial port closed.")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["rep", "target_t", "actual_t", "delta_ms",
                    "write_call_ms", "pulse_actual_ms"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Log saved → {OUTPUT_CSV}")

if rows:
    deltas = np.array([float(r["delta_ms"]) for r in rows])
    print(f"\nTrigger timing (spin-exit Δ from target):")
    print(f"  mean={deltas.mean():.3f} ms  std={deltas.std():.3f} ms"
          f"  min={deltas.min():.3f} ms  max={deltas.max():.3f} ms")
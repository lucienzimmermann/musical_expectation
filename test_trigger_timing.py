# ==============================
# TRIGGER TIMING TEST
# Sends a trigger byte every 3s exactly via serial port
# Logs perf_counter timestamps to trigger_timing_log.csv
# ==============================

import csv
import time

TRIGGER_PORT  = "COM3"      # change to your port
BAUD_RATE     = 115200
TRIGGER_CODE  = 10          # byte value to send
RESET_CODE    = 0           # byte to reset the line
PULSE_MS      = 10          # how long to hold the trigger high (ms)
INTERVAL      = 3.0         # seconds between trigger onsets
N_REPS        = 20          # number of triggers to send
OUTPUT_CSV    = "trigger_timing_log.csv"

# ==============================
# OPEN SERIAL PORT
# ==============================

try:
    import serial
except ImportError:
    raise SystemExit("pyserial is not installed — run: pip install pyserial")

print(f"Opening {TRIGGER_PORT} @ {BAUD_RATE} baud...")
port = serial.Serial(port=TRIGGER_PORT, baudrate=BAUD_RATE, timeout=0)
# Reset line
port.write(bytes([RESET_CODE]))
time.sleep(0.05)
print("Serial port open.\n")

# ==============================
# RUN LOOP
# ==============================

rows = []
pulse_s = PULSE_MS / 1000.0

print(f"Sending {N_REPS} triggers, every {INTERVAL}s.  Press Ctrl+C to abort.\n")

t0 = time.perf_counter() + 0.5   # small lead-in

for rep in range(N_REPS):
    target_t = t0 + rep * INTERVAL

    # Coarse sleep to within ~5ms of target
    sleep_until = target_t - 0.005
    now = time.perf_counter()
    if sleep_until > now:
        time.sleep(sleep_until - now)

    # Spin-wait for remaining ~5ms
    while time.perf_counter() < target_t:
        pass

    # ---- SEND TRIGGER ----
    t_send = time.perf_counter()
    port.write(bytes([TRIGGER_CODE]))
    t_after_write = time.perf_counter()

    # Coarse sleep for pulse duration, then spin-wait
    pulse_target = t_send + pulse_s
    time.sleep(max(0, pulse_s - 0.001))
    while time.perf_counter() < pulse_target:
        pass

    port.write(bytes([RESET_CODE]))
    t_reset = time.perf_counter()

    delta_from_target = (t_send - target_t) * 1000   # ms
    write_call_ms     = (t_after_write - t_send) * 1000
    pulse_actual_ms   = (t_reset - t_send) * 1000

    rows.append({
        "rep":            rep + 1,
        "target_t":       f"{target_t:.6f}",
        "actual_t":       f"{t_send:.6f}",
        "delta_ms":       f"{delta_from_target:.3f}",
        "write_call_ms":  f"{write_call_ms:.3f}",
        "pulse_actual_ms":f"{pulse_actual_ms:.3f}",
    })

    print(f"  Rep {rep+1:3d} | target={target_t:.4f}s | actual={t_send:.4f}s "
          f"| Δ={delta_from_target:+.3f}ms | write={write_call_ms:.3f}ms | pulse={pulse_actual_ms:.3f}ms")

# ==============================
# CLEANUP & SAVE CSV
# ==============================

port.write(bytes([RESET_CODE]))
port.close()
print("\nSerial port closed.")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["rep","target_t","actual_t","delta_ms","write_call_ms","pulse_actual_ms"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Log saved to {OUTPUT_CSV}")

deltas = [float(r["delta_ms"]) for r in rows]
print(f"Δ mean={sum(deltas)/len(deltas):.3f}ms  "
      f"max={max(deltas):.3f}ms  min={min(deltas):.3f}ms")

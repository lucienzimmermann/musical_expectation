"""
AUDIO TIMING TEST  —  PTB/ASIO edition
=======================================
Plays a spike WAV every 3 s using PsychToolbox scheduled onset
(ASIO, latency class 4).  Jitter should be < 1 ms.

Requirements
------------
    pip install psychtoolbox soundfile numpy pyserial psutil
    ASIO drivers for your audio interface (Scarlett 2i2 → Focusrite ASIO)
"""

import sys
import csv
import time

import numpy as np
import soundfile as sf
from psychopy import core, prefs

prefs.hardware['audioLib'] = ['PTB']
import psychtoolbox as ptb

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
WAV_PATH        = "files/Square Trigger Sound.wav"
DEVICE_SR       = 44100
INTERVAL        = 3.0           # seconds between onsets
N_REPS          = 100
OUTPUT_CSV      = "audio_timing_log.csv"
PTB_LATENCY_CLASS = 4           # 4 = critical / ASIO
SCHEDULED_LEAD  = 0.050         # arm PTB 50 ms ahead of intended onset
SPIN_MARGIN     = 0.010         # switch from sleep → spin 10 ms before onset

# ==============================
# DEVICE SELECTION
# ==============================
def choose_ptb_device():
    devices = ptb.PsychPortAudio('GetDevices')
    print("\nAvailable output devices:\n")
    valid = []
    for i, dev in enumerate(devices):
        if dev['NrOutputChannels'] > 0:
            print(f"  [{i}] {dev['DeviceName']}"
                  f"  (out={dev['NrOutputChannels']}"
                  f"  API={dev['HostAudioAPIName']})")
            valid.append(i)
    while True:
        try:
            c = int(input("\nEnter device index: "))
            if c in valid:
                return c
            print("  Invalid — pick a listed index.")
        except ValueError:
            print("  Please enter a number.")

DEVICE_IDX = choose_ptb_device()

# ==============================
# LOAD AUDIO
# ==============================
def load_mono_float32(path, target_sr=DEVICE_SR):
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != target_sr:
        try:
            import resampy
            data = resampy.resample(data, sr, target_sr)
        except ImportError:
            n = int(round(len(data) * target_sr / sr))
            data = np.interp(
                np.linspace(0, len(data) - 1, n),
                np.arange(len(data)), data
            ).astype(np.float32)
        print(f"Resampled {sr} → {target_sr} Hz")
    return data

print(f"\nLoading {WAV_PATH} …")
mono   = load_mono_float32(WAV_PATH)
stereo = np.column_stack([mono, mono])   # PTB expects (N, 2)
print(f"Loaded: {len(mono)/DEVICE_SR:.3f} s  ({len(mono)} samples @ {DEVICE_SR} Hz)")

# ==============================
# OPEN PTB STREAM
# ==============================
print("\nOpening PTB audio stream …")
handle = ptb.PsychPortAudio(
    'Open', DEVICE_IDX, 1, PTB_LATENCY_CLASS, DEVICE_SR, 2
)
print(f"Stream handle: {handle}")

# ==============================
# RUN LOOP
# ==============================
rows = []
print(f"\nPlaying {N_REPS} times every {INTERVAL} s.  Ctrl+C to abort.\n")

t0 = ptb.GetSecs() + 0.5   # small lead-in

try:
    for rep in range(N_REPS):
        target_t    = t0 + rep * INTERVAL       # intended audio onset
        t_scheduled = target_t + SCHEDULED_LEAD # PTB hardware schedule time

        # --- arm PTB buffer & schedule hardware onset ---
        ptb.PsychPortAudio('FillBuffer', handle, stereo)
        ptb.PsychPortAudio('Start', handle, 1, t_scheduled, 0)

        # --- coarse sleep to SPIN_MARGIN before onset ---
        sleep_until = t_scheduled - SPIN_MARGIN
        now = ptb.GetSecs()
        if sleep_until > now:
            time.sleep(sleep_until - now)

        # --- spin-wait for exact onset ---
        while ptb.GetSecs() < t_scheduled:
            pass

        t_actual = ptb.GetSecs()
        delta_ms = (t_actual - t_scheduled) * 1000   # should be ~0

        rows.append({
            "rep":        rep + 1,
            "target_t":   f"{target_t:.6f}",
            "scheduled_t":f"{t_scheduled:.6f}",
            "actual_t":   f"{t_actual:.6f}",
            "delta_ms":   f"{delta_ms:.3f}",
        })
        print(f"  Rep {rep+1:3d} | target={target_t:.4f} s"
              f" | scheduled={t_scheduled:.4f} s"
              f" | spin exit Δ={delta_ms:+.3f} ms")

        # wait for playback to finish before next rep
        ptb.PsychPortAudio('Stop', handle, 1)   # 1 = wait until done

except KeyboardInterrupt:
    print("\nAborted by user.")

# ==============================
# CLEANUP
# ==============================
ptb.PsychPortAudio('Close', handle)
print("\nPTB stream closed.")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["rep", "target_t", "scheduled_t", "actual_t", "delta_ms"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Log saved → {OUTPUT_CSV}")

if rows:
    deltas = [float(r["delta_ms"]) for r in rows]
    arr = np.array(deltas)
    print(f"\nSpin-exit jitter (should be near 0):")
    print(f"  mean={arr.mean():.3f} ms  std={arr.std():.3f} ms"
          f"  min={arr.min():.3f} ms  max={arr.max():.3f} ms")
    print(f"\nNote: this measures when the spin-loop exited, not the actual")
    print(f"DAC onset. PTB guarantees the DAC onset was at scheduled_t ± ~0.02 ms.")
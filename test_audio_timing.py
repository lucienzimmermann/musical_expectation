# ==============================
# AUDIO TIMING TEST
# Plays files/Square Trigger Sound.wav every 3s exactly
# Logs perf_counter timestamps to audio_timing_log.csv
# ==============================

import csv
import time
import numpy as np
import soundfile as sf
import sounddevice as sd

# ==============================
# PARAMETERS
# ==============================

WAV_PATH      = "files/Square Trigger Sound.wav"
DEVICE_SR     = 44100
INTERVAL      = 3.0        # seconds between play onsets
N_REPS        = 20         # number of times to play the sound
OUTPUT_CSV    = "audio_timing_log.csv"
DEVICE_IDX    = None       # set to your Scarlett 2i2 device index if needed

# ==============================
# LOAD AUDIO
# ==============================

def load_mono_float32(filepath, target_sr=DEVICE_SR):
    data, sr = sf.read(filepath, always_2d=False)
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

print(f"Loading {WAV_PATH}...")
audio = load_mono_float32(WAV_PATH)
stereo = np.column_stack([audio, audio])  # mono → stereo
duration = len(audio) / DEVICE_SR
print(f"Loaded: {duration:.3f}s  |  {len(audio)} samples @ {DEVICE_SR} Hz\n")

# ==============================
# RUN LOOP
# ==============================

rows = []

print(f"Playing {N_REPS} times, every {INTERVAL}s.  Press Ctrl+C to abort.\n")

# Compute schedule: onset times relative to t0
# We spin-wait near each onset for maximum precision.
t0 = time.perf_counter() + 0.5   # small lead-in before first play

for rep in range(N_REPS):
    target_t = t0 + rep * INTERVAL

    # Coarse sleep to within ~5ms of target
    sleep_until = target_t - 0.005
    now = time.perf_counter()
    if sleep_until > now:
        time.sleep(sleep_until - now)

    # Spin-wait for the remaining ~5ms
    while time.perf_counter() < target_t:
        pass

    # ---- PLAY ----
    t_play = time.perf_counter()
    sd.play(stereo, samplerate=DEVICE_SR, device=DEVICE_IDX)
    t_after_play = time.perf_counter()

    delta_from_target = (t_play - target_t) * 1000   # ms
    rows.append({
        "rep":             rep + 1,
        "target_t":        f"{target_t:.6f}",
        "actual_t":        f"{t_play:.6f}",
        "delta_ms":        f"{delta_from_target:.3f}",
        "sd_call_ms":      f"{(t_after_play - t_play)*1000:.3f}",
    })

    print(f"  Rep {rep+1:3d} | target={target_t:.4f}s | actual={t_play:.4f}s "
          f"| Δ={delta_from_target:+.3f}ms | sd.play() call={((t_after_play-t_play)*1000):.3f}ms")

    # Wait for playback to finish so the next rep starts cleanly
    sd.wait()

# ==============================
# SAVE CSV
# ==============================

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["rep","target_t","actual_t","delta_ms","sd_call_ms"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone. Log saved to {OUTPUT_CSV}")

deltas = [float(r["delta_ms"]) for r in rows]
print(f"Δ mean={sum(deltas)/len(deltas):.3f}ms  "
      f"max={max(deltas):.3f}ms  min={min(deltas):.3f}ms")

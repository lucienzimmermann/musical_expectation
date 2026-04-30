# ==============================
# TRIGGER LATENCY TEST  (precision-optimised)
#
# Key improvements over original:
#   1. Trigger sent via spin-loop at the *scheduled* PTB start time,
#      not before audio scheduling — decouples variable OS overhead.
#   2. hogCPUperiod used for all short waits — no OS sleeps in the
#      critical path.
#   3. PTB clock (GetSecs) used consistently instead of core.getTime().
#   4. Process priority raised at startup (Windows: HIGH; POSIX: nice -10).
#   5. Loopback-ready: script records what it can measure (software side)
#      and reminds you to verify with the hardware feedback loop.
# ==============================

import os
import sys
import csv
import atexit
import shutil
import tempfile

import numpy as np
import soundfile as sf
from psychopy import visual, core, event, prefs

prefs.hardware['audioLib'] = ['PTB']
import psychtoolbox as ptb_audio

try:
    import serial
except ImportError:
    serial = None

# ── raise process priority ──────────────────────────────────────────────────
try:
    import psutil
    p = psutil.Process()
    if sys.platform == "win32":
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("Process priority → HIGH (Windows)")
    else:
        p.nice(-10)
        print("Process nice → -10 (POSIX)")
except Exception as e:
    print(f"Could not raise priority ({e}) — continuing anyway.")

# ==============================
# PARAMETERS  — edit these
# ==============================

STIMULUS_FILE   = "chords_sequences/context_1/Square Trigger Sound.wav"
N_REPS          = 100
ITI             = 2.0          # inter-trial blank (seconds)
TRIGGER_CODE    = 99
DEVICE_SR       = 44100
PTB_DEVICE_IDX  = 3            # Scarlett 2i2; change if needed
PTB_LATENCY_CLASS = 4          # 1=low, 2=aggressive, 4=critical (requires ASIO/CoreAudio)
SCHEDULED_LEAD  = 0.050        # seconds ahead of trigger to schedule PTB start
                               # (gives PTB time to arm before spin-loop fires)
OUTPUT_FILE     = "latency_test_results.csv"
TRIGGER_PORT    = "COM3"
TRIGGER_BAUD    = 115200
TRIGGER_RESET_DELAY = 0.010    # seconds between trigger-on and trigger-off

START_FULLSCREEN = "--windowed" not in sys.argv

# ==============================
# TRIGGER SETUP
# ==============================

def _open_trigger(port=TRIGGER_PORT, baud=TRIGGER_BAUD):
    if serial is None:
        print("pyserial not installed — triggers disabled.")
        return None
    try:
        s = serial.Serial(port=port, baudrate=baud, timeout=0)
        s.write(bytes([0]))
        print(f"TriggerBox connected on {port}")
        return s
    except Exception as e:
        print(f"TriggerBox unavailable ({e}) — triggers disabled.")
        return None

trigger_port = _open_trigger()


def send_trigger(code):
    """
    Send trigger ON, busy-wait for reset delay, send trigger OFF.
    hogCPUperiod keeps the CPU pinned — no OS scheduler wake-up jitter.
    """
    if trigger_port is None:
        return
    try:
        trigger_port.write(bytes([code]))
        core.wait(TRIGGER_RESET_DELAY, hogCPUperiod=TRIGGER_RESET_DELAY)
        trigger_port.write(bytes([0]))
    except Exception as e:
        print(f"Trigger error: {e}")

# ==============================
# AUDIO HELPERS
# ==============================

_tmp_dir = tempfile.mkdtemp(prefix="latency_test_")
atexit.register(shutil.rmtree, _tmp_dir, ignore_errors=True)


def load_mono_float32(filepath, target_sr=DEVICE_SR):
    """Load wav, mix to mono, resample if needed. Returns float32 1-D array."""
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
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        print(f"  {os.path.basename(filepath)}: resampled {sr} → {target_sr} Hz")
    return data


def _to_stereo(mono):
    """Duplicate mono array to (2, N) layout PTB expects."""
    return np.column_stack([mono, mono])

# ==============================
# AUDIO SETUP
# ==============================

print("\nOpening PTB audio stream...")
_ptb_handle = ptb_audio.PsychPortAudio(
    'Open', PTB_DEVICE_IDX, 1, PTB_LATENCY_CLASS, DEVICE_SR, 2
)
print(f"PTB stream handle: {_ptb_handle}")

print("Preloading audio...")
if not os.path.exists(STIMULUS_FILE):
    raise FileNotFoundError(f"Stimulus not found: {STIMULUS_FILE}")

audio_array   = load_mono_float32(STIMULUS_FILE)
stim_duration = len(audio_array) / DEVICE_SR
print(f"Loaded: {STIMULUS_FILE}  ({stim_duration:.3f} s)\n")

# ==============================
# PRECISION PLAY FUNCTION
# ==============================

def ptb_play_scheduled(array, lead=SCHEDULED_LEAD):
    """
    Fill buffer, schedule playback LEAD seconds from now, then spin until
    that moment and send the EEG trigger.

    Returns
    -------
    t_scheduled : float   — PTB clock time the hardware was told to start
    t_trigger   : float   — PTB clock time the trigger byte was written
    delta_ms    : float   — (t_trigger - t_scheduled) * 1000  [signed, ~0 ideal]
    """
    ptb_audio.PsychPortAudio('FillBuffer', _ptb_handle, _to_stereo(array))

    t_scheduled = ptb_audio.GetSecs() + lead
    # 'Start' with a whenToStart > 0 tells PTB to arm the hardware at that time
    ptb_audio.PsychPortAudio('Start', _ptb_handle, 1, t_scheduled, 0)

    # Spin-wait until scheduled moment — no OS sleep, stays on CPU
    while ptb_audio.GetSecs() < t_scheduled:
        pass

    t_trigger = ptb_audio.GetSecs()
    send_trigger(TRIGGER_CODE)

    delta_ms = (t_trigger - t_scheduled) * 1000
    return t_scheduled, t_trigger, delta_ms


def ptb_stop():
    ptb_audio.PsychPortAudio('Stop', _ptb_handle)


def ptb_wait_until_done(timeout=30.0):
    t0 = ptb_audio.GetSecs()
    while ptb_audio.GetSecs() - t0 < timeout:
        status = ptb_audio.PsychPortAudio('GetStatus', _ptb_handle)
        if not status['Active']:
            return
        core.wait(0.005, hogCPUperiod=0)

# ==============================
# WINDOW & UI
# ==============================

win = visual.Window(
    size=[1280, 720],
    fullscr=START_FULLSCREEN,
    color="black",
    units="height",
    allowGUI=True,
)

fixation  = visual.TextStim(win, text="+",   color="white", height=0.2)
wait_text = visual.TextStim(
    win,
    text="Press SPACE to start\n\nQ / ESC = quit",
    color="white", height=0.07, alignText="center",
)
done_text = visual.TextStim(
    win,
    text="Done!  Results saved.",
    color="white", height=0.07, alignText="center",
)

# ==============================
# HELPERS
# ==============================

def check_quit():
    """Returns True if the user pressed Q or Escape."""
    return bool(set(event.getKeys()) & {'q', 'escape'})


def wait_blank(duration):
    """Blank-screen ITI using PTB clock; checks for quit."""
    t_end = ptb_audio.GetSecs() + duration
    while ptb_audio.GetSecs() < t_end:
        if check_quit():
            raise KeyboardInterrupt
        win.flip()

# ==============================
# RUN
# ==============================

results = []

try:
    # ── Wait for SPACE ────────────────────────────────────────────────────
    while True:
        wait_text.draw()
        win.flip()
        keys = event.getKeys()
        if "space" in keys:
            break
        if set(keys) & {'q', 'escape'}:
            raise KeyboardInterrupt

    # ── Trial loop ────────────────────────────────────────────────────────
    for rep in range(N_REPS):
        print(f"Rep {rep + 1}/{N_REPS}")

        # ITI — blank screen
        wait_blank(ITI)

        # Show fixation cross
        fixation.draw()
        win.flip()

        # ── CRITICAL SECTION ──────────────────────────────────────────────
        # PTB schedules audio SCHEDULED_LEAD ms ahead; spin-loop fires the
        # trigger at the exact scheduled onset; delta ≈ 0 is the ideal.
        t_scheduled, t_trigger, delta_ms = ptb_play_scheduled(audio_array)
        # ─────────────────────────────────────────────────────────────────

        print(
            f"  t_scheduled={t_scheduled:.6f}s  "
            f"t_trigger={t_trigger:.6f}s  "
            f"Δ(scheduled→trigger)={delta_ms:+.3f} ms"
        )

        results.append({
            "rep":                rep + 1,
            "t_scheduled_s":      round(t_scheduled, 6),
            "t_trigger_s":        round(t_trigger, 6),
            "delta_scheduled_to_trigger_ms": round(delta_ms, 3),
        })

        # Keep fixation while audio plays, then stop cleanly
        ptb_wait_until_done()
        ptb_stop()

        if check_quit():
            raise KeyboardInterrupt

    # ── End screen ────────────────────────────────────────────────────────
    done_text.draw()
    win.flip()
    core.wait(2.0)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

# ==============================
# SAVE & SUMMARY
# ==============================

fieldnames = [
    "rep",
    "t_scheduled_s",
    "t_trigger_s",
    "delta_scheduled_to_trigger_ms",
]

with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {OUTPUT_FILE}  ({len(results)} reps recorded)")

if results:
    deltas = [r["delta_scheduled_to_trigger_ms"] for r in results]
    print(
        f"\nΔ (scheduled → trigger write)\n"
        f"  mean = {np.mean(deltas):+.3f} ms\n"
        f"  std  = {np.std(deltas):.3f} ms\n"
        f"  min  = {np.min(deltas):+.3f} ms\n"
        f"  max  = {np.max(deltas):+.3f} ms\n"
    )
    print(
        "NOTE: the delta above measures spin-loop precision only.\n"
        "Use your hardware feedback loop to verify end-to-end latency.\n"
        "A stable delta here (std < 1 ms) means jitter is downstream\n"
        "(USB serial, soundcard buffer, or measurement chain)."
    )

# ==============================
# CLEANUP
# ==============================

ptb_audio.PsychPortAudio('Close', _ptb_handle)
win.close()
core.quit()
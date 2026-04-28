#edited version of trigger latency with changes trying to diminish the variation of the latency

# ==============================
# TRIGGER LATENCY TEST
# Plays a single stimulus N times, sends a trigger each time.
# Keeps audio/PTB/trigger structure identical to the main experiment.
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

# ==============================
# PARAMETERS  — edit these
# ==============================

STIMULUS_FILE  = "chords_sequences/context_1/your_stimulus.wav"  # path to the single .wav to test
N_REPS         = 20           # how many times to play it
ITI            = 2.0          # inter-trial interval (seconds), blank screen
TRIGGER_CODE   = 99           # EEG trigger value sent at playback start
DEVICE_SR      = 44100
PTB_DEVICE_IDX = 3            # Scarlett 2i2; change if needed
OUTPUT_FILE    = "latency_test_results.csv"
START_FULLSCREEN = "--windowed" not in sys.argv

# ==============================
# TRIGGER SETUP  (identical to main experiment)
# ==============================

def _open_trigger(port="COM3", baud=115200):
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

# ==============================
# AUDIO HELPERS  (identical to main experiment)
# ==============================

_tmp_dir = tempfile.mkdtemp(prefix="latency_test_")
atexit.register(shutil.rmtree, _tmp_dir, ignore_errors=True)

def load_mono_float32(filepath, target_sr=DEVICE_SR):
    """Load a wav file, mix to mono, resample if needed. Returns float32 1-D array."""
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
            data = np.interp(np.linspace(0, len(data) - 1, n),
                             np.arange(len(data)), data).astype(np.float32)
        print(f"  {os.path.basename(filepath)}: resampled {sr} → {target_sr} Hz")
    return data

# ==============================
# AUDIO SETUP  (identical to main experiment)
# ==============================

print("\nOpening PTB audio stream...")
_ptb_handle = ptb_audio.PsychPortAudio(
    'Open',
    PTB_DEVICE_IDX,
    1,                  # playback
    4,                  # low-latency mode
    DEVICE_SR,
    2,                  # stereo
    [],                 # buffer size (frames) → usually leave empty
    [],                 # suggested latency (deprecated)
    0.005               # latency/buffer
)
print(f"PTB stream handle: {_ptb_handle}")

print("Preloading audio...")
if not os.path.exists(STIMULUS_FILE):
    raise FileNotFoundError(f"Stimulus not found: {STIMULUS_FILE}")
audio_array = load_mono_float32(STIMULUS_FILE)
stim_duration = len(audio_array) / DEVICE_SR
print(f"Loaded: {STIMULUS_FILE}  ({stim_duration:.3f} s)\n")

def _to_stereo(mono):
    """Duplicate mono array to (2, N) interleaved layout PTB expects."""
    return np.column_stack([mono, mono])

def ptb_play(array):
    """Load buffer and start non-blocking playback."""
    ptb_audio.PsychPortAudio('FillBuffer', _ptb_handle, _to_stereo(array))

    now = ptb_audio.PsychPortAudio('GetStatus', _ptb_handle)['CurrentTime'] # CHHANGE NUMBER 1 : Schedule play
    when = now + 0.1  # schedule 100 ms in future
    ptb_audio.PsychPortAudio('Start', _ptb_handle, 1, when, 1)

def ptb_stop():
    ptb_audio.PsychPortAudio('Stop', _ptb_handle)

def ptb_wait_until_done(timeout=30.0):
    """Block until PTB reports playback finished."""
    t0 = core.getTime()
    while core.getTime() - t0 < timeout:
        status = ptb_audio.PsychPortAudio('GetStatus', _ptb_handle)
        if not status['Active']:
            return
        core.wait(0.005, hogCPUperiod=0)

# ==============================
# WINDOW & UI  (minimal — no instructions or ratings)
# ==============================

win = visual.Window(
    size=[1280, 720],
    fullscr=START_FULLSCREEN,
    color="black",
    units="height",
    allowGUI=True,
)

fixation  = visual.TextStim(win, text="+",  color="white", height=0.2)
wait_text = visual.TextStim(win, text="Press SPACE to start\n\nQ / ESC = quit",
                             color="white", height=0.07, alignText="center")
done_text = visual.TextStim(win, text="Done!  Results saved.",
                             color="white", height=0.07, alignText="center")

# ==============================
# HELPERS
# ==============================

def check_window_keys():
    """Returns True if quit was requested."""
    keys = event.getKeys()
    if set(keys) & {'q', 'escape'}:
        return True
    return False

def send_trigger(code):
    if trigger_port is None:
        return
    try:
        trigger_port.write(bytes([code]))
        trigger_port.write(bytes([0]))
    except Exception as e:
        print(f"Trigger error: {e}")

def wait_blank(duration):
    clock = core.Clock()
    while clock.getTime() < duration:
        if check_window_keys():
            raise KeyboardInterrupt
        win.flip()

# ==============================
# RUN
# ==============================

results = []

try:
    # Wait for SPACE
    while True:
        wait_text.draw()
        win.flip()
        keys = event.getKeys()
        if "space" in keys:
            break
        if set(keys) & {'q', 'escape'}:
            raise KeyboardInterrupt

    for rep in range(N_REPS):
        print(f"Rep {rep + 1}/{N_REPS}")

        # Pre-trial blank (ITI)
        wait_blank(ITI)

        # Fixation on screen
        fixation.draw()
        win.flip()

        # --- critical timing sequence (same as main experiment) ---
        t_trigger = core.getTime()

        now = ptb_audio.PsychPortAudio('GetStatus', _ptb_handle)['CurrentTime'] # CHHANGE NUMBER 1 : Schedule play
        when = now + 0.1  # schedule 100 ms in future
        core.wait(when - core.getTime())
        send_trigger(TRIGGER_CODE)
        ptb_play(audio_array)
        t_play_call = core.getTime()
        # ----------------------------------------------------------

        estimated_delay_ms = (t_play_call - t_trigger) * 1000
        print(f"  t_trigger={t_trigger:.6f}s  "
              f"t_play_call={t_play_call:.6f}s  "
              f"Δ(trigger→ptb_play return)={estimated_delay_ms:.3f} ms")

        results.append({
            "rep":              rep + 1,
            "t_trigger_s":      round(t_trigger, 6),
            "t_play_call_s":    round(t_play_call, 6),
            "delta_trigger_to_play_call_ms": round(estimated_delay_ms, 3),
        })

        # Keep fixation visible while audio plays
        while True:
            status = ptb_audio.PsychPortAudio('GetStatus', _ptb_handle)
            if not status['Active']:
                break
            if check_window_keys():
                raise KeyboardInterrupt
            fixation.draw()
            win.flip()

        ptb_stop()

    # End screen
    done_text.draw()
    win.flip()
    core.wait(2.0)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

# ==============================
# SAVE & CLOSE
# ==============================

fieldnames = ["rep", "t_trigger_s", "t_play_call_s", "delta_trigger_to_play_call_ms"]
with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {OUTPUT_FILE}  ({len(results)} reps recorded)")

# Print summary
if results:
    deltas = [r["delta_trigger_to_play_call_ms"] for r in results]
    print(f"Δ trigger→ptb_play  mean={np.mean(deltas):.3f} ms  "
          f"std={np.std(deltas):.3f} ms  "
          f"min={np.min(deltas):.3f} ms  max={np.max(deltas):.3f} ms")

ptb_audio.PsychPortAudio('Close', _ptb_handle)
win.close()
core.quit()
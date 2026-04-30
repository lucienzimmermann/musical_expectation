# ==============================
# CHORD SEQUENCE EXPERIMENT
# Optimized for Scarlett 2i2 on Windows + PTB
# Precision-timed triggers via scheduled PTB onset + spin-loop
# ==============================

import os
import sys
import random
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
# PARAMETERS
# ==============================

BASE_PATH         = "chords_sequences"
N_TRIALS          = 30
BREAK_MEAN        = 1.5
BREAK_JITTER      = 0.3
OUTPUT_FILE       = "results.csv"
DEVICE_SR         = 44100
NOISE_DURATION    = 1.0
CONTEXTS          = ["context_1", "context_2"]
TRIGGER_MAP       = {"context_1": 10, "context_2": 20}
PTB_LATENCY_CLASS = 4          # 1=low, 2=aggressive, 4=critical (ASIO/CoreAudio)
SCHEDULED_LEAD    = 0.050      # seconds PTB is armed ahead of trigger spin-loop
TRIGGER_PORT      = "COM3"
TRIGGER_BAUD      = 115200
TRIGGER_RESET_DELAY = 0.001    # 1 ms — BP TriggerBox minimum pulse width
START_FULLSCREEN  = "--windowed" not in sys.argv

if N_TRIALS % len(CONTEXTS) != 0:
    raise ValueError(f"N_TRIALS ({N_TRIALS}) must be divisible by {len(CONTEXTS)}")

# ==============================
# DEVICE SELECTION
# ==============================

def choose_ptb_device():
    """List available PTB devices and ask user to select one."""
    devices = ptb_audio.PsychPortAudio('GetDevices')
    print("\nAvailable audio devices:\n")
    valid_devices = []
    for i, dev in enumerate(devices):
        if dev['NrOutputChannels'] > 0:
            print(f"[{i}] {dev['DeviceName']} "
                  f"(Outputs: {dev['NrOutputChannels']}, "
                  f"Host API: {dev['HostAudioAPIName']})")
            valid_devices.append(i)
    if not valid_devices:
        raise RuntimeError("No output audio devices found!")
    while True:
        try:
            choice = int(input("\nEnter device index: "))
            if choice in valid_devices:
                return choice
            print("Invalid choice. Pick a listed index.")
        except ValueError:
            print("Please enter a number.")

PTB_DEVICE_IDX = choose_ptb_device()

# ==============================
# TRIGGER SETUP
# ==============================

def _open_trigger(port=TRIGGER_PORT, baud=TRIGGER_BAUD):
    if serial is None:
        print("pyserial not installed — triggers disabled.")
        return None
    try:
        s = serial.Serial(port=port, baudrate=baud, timeout=0, write_timeout=0)
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

_tmp_dir = tempfile.mkdtemp(prefix="chord_exp_")
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


def rms(arr):
    return float(np.sqrt(np.mean(arr ** 2)))


def normalize_to_rms(arr, target_rms):
    r = rms(arr)
    return arr * (target_rms / r) if r > 1e-9 else arr


def _to_stereo(mono):
    """Duplicate mono array to (2, N) interleaved layout PTB expects."""
    return np.column_stack([mono, mono])

# ==============================
# LOAD & BALANCE TRIALS
# ==============================

files_dict = {}
for ctx in CONTEXTS:
    path = os.path.join(BASE_PATH, ctx)
    if not os.path.exists(path):
        raise ValueError(f"Folder not found: {path}")
    wavs = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".wav")]
    if not wavs:
        raise ValueError(f"No .wav files in: {path}")
    files_dict[ctx] = wavs

trials_per_context = N_TRIALS // len(CONTEXTS)
selected_trials = []
for ctx in CONTEXTS:
    pool = files_dict[ctx]
    chosen = (random.sample(pool, trials_per_context)
              if len(pool) >= trials_per_context
              else random.choices(pool, k=trials_per_context))
    for f in chosen:
        selected_trials.append({"context": ctx, "file": f, "name": os.path.basename(f)})
random.shuffle(selected_trials)

print(f"\nTrial order ({len(selected_trials)} trials):")
for t in selected_trials:
    print(f"  [{t['context']}] {t['name']}")

# ==============================
# AUDIO SETUP
# ==============================

print("\nOpening PTB audio stream...")
_ptb_handle = ptb_audio.PsychPortAudio(
    'Open', PTB_DEVICE_IDX, 1, PTB_LATENCY_CLASS, DEVICE_SR, 2
)
print(f"PTB stream handle: {_ptb_handle}")

print("Preloading audio...")
_audio_cache = {}
for trial in selected_trials:
    src = trial["file"]
    if src not in _audio_cache:
        _audio_cache[src] = load_mono_float32(src)
    trial["audio_array"] = _audio_cache[src]
    trial["duration"]    = len(_audio_cache[src]) / DEVICE_SR

_mean_chord_rms = float(np.mean([rms(a) for a in _audio_cache.values()]))
_raw_noise = np.random.normal(0, 1.0, int(DEVICE_SR * NOISE_DURATION)).astype(np.float32)
_noise_array = normalize_to_rms(_raw_noise, _mean_chord_rms)
print(f"Noise RMS matched to chords: {_mean_chord_rms:.5f}")
print("All audio preloaded.\n")

# ==============================
# PRECISION PLAY FUNCTION
# ==============================

def ptb_play_scheduled(array, trigger_code, lead=SCHEDULED_LEAD):
    """
    Fill buffer, schedule playback LEAD seconds from now, then spin until
    that moment and send the EEG trigger.

    The trigger is sent at the exact scheduled hardware onset rather than
    before PTB scheduling, eliminating variable scheduling overhead from
    the trigger-to-audio alignment.
    """
    ptb_audio.PsychPortAudio('FillBuffer', _ptb_handle, _to_stereo(array))

    t_scheduled = ptb_audio.GetSecs() + lead
    ptb_audio.PsychPortAudio('Start', _ptb_handle, 1, t_scheduled, 0)

    # Spin-wait until scheduled moment — no OS sleep, stays on CPU
    while ptb_audio.GetSecs() < t_scheduled:
        pass

    send_trigger(trigger_code)


def ptb_play_plain(array):
    """
    Non-triggered playback (used for noise bursts).
    Simple fill + start, no scheduling needed.
    """
    ptb_audio.PsychPortAudio('FillBuffer', _ptb_handle, _to_stereo(array))
    ptb_audio.PsychPortAudio('Start', _ptb_handle, 1, 0, 1)


def ptb_stop():
    ptb_audio.PsychPortAudio('Stop', _ptb_handle)


def ptb_wait_until_done(timeout=30.0):
    """Block until PTB reports playback finished."""
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

fixation = visual.TextStim(win, text="+", color="white", height=0.2)

question_text = visual.TextStim(
    win,
    text=(
        "How complete did the chord sequence feel?\n\n"
        "1 = Not surprising at all\n"
        "5 = Moderately surprising\n"
        "9 = Highly surprising"
    ),
    color="white", height=0.06, wrapWidth=1.5, pos=(0, 0.2), alignText="center",
)

start_text = visual.TextStim(
    win,
    text=(
        "Press SPACE to start\n\n"
        "F = toggle fullscreen   M = windowed mode\n"
        "Q / ESC = quit"
    ),
    color="white", height=0.06, wrapWidth=1.5, alignText="center",
)

hint_text = visual.TextStim(
    win, text="F = fullscreen  M = windowed  Q = quit",
    color="grey", height=0.035, pos=(0, -0.45),
)

_BW, _BH = 0.04, 0.04
buttons, labels = [], []
for i in range(9):
    x = -0.4 + i * 0.1
    buttons.append(visual.ShapeStim(
        win,
        vertices=[[-_BW, -_BH], [_BW, -_BH], [_BW, _BH], [-_BW, _BH]],
        pos=(x, -0.2), fillColor="darkgrey", lineColor="white", closeShape=True,
    ))
    labels.append(visual.TextStim(win, text=str(i + 1), pos=(x, -0.2),
                                  height=0.06, color="white"))

mouse = event.Mouse(win=win)

# ==============================
# RUNTIME HELPERS
# ==============================

def toggle_fullscreen():
    win.fullscr = not win.fullscr
    win._isFullScr = win.fullscr
    win.winHandle.set_fullscreen(win.fullscr)
    win.winHandle.activate()


def set_windowed():
    if win.fullscr:
        toggle_fullscreen()


def check_window_keys():
    """Handle window-management keys. Returns True if quit requested."""
    keys = event.getKeys()
    if set(keys) & {'q', 'escape'}:
        return True
    if 'f' in keys:
        toggle_fullscreen()
    if 'm' in keys:
        set_windowed()
    return False


def wait_blank(duration):
    """Blank-screen wait using PTB clock; checks for quit."""
    t_end = ptb_audio.GetSecs() + duration
    while ptb_audio.GetSecs() < t_end:
        if check_window_keys():
            raise KeyboardInterrupt
        win.flip()


def get_rating():
    """Display rating screen and return 1–9 via mouse click or keyboard."""
    mouse.clickReset()
    while True:
        if check_window_keys():
            raise KeyboardInterrupt
        question_text.draw()
        for rect, lbl in zip(buttons, labels):
            rect.draw()
            lbl.draw()
        hint_text.draw()
        win.flip()
        for i, rect in enumerate(buttons):
            if mouse.isPressedIn(rect):
                core.wait(0.2)
                return i + 1
        for key in event.getKeys():
            if key in [str(i) for i in range(1, 10)]:
                core.wait(0.2)
                return int(key)

# ==============================
# RUN EXPERIMENT
# ==============================

results = []

try:
    # Start screen
    while True:
        start_text.draw()
        win.flip()
        keys = event.getKeys()
        if "space" in keys:
            break
        if set(keys) & {'q', 'escape'}:
            raise KeyboardInterrupt
        if 'f' in keys:
            toggle_fullscreen()
        if 'm' in keys:
            set_windowed()

    for trial_num, trial in enumerate(selected_trials):
        print(f"\nTrial {trial_num + 1}/{len(selected_trials)}: "
              f"{trial['name']} [{trial['context']}]")

        # Pre-trial blank
        wait_blank(BREAK_MEAN)

        trigger_code = TRIGGER_MAP.get(trial["context"], 0)

        # Fixation on screen
        fixation.draw()
        win.flip()

        # ── CRITICAL SECTION ──────────────────────────────────────────────
        # PTB schedules audio SCHEDULED_LEAD ahead; spin-loop fires the
        # trigger at the exact scheduled onset.
        ptb_play_scheduled(trial["audio_array"], trigger_code)
        # ─────────────────────────────────────────────────────────────────

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

        # Rating
        rating = get_rating()
        print(f"  Rating: {rating}")
        results.append({
            "chord progression name": trial["name"],
            "context":                trial["context"],
            "grade":                  rating,
        })

        win.flip()

        # Post-trial blank + noise + second blank
        jitter = np.random.uniform(-BREAK_JITTER, BREAK_JITTER)
        wait_blank(BREAK_MEAN + jitter)

        ptb_play_plain(_noise_array)
        ptb_wait_until_done(timeout=NOISE_DURATION + 1.0)
        ptb_stop()

        wait_blank(BREAK_MEAN - 1.0)

except KeyboardInterrupt:
    print("\nExperiment interrupted by user.")

# ==============================
# SAVE & CLOSE
# ==============================

with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["chord progression name", "context", "grade"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {OUTPUT_FILE}  ({len(results)} trials recorded)")

ptb_audio.PsychPortAudio('Close', _ptb_handle)
win.close()
core.quit()
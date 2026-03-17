from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']

import os
import random
import csv
import numpy as np
import soundfile as sf
from psychopy import visual, core, event, sound

# ==============================
# -------- PARAMETERS ----------
# ==============================

BASE_PATH = "chords_sequences"
N_TRIALS = 30
BREAK_MEAN = 1.5
BREAK_JITTER = 0.3
OUTPUT_FILE = "results.csv"

contexts = ["context_1", "context_2"]


# ==============================
# -------- LOAD FILES ----------
# ==============================

files_dict = {}

for ctx in contexts:
    path = os.path.join(BASE_PATH, ctx)
    
    if not os.path.exists(path):
        raise ValueError(f"Folder not found: {path}")
    
    wav_files = [f for f in os.listdir(path) if f.endswith(".wav")]
    
    if len(wav_files) == 0:
        raise ValueError(f"No wav files in {ctx}")
    
    full_paths = [os.path.join(path, f) for f in wav_files]
    files_dict[ctx] = full_paths
    print(wav_files)

# ==============================
# ---- BALANCED RANDOM PICK ----
# ==============================

if N_TRIALS % 2 != 0:
    raise ValueError("N_TRIALS must be divisible by 2")

trials_per_context = N_TRIALS // 2
selected_trials = []

for ctx in contexts:
    available_files = files_dict[ctx]
    
    if len(available_files) >= trials_per_context:
        chosen = random.sample(available_files, trials_per_context)
    else:
        chosen = random.choices(available_files, k=trials_per_context)
    
    for file in chosen:
        selected_trials.append({
            "context": ctx,
            "file": file,
            "name": os.path.basename(file)
        })

random.shuffle(selected_trials)
print(selected_trials)

# ==============================
# -------- PSYCHOPY UI ---------
# ==============================

# Create window
win = visual.Window(
    size=[1920, 1080],
    fullscr=True,
    color="black",
    units="height"
)

# Question text (centered, larger, and more elegant)
question_text = visual.TextStim(
    win,
    text="How surprising was the last chord?\n\n1 = Not surprising at all\n5 = Moderately surprising\n9 = Highly surprising",
    color="white",
    height=0.08,
    wrapWidth=1.5,
    pos=(0, 0.2),
    alignText="center"
)

# Create buttons and labels (centered, visually appealing)
buttons = []
labels = []

for i in range(9):
    x_pos = -0.4 + i * 0.1
    w, h = 0.04, 0.04  # half-width, half-height
    rect = visual.ShapeStim(
        win,
        vertices=[[-w, -h], [w, -h], [w, h], [-w, h]],
        pos=(x_pos, -0.2),
        fillColor="darkgrey",
        lineColor="white",
        closeShape=True
    )
    label = visual.TextStim(
        win,
        text=str(i+1),
        pos=(x_pos, -0.2),
        height=0.06,
        color="white"
    )

    buttons.append(rect)
    labels.append(label)

# Mouse setup
mouse = event.Mouse(win=win)

# ==============================
# ---- WHITE NOISE SETUP -------
# ==============================

NOISE_DURATION = 1.0
NOISE_SR = 44100
noise_array = np.random.normal(0, 0.2, int(NOISE_SR * NOISE_DURATION)) * 0.10
# PsychoPy Sound expects shape (n_samples, n_channels) or (n_samples,)
white_noise_sound = sound.Sound(noise_array, sampleRate=NOISE_SR, stereo=False)

# ==============================
# -------- RUN EXPERIMENT ------
# ==============================

results = []
counter = 0

fixation = visual.TextStim(
    win,
    text="+",
    color="white",
    height=0.2)


def check_quit():
    keys = event.getKeys()
    if 'q' in keys or 'escape' in keys:
        return True
    return False


try:
    for trial in selected_trials:

        # -------- START SCREEN --------
        if counter == 0:
            start_text = visual.TextStim(
                win,
                text="Press SPACE when you are ready to start the experiment\n\n(Press Q or ESC to quit)",
                color="white",
                height=0.06,
                wrapWidth=1.5
            )

            waiting = True
            while waiting:
                start_text.draw()
                win.flip()
                counter = counter + 1
                keys = event.getKeys()

                if "space" in keys:
                    waiting = False

                if "q" in keys or "escape" in keys:
                    win.close()
                    core.quit()

        # ---- Quick Break -------

        pause_time = BREAK_MEAN

        pause_clock = core.Clock()
        while pause_clock.getTime() < pause_time:
            if check_quit():
                raise KeyboardInterrupt
            win.flip()

        # ---- FIXATION + PLAYBACK -----

        # Load and mix down to mono manually
        snd_data, snd_sr = sf.read(trial["file"])
        if snd_data.ndim > 1:
            snd_data = snd_data.mean(axis=1)  # stereo -> mono
        snd_data = snd_data.astype(np.float32)
        duration = len(snd_data) / snd_sr

        trial_sound = sound.Sound(snd_data, sampleRate=snd_sr, stereo=False)

        trial_sound.play()

        clock = core.Clock()
        while clock.getTime() < duration:
            if check_quit():
                raise KeyboardInterrupt
            fixation.draw()
            win.flip()

        trial_sound.stop()

        # -------- RATING --------------

        mouse.clickReset()
        rating = None

        while rating is None:

            if check_quit():
                raise KeyboardInterrupt

            question_text.draw()
            for rect, label in zip(buttons, labels):
                rect.draw()
                label.draw()
            win.flip()

            # ---- Mouse input ----
            for i, rect in enumerate(buttons):
                if mouse.isPressedIn(rect):
                    rating = i + 1
                    core.wait(0.2)

            # ---- Keyboard input ----
            keys = event.getKeys()

            for key in keys:
                if key in [str(i) for i in range(1, 10)]:
                    rating = int(key)
                    core.wait(0.2)

        # -------- LOG RESULT ----------

        results.append({
            "chord progression name": trial["name"],
            "context": trial["context"],
            "grade": rating
        })

        # Clear screen after click
        win.flip()

        # -------- PAUSE ---------------

        jitter = np.random.uniform(-BREAK_JITTER, BREAK_JITTER)
        pause_time = BREAK_MEAN + jitter

        pause_clock = core.Clock()
        while pause_clock.getTime() < pause_time:
            if check_quit():
                raise KeyboardInterrupt
            win.flip()

        # -------- WHITE NOISE ---------

        white_noise_sound.play()
        core.wait(NOISE_DURATION)
        white_noise_sound.stop()

        # -------- PAUSE ---------------

        pause_time = BREAK_MEAN - 1

        pause_clock = core.Clock()
        while pause_clock.getTime() < pause_time:
            if check_quit():
                raise KeyboardInterrupt
            win.flip()

except KeyboardInterrupt:
    print("Experiment stopped early by user.")

# ==============================
# -------- SAVE DATA -----------
# ==============================

with open(OUTPUT_FILE, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["chord progression name", "context", "grade"])
    writer.writeheader()
    writer.writerows(results)

win.close()
core.quit()
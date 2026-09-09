# Per-gesture summary across all 119 trials for one participant.

import numpy as np

from load import FOREARM, GESTURES, TRIALS, load_trial

rows = []

for gesture, name in GESTURES.items():
    # RMS per channel, per trial. axis=0 averages over time, so need to convert all waveforms to +ve
    # leaving one number per channel, so stacking the 7 trials gives shape (7, 16).
    per_trial = np.stack([
        np.sqrt(np.mean(load_trial(1, 1, gesture, trial) ** 2, axis=0))
        for trial in TRIALS
    ])
    # np.sqrt returns value to millivolts instead of keeping it in millivolts squared 
    # Forgetting to do this gave weird and incorrect results

    channels = per_trial.mean(axis=0)              # (16,) averaged over trials
    rms = channels.mean()                          # overall loudness
    sd = per_trial.mean(axis=1).std()              # trial-to-trial variability
    rows.append((rms, sd, name, FOREARM[channels.argmax()]))

for rms, sd, name, loudest in sorted(rows, reverse=True):
    print(f"{name:<36} {rms:.5f}  +/-{sd:.5f}   {loudest}")
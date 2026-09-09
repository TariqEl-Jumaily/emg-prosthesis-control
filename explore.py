# First look at the data: one electrode during rest, hand open and hand close.
# Scratch script, nothing imports it.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb

DATA = Path("data/raw/Session1/session1_participant1")

OUTPUT = Path("data/explore_data")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Channels are named in the header: F1-F16 forearm, W1-W12 wrist, U1-U4 unused.
# Asking by name means I can't accidentally grab one of the four dead columns.
FOREARM = [f"F{i}" for i in range(1, 17)]

GESTURES = {17: "rest", 15: "hand open", 16: "hand close"}
CHANNEL, TRIAL = 0, 1        # CHANNEL indexes into FOREARM, so 0 is "F1"

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=True)

for ax, (gesture, name) in zip(axes, GESTURES.items()):
    stem = f"session1_participant1_gesture{gesture}_trial{TRIAL}"

    # No extension: rdrecord reads the .hea to learn the format, then the .dat.
    record = wfdb.rdrecord(str(DATA / stem), channel_names=FOREARM)

    # p_signal is already divided by the gain from the header, so millivolts.
    signal = record.p_signal[:, CHANNEL]
    rms = np.sqrt(np.mean(signal ** 2))      # standard one-number EMG amplitude

    ax.plot(np.arange(len(signal)) / record.fs, signal, linewidth=0.4)
    ax.set_ylabel("mV")
    ax.set_title(f"{name}  (RMS {rms:.5f} mV)", loc="left", fontsize=10)

axes[-1].set_xlabel("time (s)")
fig.suptitle(f"participant 1, session 1, trial {TRIAL}, channel {FOREARM[CHANNEL]}")
fig.tight_layout()
plt.show()

# Save the figure
filename = (
    f"participant1_session1_trial{TRIAL}_channel{FOREARM[CHANNEL]}.png"
)
output_path = OUTPUT / filename

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved plot to: {output_path}")
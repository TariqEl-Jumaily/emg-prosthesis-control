# The function of this script is to read GRABMyo trials off the disk and hand back plain numpy arrays.

#Everything else in the project imports from here, and nothing else imports wfdb.


from pathlib import Path

import numpy as np
import wfdb

# Where get_data.py puts things.
DATA_ROOT = Path("data/raw")

# Sampling frequency: the rig measured the voltage 2048 times per second.
FS = 2048

# The header names all 32 channels: F1-F16 forearm, W1-W12 wrist, U1-U4 unused.
# Asking by name instead of slicing columns 0:16 means I can't accidentally pick up one of the four dead channels.
FOREARM = []
for i in range(1, 17):
    FOREARM.append(f"F{i}")       # Will save as "F1", "F2", "F3", ...)

# 16 gestures from the dataset's GestureList, plus rest, which is number 17. (Dictionary/Lookup table)
GESTURES = {
    1: "lateral prehension",
    2: "thumb adduction",
    3: "thumb + little finger opposition",
    4: "thumb + index finger opposition",
    5: "thumb + little finger extension",
    6: "thumb + index finger extension",
    7: "index + middle finger extension",
    8: "little finger extension",
    9: "index finger extension",
    10: "thumb extension",
    11: "wrist flexion",
    12: "wrist extension",
    13: "forearm supination",
    14: "forearm pronation",
    15: "hand open",
    16: "hand close",
    17: "rest",
}

TRIALS = range(1, 8)     # 7 repetitions of each gesture, 5 seconds each


def load_trial(session, participant, gesture, trial):
    # Rows are time samples, columns are F1 to F16 in that order.
    
    folder = f"session{session}_participant{participant}"
    stem = f"{folder}_gesture{gesture}_trial{trial}"
    path = DATA_ROOT / f"Session{session}" / folder / stem

    # No file extension: wfdb reads the .hea to learn the format, then the .dat.
    record = wfdb.rdrecord(str(path), channel_names=FOREARM)

    # wfdb returns float64. That's pointless precision for a signal that came
    # off a 16 bit ADC, and twice the memory, which matters once loading thousands of trials
    # at once.
    return record.p_signal.astype(np.float32)

if __name__ == "__main__":
    x = load_trial(1, 1, 15, 1)
    print(x.shape, x.dtype, f"{x.min():.5f} to {x.max():.5f} mV")
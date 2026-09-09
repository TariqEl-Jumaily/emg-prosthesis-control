"""
Filters a trial and cuts it into short overlapping windows.

A window is one example for the classifier. Everything from here on works on
windows, not on whole trials.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from load import FS


def bandpass(x, low=20, high=450):
    """Keep only the frequencies where muscle activity lives.

    Below 20 Hz is electrode movement and drift, above 450 Hz is mostly
    equipment noise. This data was already filtered in hardware at 10 to
    500 Hz, so this is a second pass.
    """
    # Design the filter. 4 is the "order", meaning how sharply it cuts off.
    sos = butter(4, [low, high], btype="bandpass", fs=FS, output="sos")

    # Apply it. axis=0 means run down the time direction, one column at a time.
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


def window(x, win_ms=200, hop_ms=50, trim_s=0.5):
    #Cut a trial into overlapping windows.
    #Input is one trial, shape (10240, 16).
    #Output is (77, 410, 16): 77 windows, each 410 samples long, 16 channels.
    
    win = round(win_ms / 1000 * FS)      # 200 ms -> 410 samples
    hop = round(hop_ms / 1000 * FS)      # 50 ms  -> 102 samples
    trim = round(trim_s * FS)            # 0.5 s  -> 1024 samples

    # Drop the ramp in at the start and the relax out at the end. Those
    # moments are labelled as the gesture but are really half rest.
    x = x[trim:len(x) - trim]

    # Every position a window can start at without running off the end.
    starts = range(0, len(x) - win + 1, hop)

    return np.stack([x[s:s + win] for s in starts])


if __name__ == "__main__":
    from load import load_trial

    raw = load_trial(1, 1, 15, 1)
    windows = window(bandpass(raw))

    print(f"trial   {raw.shape}")
    print(f"windows {windows.shape}")
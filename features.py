#Turns windows into the Hudgins time-domain feature set: four numbers per
#channel, so 64 for our 16 channels.

#Proposed in 1993, still the standard baseline in myoelectric control, mainly
#because it is cheap enough to run on a microcontroller and captures both
#amplitude and frequency content without an FFT.

import numpy as np


def hudgins(w, threshold=1e-4):

    # MAV: mean absolute value. Amplitude.
    mav = np.mean(np.abs(w), axis=1)

    # Difference between each sample and the next. Used by the three below.
    d = np.diff(w, axis=1)

    # WL: waveform length. Total distance the trace travels. Rises with both amplitude and frequency.
    wl = np.sum(np.abs(d), axis=1)

    # ZC: zero crossings. Count sign changes, but only if the two samples are
    # far enough apart to be real signal rather than noise.
    a, b = w[:, :-1], w[:, 1:]
    zc = np.sum((np.sign(a) != np.sign(b)) & (np.abs(a - b) > threshold), axis=1)

    # SSC: slope sign changes. Count how often the gradient flips, which is
    # how many peaks and troughs there are.
    d1, d2 = d[:, :-1], d[:, 1:]
    ssc = np.sum((d1 * d2 < 0) & ((np.abs(d1) > threshold) | (np.abs(d2) > threshold)),
                 axis=1)

    # Stick the four blocks side by side: 16 + 16 + 16 + 16 = 64 columns.
    return np.concatenate([mav, wl, zc, ssc], axis=1).astype(np.float32)
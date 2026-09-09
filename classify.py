# First attempt at a classifier. Two gestures, one feature, one participant.
# The point is to get an end to end path from files to a number, not to be good.


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from load import TRIALS, load_trial
from prep import bandpass, window

CLASSES = {15: "hand open", 16: "hand close"}

X = []      # features: 16 numbers per window
y = []      # labels: 0 or 1, one per window

for label, gesture in enumerate(CLASSES):
    for trial in TRIALS:
        w = window(bandpass(load_trial(1, 1, gesture, trial)))

        # MAV: average size of the frequency on each channel, ignoring sign.
        # axis=1 collapses time, so (77, 410, 16) becomes (77, 16).
        X.append(np.mean(np.abs(w), axis=1))

        # Every window from this trial gets the same label.
        y.append(np.full(len(w), label))

X = np.concatenate(X)
y = np.concatenate(y)
print("windows:", X.shape)

# Hold back 30% of the windows to test on.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearDiscriminantAnalysis().fit(X_train, y_train)
print(f"accuracy: {model.score(X_test, y_test):.3f}")
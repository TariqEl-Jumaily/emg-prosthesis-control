# First attempt at a classifier. Two gestures, one feature, one participant.
# The point is to get an end to end path from files to a number, not to be good.


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from load import TRIALS, load_trial
from prep import bandpass, window

CLASSES = {15: "hand open", 16: "hand close"}

TRAIN_TRIALS = (1, 2, 3, 4, 5)
TEST_TRIALS = (6, 7)

X = []      # features: 16 numbers per window
Y = []      # labels: 0 or 1, one per window

trial_of = []      # remember which trial each window came from

for label, gesture in enumerate(CLASSES):
    for trial in TRIALS:
        w = window(bandpass(load_trial(1, 1, gesture, trial)))

        # MAV: average size of the frequency on each channel, ignoring sign.
        # axis=1 collapses time, so (77, 410, 16) becomes (77, 16).
        X.append(np.mean(np.abs(w), axis=1))

        # Every window from this trial gets the same label.
        Y.append(np.full(len(w), label))

        trial_of.append(np.full(len(w), trial))
        

X = np.concatenate(X)
Y = np.concatenate(Y)
trial_of = np.concatenate(trial_of)


# Hold back 30% of the windows to test on.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
random_acc = LinearDiscriminantAnalysis().fit(X_train, Y_train).score(X_test, Y_test)

train = np.isin(trial_of, TRAIN_TRIALS)
test = np.isin(trial_of, TEST_TRIALS)
assert not (train & test).any(), "a window ended up in both splits"

trial_acc = LinearDiscriminantAnalysis().fit(X[train], Y[train]).score(X[test], Y[test])


print(f"random window split : {random_acc:.3f}")
print(f"held-out trials     : {trial_acc:.3f}")
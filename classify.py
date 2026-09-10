# All 17 gestures, Hudgins TD4 features, LDA with shrinkage.
# Held-out trials only. The random split is gone, it was never real.

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import hudgins
from load import GESTURES, TRIALS, load_trial
from prep import bandpass, window

TRAIN_TRIALS = (1, 2, 3, 4, 5)
TEST_TRIALS = (6, 7)

X = []
y = []
trial_of = []

for gesture in GESTURES:
    for trial in TRIALS:
        w = window(bandpass(load_trial(1, 1, gesture, trial)))
        X.append(hudgins(w))
        y.append(np.full(len(w), gesture))       # label is the gesture number itself
        trial_of.append(np.full(len(w), trial))

X = np.concatenate(X)
y = np.concatenate(y)
trial_of = np.concatenate(trial_of)
print("windows:", X.shape)

train = np.isin(trial_of, TRAIN_TRIALS)
test = np.isin(trial_of, TEST_TRIALS)
assert not (train & test).any(), "a window ended up in both splits"

# StandardScaler because the four features are on wildly different scales:
# MAV is around 0.03, ZC is in the hundreds. Without it the big-numbered
# features dominate.
# shrinkage because 64 features estimated from overlapping windows makes the
# plain covariance estimate unstable.
model = make_pipeline(
    StandardScaler(),
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
)
model.fit(X[train], y[train])
pred = model.predict(X[test])

print(f"accuracy: {np.mean(pred == y[test]):.3f}   (chance is {1/17:.3f})")

# Which gestures get mistaken for which?
mistakes = {}
for true, guess in zip(y[test], pred):
    if true != guess:
        mistakes[(true, guess)] = mistakes.get((true, guess), 0) + 1

print("\nmost common mistakes:")
for (true, guess), n in sorted(mistakes.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {n:>4}  {GESTURES[true]:<34} -> {GESTURES[guess]}")
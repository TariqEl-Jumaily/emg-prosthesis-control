# This script is designed to download and prepare the EMG data from physionet.org
import ssl
import urllib.request
from pathlib import Path

import certifi

BASE = "https://physionet.org/files/grabmyo/1.1.0"

SESSION = 1
PARTICIPANT = 1

# 16 hand gestures, plus a rest condition that the dataset numbers as gesture 17.
GESTURES = range(1, 18)

# Each gesture was repeated 7 times, 5 second of recording per repetition,
# with 10 seconds of rest in between so the muscle wasn't fatigued.
TRIALS = range(1, 8)

# The python.org macOS build ships without a CA bundle, so hand it certifi's.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


# Export to the output directories:
folder = f"session{SESSION}_participant{PARTICIPANT}"
out_dir = Path("data/raw") / f"Session{SESSION}" / f"session{SESSION}_participant{PARTICIPANT}"
out_dir.mkdir(parents=True, exist_ok=True)   # parents=True makes the whole chain


for gesture in GESTURES:
    for trial in TRIALS:
        stem = f"{folder}_gesture{gesture}_trial{trial}"

        # A WFDB record is a header plus a data file, and one is useless
        # without the other, so always grab both.
        for ext in ("hea", "dat"):
            target = out_dir / f"{stem}.{ext}"
            if target.exists():          # so a re-run is instant
                continue

            url = f"{BASE}/Session{SESSION}/{folder}/{stem}.{ext}"
            with urllib.request.urlopen(url, context=SSL_CONTEXT) as response:
                target.write_bytes(response.read())

    print(f"gesture {gesture}/17")

print(f"done, {len(list(out_dir.glob('*')))} files in {out_dir}")
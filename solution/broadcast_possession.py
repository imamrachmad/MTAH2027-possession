"""Possession share from the broadcast recording, with the same model as the tactical submission.

    python solution/broadcast_possession.py

The broadcast feed is not on the submission timeline, so this is an analysis, not a submission.
The whole broadcast timeline is decoded as one block. Possession is then counted only on frames
whose homography is valid, i.e. where the camera is showing the pitch; replays, close-ups and
crowd shots carry no usable geometry. Replays that do show the pitch can still count a moment twice.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession as M  # noqa: E402
from games import asset_paths, source_frames  # noqa: E402

FPS = 25


def share(labels, mask, name):
    s = labels[mask]
    h, a, d = (s == "H").sum(), (s == "A").sum(), (s == "D").sum()
    alive = h + a
    print("{:<34} Barcelona {:5.1f}%  Athletic {:5.1f}%   in play {:5.1f} min, dead {:5.1f} min".format(
        name, 100 * h / alive, 100 * a / alive, alive / FPS / 60, d / FPS / 60))


if __name__ == "__main__":
    n = source_frames("broadcast")
    labels = M.predict(source="broadcast", blocks=[(0, n - 1)])
    valid = pd.read_csv(asset_paths(source="broadcast")["homography"])["valid"].values.astype(bool)
    players, _, ball, _ = M.load(source="broadcast")
    tracked = np.sum(~np.isnan(players[:, :, 0]), axis=1)

    share(labels, np.ones(n, bool), "all broadcast frames")
    share(labels, valid, "valid homography (pitch in view)")
    share(labels, valid & (tracked >= 10), "valid and >= 10 players tracked")

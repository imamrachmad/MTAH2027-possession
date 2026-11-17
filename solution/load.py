"""Fast loader for the tactical ATD: everything into numpy arrays, cached as .npz under data/.

    players  float32 (n_frames, n_slots, 2)   normalised 0-1, NaN when missing
    pspeed   float32 (n_frames, n_slots)      m/s as reported by the tracker
    ball     float32 (n_frames, 2)            normalised 0-1, NaN when missing
    team     int8    (n_slots,)               0 = home, 1 = away

Indexed by *video* frame (ATD frame_id - 1), so row i is submission row i.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, require, source_frames, DATA_DIR  # noqa: E402

CACHE = os.path.join(DATA_DIR, "tactical", "_atd_cache.npz")


def team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def attack_direction_first_half(meta_path):
    """{0: 'left_to_right', 1: 'right_to_left'} -- home/away direction in the first half."""
    root = ET.parse(meta_path).getroot()
    out = {}
    for i, t in enumerate(root.iter("Team")):
        for p in t.iter("ProviderParameter"):
            if p.findtext("Name") == "attack_direction_first_half":
                out[i] = p.findtext("Value")
    return out


def _parse(raw_path, n_frames):
    with open(raw_path) as fh:
        text = fh.read()
    text = text.replace(":", ",").replace(";", ",")
    arr = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32,
                      na_values=["NaN"], engine="c").to_numpy()
    del text
    frame_id = arr[:, 0].astype(np.int64)
    n_slots = (arr.shape[1] - 1 - 3) // 3
    body = arr[:, 1:1 + 3 * n_slots].reshape(len(arr), n_slots, 3)
    ballv = arr[:, 1 + 3 * n_slots:1 + 3 * n_slots + 2]

    players = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    pspeed = np.full((n_frames, n_slots), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    idx = frame_id - 1
    players[idx] = body[:, :, :2]
    pspeed[idx] = body[:, :, 2]
    ball[idx] = ballv
    return players, pspeed, ball


def load(game=GAME, rebuild=False, source="tactical"):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    team = team_of_each_track(meta_path)
    cache = os.path.join(DATA_DIR, source, "_atd_cache.npz")
    if os.path.isfile(cache) and not rebuild:
        z = np.load(cache)
        return z["players"], z["pspeed"], z["ball"], team
    players, pspeed, ball = _parse(raw_path, source_frames(source, game))
    np.savez(cache, players=players, pspeed=pspeed, ball=ball)
    return players, pspeed, ball, team


if __name__ == "__main__":
    import time
    t = time.time()
    P, S, B, T = load(rebuild=True)
    print(P.shape, S.shape, B.shape, T.shape, "in %.1fs" % (time.time() - t))
    print("ball present %.1f%%" % (100 * np.mean(~np.isnan(B[:, 0]))))
    print("home slots", int(np.sum(T == 0)), "away slots", int(np.sum(T == 1)))

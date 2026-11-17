"""Ball possession from automatic tracking: a two-stage HMM.

    python solution/possession.py                      # writes evaluation/results/<name>.csv

Stage 1  alive / dead.   Per-frame log-odds that the ball is dead, from physical cues only:
                         ball off the pitch, the ball track lost for a long stretch, and the whole
                         team walking. Decoded with Viterbi and a switching cost, so a stoppage is
                         one contiguous stretch rather than a flicker.
Stage 2  restart refine. A restart is a kick or a throw: the ball goes from stationary or unseen to
                         moving fast. Each dead->alive boundary is snapped to that onset; each
                         alive->dead boundary to the frame the ball crossed a line, when it did.
Stage 3  who has it.     Inside every alive stretch, per-frame control evidence for each team from
                         the distance of its nearest player to the ball, decoded with Viterbi and a
                         switch cost. A touch or a deflection does not pay for a switch; a received,
                         controlled ball does. A loose ball carries no evidence, so possession stays
                         with the last team that had it -- the rule in CHALLENGE.md.

No labels are used anywhere. Every constant is a physical quantity (metres, m/s, seconds) or a
penalty expressed in frames of evidence, chosen from football priors -- see TECHNICAL.md.
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
from load import load  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

FPS = 25

CONFIG = dict(
    # control
    control_sigma=1.5,       # m. A player within ~1.5 m of the ball is "on it"
    control_speed=8.0,       # m/s. Faster than this the ball is in flight, not at someone's feet
    team_weight=1.0,         # evidence per frame of clear one-team control
    switch_cost=25.0,        # frames of net control the other team needs to take possession (1.0s)
    tag_weight=0.0,          # per-frame team evidence from a Smart Tagging phase. Tried at 0.1
                             # and dropped: phase edges are only good to a few seconds and they
                             # drag possession boundaries with them. See TECHNICAL.md
    # dead ball
    walk_speed=1.6,          # m/s. Team mean speed below this reads as a stoppage
    speed_weight=2.0,        # log-odds per m/s below walk_speed
    speed_smooth=1.0,        # s, half-width of the speed smoothing window
    long_gap=2.0,            # s. A ball gap this long is not an ordinary occlusion
    gap_weight=1.0,          # log-odds while inside a long gap
    oob_margin=0.5,          # m beyond a line before the ball counts as out
    oob_weight=4.0,          # log-odds while the ball is off the pitch
    oob_min_frames=5,        # an off-pitch reading must last this long to count
    exit_horizon=2.0,        # s. How far a vanished ball's last velocity is extrapolated
    exit_weight=2.0,         # log-odds once that extrapolation has left the pitch (a guess, so
                             # weaker than seeing it out)
    set_team_speed=2.5,      # m/s. Set pieces are taken with the teams set, not sprinting
    kick_min_frames=3,       # frames of moving ball that make a kick
    restart_speed=4.0,       # m/s. A restart pass can be short and soft
    still_speed=2.0,         # m/s. Median ball speed before a restart, i.e. a placed ball
    still_before=1.0,        # s of still or unseen ball before a restart
    pending_max=60.0,        # s. Longest a stoppage is held open waiting for the restart
    pending_weight=3.0,      # log-odds between the ball going out and the restart kick
    ball_fast=6.0,           # m/s. A ball moving this fast is in play (or being restarted)
    fast_weight=1.0,         # log-odds against dead while the ball is fast
    dead_bias=-0.5,          # prior log-odds of dead
    dead_switch_cost=40.0,   # cost of entering or leaving a stoppage
    min_dead=2.0,            # s. Shorter dead stretches are dropped
    # restart refinement
    restart_back=3.0,        # s searched before a decoded restart for the kick
    restart_fwd=2.0,         # s searched after it
    stop_back=2.0,           # s searched before a decoded stop for the line crossing
)


# features

def runs(mask):
    """(starts, ends) of the True runs of a boolean array, ends exclusive."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def centred_mean(x, half):
    """Mean over a centred window of +/-half frames, ignoring NaN."""
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    cs = np.r_[0.0, np.cumsum(v)]
    cn = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(x))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(x), i + half + 1)
    n = cn[hi] - cn[lo]
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def features(players, pspeed, ball, team, pitch, cfg):
    L, W = pitch
    n = len(ball)
    bx, by = ball[:, 0] * L, ball[:, 1] * W
    present = ~np.isnan(bx)

    px, py = players[:, :, 0] * L, players[:, :, 1] * W
    d = np.hypot(px - bx[:, None], py - by[:, None])
    dh = np.nanmin(np.where(team[None] == 0, d, np.nan), axis=1)
    da = np.nanmin(np.where(team[None] == 1, d, np.nan), axis=1)
    dh = np.where(np.isnan(dh), np.inf, dh)
    da = np.where(np.isnan(da), np.inf, da)

    s2 = 2 * cfg["control_sigma"] ** 2
    ch = np.where(present, np.exp(-dh ** 2 / s2), 0.0)
    ca = np.where(present, np.exp(-da ** 2 / s2), 0.0)

    # ball speed, central difference over +/-2 frames, m/s
    k = 2
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    vx[k:-k] = (bx[2 * k:] - bx[:-2 * k]) / (2 * k / FPS)
    vy[k:-k] = (by[2 * k:] - by[:-2 * k]) / (2 * k / FPS)
    bspeed = np.hypot(vx, vy)

    # a ball in flight past a player is not under his control
    flight = np.nan_to_num(bspeed, nan=0.0)
    ctrl_w = 1.0 / (1.0 + np.maximum(0.0, flight - cfg["control_speed"]) / cfg["control_speed"])
    ch, ca = ch * ctrl_w, ca * ctrl_w

    m = cfg["oob_margin"]
    oob = present & ((bx < -m) | (bx > L + m) | (by < -m) | (by > W + m))
    # a lofted ball projected onto the ground can read as off the pitch for a frame or two
    st, en = runs(oob)
    for a, b in zip(st, en):
        if b - a < cfg["oob_min_frames"]:
            oob[a:b] = False

    # length of the ball gap each frame sits in; and, when the ball vanishes, where its last
    # seen velocity would have taken it -- out of shot often means out of play
    gap_len = np.zeros(n)
    exit_ = np.zeros(n, bool)
    st, en = runs(~present)
    horizon = int(cfg["exit_horizon"] * FPS)
    for a, b in zip(st, en):
        gap_len[a:b] = (b - a) / FPS
        if (b - a) < cfg["long_gap"] * FPS or a < 6:
            continue
        p0x, p0y = bx[a - 1], by[a - 1]
        ux = (bx[a - 1] - bx[a - 6]) / (5 / FPS)
        uy = (by[a - 1] - by[a - 6]) / (5 / FPS)
        if np.isnan(ux) or np.isnan(uy):
            continue
        for k in range(1, min(horizon, b - a)):
            x, y = p0x + ux * k / FPS, p0y + uy * k / FPS
            if x < -m or x > L + m or y < -m or y > W + m:
                exit_[a + k:b] = True
                break

    team_speed = centred_mean(np.nanmean(pspeed, axis=1), int(cfg["speed_smooth"] * FPS))
    ts0 = np.nan_to_num(team_speed, nan=0.0)

    # a restart kick: the ball inside the pitch moving for a few frames, after a spell in which it
    # was still (placed for a free kick, a goal kick, a corner) or unseen (held for a throw-in).
    # A ball dribbled or booted back onto the pitch during a stoppage is moving beforehand, so it
    # does not count.
    bs0 = np.nan_to_num(bspeed, nan=0.0)
    moving = (bs0 > cfg["restart_speed"]) & present & ~oob
    still_w = int(cfg["still_before"] * FPS)
    kick = np.zeros(n, bool)
    st, en = runs(moving)
    for a, b in zip(st, en):
        if b - a < cfg["kick_min_frames"] or a < still_w + 5:
            continue
        if ts0[a] > cfg["set_team_speed"]:
            continue
        before = bspeed[a - still_w - 5:a - 5]
        if np.median(np.nan_to_num(before, nan=0.0)) < cfg["still_speed"]:
            kick[a] = True

    # once the ball has gone out, it stays dead until somebody restarts it
    pending = np.zeros(n, bool)
    out_st, out_en = runs(oob)
    kicks = np.flatnonzero(kick)
    cap = int(cfg["pending_max"] * FPS)
    for a, b in zip(out_st, out_en):
        j = np.searchsorted(kicks, b)
        stop = kicks[j] if j < len(kicks) else n
        pending[b:min(stop, b + cap)] = True

    return dict(present=present, bx=bx, by=by, dh=dh, da=da, ch=ch, ca=ca, bspeed=bspeed,
                oob=oob, exit=exit_, gap_len=gap_len, team_speed=team_speed, kick=kick,
                pending=pending)


def dead_logodds(f, cfg):
    z = np.full(len(f["bx"]), cfg["dead_bias"])
    v = np.nan_to_num(f["team_speed"], nan=cfg["walk_speed"])
    z += np.clip(cfg["speed_weight"] * (cfg["walk_speed"] - v), -3, 3)
    z += cfg["gap_weight"] * (f["gap_len"] >= cfg["long_gap"])
    z += cfg["oob_weight"] * f["oob"]
    z += cfg["exit_weight"] * f["exit"]
    z += cfg["pending_weight"] * f["pending"]
    fast = np.nan_to_num(f["bspeed"], nan=0) > cfg["ball_fast"]
    z -= cfg["fast_weight"] * (fast & ~f["oob"] & ~f["pending"])
    return z


# decoding

def viterbi2(score0, score1, cost):
    """Best 0/1 path maximising sum(score of chosen state) - cost * switches."""
    n = len(score0)
    back = np.zeros((n, 2), dtype=np.bool_)  # True = came from the other state
    v0, v1 = float(score0[0]), float(score1[0])
    s0 = score0.tolist(); s1 = score1.tolist()
    for t in range(1, n):
        a0, b0 = v0, v1 - cost
        a1, b1 = v1, v0 - cost
        if a0 >= b0:
            n0 = a0 + s0[t]
        else:
            n0 = b0 + s0[t]; back[t, 0] = True
        if a1 >= b1:
            n1 = a1 + s1[t]
        else:
            n1 = b1 + s1[t]; back[t, 1] = True
        v0, v1 = n0, n1
    path = np.zeros(n, dtype=np.int8)
    state = 0 if v0 >= v1 else 1
    for t in range(n - 1, -1, -1):
        path[t] = state
        if back[t, state]:
            state = 1 - state
    return path


def alive_dead(f, cfg, lo, hi):
    """Boolean dead mask over [lo, hi)."""
    z = dead_logodds(f, cfg)[lo:hi]
    dead = viterbi2(np.zeros_like(z), z, cfg["dead_switch_cost"]).astype(bool)
    return dead


def refine(dead, f, cfg, lo):
    """Snap stoppage edges to physical events: line crossings and restart kicks."""
    dead = dead.copy()
    n = len(dead)
    kick = f["kick"][lo:lo + n]
    oob = f["oob"][lo:lo + n]
    st, en = runs(dead)
    for a, b in zip(st, en):
        # stop: the first off-pitch frame shortly before the decoded stop, or the moment the
        # ball track was lost if the ball vanished into the stoppage
        w0 = max(0, a - int(cfg["stop_back"] * FPS))
        hit = np.flatnonzero(oob[w0:a + FPS])
        if len(hit):
            new_a = w0 + hit[0]
            dead[new_a:a] = True
            if new_a > a:
                dead[a:new_a] = False
        # restart: the restart kick nearest the decoded restart
        if b >= n:
            continue
        w0 = max(a + 1, b - int(cfg["restart_back"] * FPS))
        w1 = min(n, b + int(cfg["restart_fwd"] * FPS))
        onsets = np.flatnonzero(kick[w0:w1])
        if len(onsets):
            # the restart is the first kick from a still ball; later ones are play
            k = w0 + onsets[0]
            # a couple of frames back: the speed estimate is centred over +/-2 frames
            k = max(a + 1, k - 2)
            if k < b:
                dead[k:b] = False
            else:
                dead[b:k] = True
    # drop stoppages too short to be real
    st, en = runs(dead)
    for a, b in zip(st, en):
        if (b - a) < cfg["min_dead"] * FPS:
            dead[a:b] = False
    return dead


# Smart Tagging phases that say which side has the ball. "Attacking" codes name the team in
# possession; "defending" codes name the team without it. Set pieces name the restarting team.
ATTACK_CODES = {"build up", "progression", "ball in final third", "ball in the box",
                "players in the box", "creating chances", "attacking transition", "long ball",
                "set pieces"}
DEFEND_CODES = {"defending in defensive third", "defending in middle third",
                "defending in attacking third", "defensive transition"}
TEAM_NAMES = {"barcelona": 0, "atletic club": 1}


def tag_prior(n, game=GAME):
    """+1 where Smart Tagging says home has the ball, -1 away, 0 unknown or contradictory."""
    try:
        from load_smart_tagging import load as load_tags
        home = np.zeros(n); away = np.zeros(n)
        for e in load_tags(game):
            side = TEAM_NAMES.get((e["tags"].get("Team") or "").strip().lower())
            code = (e["code"] or "").strip().lower()
            if side is None or (code not in ATTACK_CODES and code not in DEFEND_CODES):
                continue
            if code in DEFEND_CODES:
                side = 1 - side
            a, b = max(0, e["start_frame"]), min(n, e["end_frame"])
            (home if side == 0 else away)[a:b] = 1
        return home - away
    except (IOError, OSError, FileNotFoundError):
        return np.zeros(n)


def possession(f, cfg, lo, hi, dead):
    """Per-frame H/A over [lo, hi), decoded independently inside each alive stretch."""
    ev = cfg["team_weight"] * (f["ch"][lo:hi] - f["ca"][lo:hi])
    ev = ev + cfg["tag_weight"] * f["tags"][lo:hi]
    out = np.full(hi - lo, "D", dtype="<U1")
    st, en = runs(~dead)
    for a, b in zip(st, en):
        path = viterbi2(ev[a:b], -ev[a:b], cfg["switch_cost"])
        out[a:b] = np.where(path == 0, "H", "A")
    return out


def predict(cfg=CONFIG, game=GAME, source="tactical", blocks=None):
    """Labels for every frame of a source. `blocks` = [(first, last)] frame spans to decode;
    defaults to the two graded halves of the tactical video."""
    players, pspeed, ball, team = load(game, source=source)
    f = features(players, pspeed, ball, team, GAMES[game]["pitch"], cfg)
    # Smart Tagging is timed to the tactical video only
    f["tags"] = tag_prior(len(ball), game) if source == "tactical" else np.zeros(len(ball))
    labels = np.full(source_frames(source, game), "X", dtype="<U1")
    if blocks is None:
        blocks = [(first, last) for _, first, last in half_video_blocks(game)]
    for first, last in blocks:
        lo, hi = first, last + 1
        dead = alive_dead(f, cfg, lo, hi)
        dead = refine(dead, f, cfg, lo)
        labels[lo:hi] = possession(f, cfg, lo, hi, dead)
    return labels


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="possession_hmm")
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = ap.parse_args()
    labels = predict()
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    graded = labels[labels != "X"]
    print("wrote", path)
    print("graded frames: H %d  A %d  D %d" % tuple(np.sum(graded == c) for c in "HAD"))

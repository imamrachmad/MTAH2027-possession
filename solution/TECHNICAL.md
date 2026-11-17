# Ball possession from automatic tracking — a two-stage HMM

## 1. Approach

The ranking metric scores whole possessions, so the system is built to produce **segments**, not
per-frame guesses. Each stage is a two-state Viterbi decode: per-frame evidence on one side and a
fixed cost per switch on the other. The cost is what stops one noisy frame from opening or closing a
possession or a stoppage. Every constant is a physical quantity (metres, m/s, seconds), or a cost
counted in frames of evidence.

**Stage 1 — alive / dead.** Per-frame log-odds that the ball is dead:

| cue | why | weight |
|---|---|---|
| ball observed off the pitch (> 0.5 m past a line, ≥ 5 frames) | the definition of out of play | +4 |
| ball vanished and its last velocity leaves the pitch within 2 s | the ball usually goes out of shot as it goes out of play | +2 |
| ball went out and **no restart since** | it stays dead until someone restarts it | +3 |
| inside a ball-track gap ≥ 2 s | in this tracking, gaps are either < 1 s or ≥ 4 s, and long ones are mostly stoppages | +1 |
| team mean speed below 1.6 m/s (smoothed ±1 s) | in a stoppage everyone walks | +2 per m/s, clipped ±3 |
| ball faster than 6 m/s on the pitch, nothing pending | a moving ball is in play | −1 |

Switching cost: 40.

**Restart detection.** A *restart kick* is when the ball starts moving on the pitch (> 4 m/s for
≥ 3 frames) after about 1 s of a **still or unseen** ball (median speed < 2 m/s: placed for a
free kick, goal kick or corner, or held for a throw-in), with the team mean speed under 2.5 m/s
(set pieces are taken with the teams set). A ball dribbled or booted back during a stoppage fails
one of these tests. A restart kick closes the "pending" state above.

**Stage 2 — boundary refinement.** The start of each decoded stoppage is moved to the first frame
the ball is seen off the pitch, when that happens near the decoded start. The end is moved to the
**first** restart kick in a window from 3 s before to 2 s after. Stoppages shorter than 2 s are
dropped.

**Stage 3 — who has it.** Per-frame control evidence for each team:
`c = exp(−d²/2σ²)`, where `d` is that team's nearest tracked player to the ball and σ = 1.5 m. It is
down-weighted when the ball moves faster than 8 m/s, because a ball flying past a player is not
under control. The evidence is `c_home − c_away`, and a switch costs 25 frames of net evidence.
Each alive stretch is decoded on its own, so the restarting team is simply whoever controls the
ball first.

## 2. Possession rules as implemented

- **Loose or contested ball, or a ball in flight:** no evidence, so possession stays with the last
  team that had it (CHALLENGE.md).
- **Touch, deflection or aerial duel without control:** a few frames of evidence don't cover the
  switch cost, so possession doesn't change.
- **Controlled reception or interception:** about one second of clear control flips
  possession. The flip lands at the frame the evidence turns.
- **Dead ball:** from the line crossing (seen or extrapolated), or from a team-wide slowdown for
  whistles, until the first restart kick. Always `D`, never owned.
- **No `X` on graded frames.** Under spellF1 an `X` repeats the previous label anyway.

## 3. Assets used

| asset | use |
|---|---|
| tactical ATD (players, ball, slot teams) | everything above |
| Smart Tagging | **tried and dropped** as a weak team prior (0.1 frames of evidence per tagged frame). Phase edges are only good to a few seconds and they dragged possession boundaries. The code is kept, with `tag_weight = 0` |
| homography, videos, broadcast feed | not used. The tracking already lives in pitch coordinates |

## 4. Training data

None. No model is trained, and no human labels on this game or any other were used.

## 5. Use of the public ground truth

Used as the rules allow: to check frame alignment and to **look at** where the system went wrong
segment by segment, which led to the structural fixes above (restart = first kick from a still
ball; extrapolated exits weaker than observed ones). Constants were not grid-searched against it.
One decision was informed by it: dropping the Smart Tagging prior. With 17 possessions the
bootstrap noise band on spellF1 is ±0.33, so the public window can catch a broken pipeline but
cannot rank two variants.

Public window: macroF1 0.856, spellF1 0.491 (baseline_smooth50: 0.621 / 0.326). Full-match
structure, with no labels involved: 104 stoppages (median 21 s), 289 possessions (median
7.5 s), 47% of graded time dead. Those numbers are within football norms.

## 6. Limitations and next steps

- **Untracked players.** When the player on the ball is not tracked, the nearest *tracked* player
  takes the credit. Short possessions near the touchline suffer most.
- **Whistles without the ball going out** (fouls, offsides) are found only through the team
  slowing down, which lags the whistle by 1–2 s.
- **Restart timing** comes from a ball-speed onset. A soft restart pass (< 4 m/s) is missed and
  falls back to the HMM edge.
- With more time: fit the per-frame emissions on public tracking data from **other** games;
  recover the ball inside long gaps from player motion; map the broadcast feed to tactical frames
  and use its close-ups for the occluded ball.

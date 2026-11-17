# Ball possession from automatic tracking — More than a Hack 2027, Phase 1

A two-stage HMM that labels every frame of the tactical video `H` / `A` / `D`. The first stage
decides whether the ball is alive or dead; the second decides which team controls it. It uses
only the tactical tracking (ATD), with no training and no labels. The method is described in
[solution/TECHNICAL.md](solution/TECHNICAL.md).

## Reproduce the submission

Requires Python 3.8+ (tested on 3.14.3).

```bash
pip install -r requirements.txt
# Unpack the match bundle into ./data (see data/README.md), or set MTAH_DATA_DIR to point at it
python solution/possession.py --name <team_name>
# -> evaluation/results/<team_name>.csv, 199,319 rows
```

It runs in about 30 seconds and is fully deterministic. The first run caches the parsed tracking
under `data/` (git-ignored); deleting the cache reproduces the identical file. SHA-256 of the
output on the provided bundle:
`C31A27DC7F26BC74D97DC68E1161B7EAE86EF670A9062115DB46924D10BD910F`.

To validate and score the file, copy it into a checkout of the
[official challenge repository](https://github.com/metrica-sports/more-than-a-hack-2027) and run
`python evaluation/validate_submission.py <file>` and `python evaluation/run_evaluation.py`.

## Layout

| file | |
|---|---|
| `solution/load.py` | parses the raw ATD into numpy arrays |
| `solution/possession.py` | features, the alive/dead HMM, restart refinement, the possession HMM, and the CSV writer |
| `solution/broadcast_possession.py` | the same model on the broadcast feed, for possession-share analysis (not a submission) |
| `solution/TECHNICAL.md` | the technical document |
| `games.py`, `starter/load_smart_tagging.py` | unmodified helpers from the official repository (MIT, see `LICENSE-metrica`) |

## Data

The match data is **not** in this repository and must not be added to it: it is licensed for the
competition only. `.gitignore` excludes everything under `data/` apart from its README.

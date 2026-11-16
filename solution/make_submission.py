"""Generate the possession file for More than a Hack 2027, Phase 1 (BAR-ATH, tactical timeline).

Run from the repository root, with the match data unpacked in data/ (or MTAH_DATA_DIR set):

    python solution/make_submission.py

Writes submission/possession.csv: one row per tactical video frame (199,319 rows), labels H / A / D / X.

Nothing here reads the public ground truth and nothing is fitted to labels. Every constant is
either HAND-SET (chosen by reasoning) or LABEL-FREE (found from the tracking alone, no labels).

Pipeline:
  1. load the tracking, convert to metres
  2. per-frame features: nearest-player distances to the ball, speed of the 4 players nearest the ball
  3. Smart Tagging vote (team in possession implied by a few phase tags)
  4. dead-ball speed thresholds from a two-component mixture over the whole match (LABEL-FREE)
  5. per-frame costs for H / A / D, then Viterbi per half with a switch cost
  6. dead-ball runs shorter than 3 s are handed back to the previous possession
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from kloppy import sportscode
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
from starter.load_atd import load as load_atd  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices are expected here

OUT = os.path.join(ROOT, "submission", "possession.csv")
FPS = 25

# %% HAND-SET constants (nothing below is fitted to labels)
K_GAP = 0.4                  # log-odds per metre of nearest-player gap (clipped at 15 m); 5 m gap ~ 88% confident
TAG_W = 2.0                  # weight of a strong Smart Tagging vote (those tags agree ~88-91% with confident ball frames)
A_NEAR, A_PLAY = 3.0, 2.0    # slope of the dead-ball evidence around its threshold
SWITCH = 1.0                 # cost of changing state: a confident handover flips in about a second
LAM = 1 / FPS                # neighbouring frames are near-copies, so scale per-frame evidence by 1/25
MIN_D = 75                   # a dead-ball run shorter than 3 s is handed back to the previous possession

# %% 1. Load the tracking and convert to metres
ds = load_atd()
df = ds.to_df()
PL, PW = GAMES[GAME]["pitch"]
df["video_frame"] = df["frame_id"] - 1                       # submission row = video frame, counting from 0
x_cols = [c for c in df.columns if c.endswith("_x")]
y_cols = [c for c in df.columns if c.endswith("_y")]
df[x_cols] = df[x_cols] * PL
df[y_cols] = df[y_cols] * PW
slots = {str(t.ground): [p.player_id for p in t.players] for t in ds.metadata.teams}
F = df.set_index("video_frame")
assert len(F) == 199319 and F.index[0] == 0 and F.index[-1] == 199318

# %% 2. Per-frame features
def arr(ids, suffix):
    return F[[f"{i}_{suffix}" for i in ids]].to_numpy()

bx = F["ball_x"].to_numpy()[:, None]
by = F["ball_y"].to_numpy()[:, None]

def min_dist(ids):
    dd = np.hypot(arr(ids, "x") - bx, arr(ids, "y") - by)
    return np.nanmin(np.where(np.isnan(dd), np.inf, dd), axis=1)

dh, da = min_dist(slots["home"]), min_dist(slots["away"])
dh[np.isinf(dh)] = np.nan
da[np.isinf(da)] = np.nan

ids_all = slots["home"] + slots["away"]
px, py, ps = arr(ids_all, "x"), arr(ids_all, "y"), arr(ids_all, "s")
d = np.hypot(px - bx, py - by)
d = np.where(np.isnan(d), np.inf, d)
nearest = np.argsort(d, axis=1)[:, :4]                       # the 4 players nearest the ball
near_speed = np.nanmean(np.take_along_axis(ps, nearest, axis=1), axis=1)

feat = pd.DataFrame(index=F.index)
feat["ball_seen"] = ~np.isnan(bx[:, 0])
feat["d_home"], feat["d_away"] = dh, da
feat["gap"] = da - dh                                        # positive: a home player is closer to the ball
feat["near_speed"] = np.where(feat["ball_seen"], near_speed, np.nan)
feat["player_speed"] = np.nanmean(ps, axis=1)                # fallback dead-ball clue when the ball is missing
feat["near_s"] = feat["near_speed"].rolling(75, center=True, min_periods=15).median()     # 3 s window
feat["play_s"] = feat["player_speed"].rolling(75, center=True, min_periods=15).median()
gi = np.concatenate([np.arange(a, b + 1) for _, a, b in half_video_blocks(GAME)])          # graded frames

# %% 3. Smart Tagging vote: +1 home in possession, -1 away, 0 no strong tag
ev = sportscode.load(asset_paths()["smart_tagging"]).to_df()
ev["f0"] = (ev["timestamp"].dt.total_seconds() * FPS).round().astype(int)
ev["f1"] = (ev["end_timestamp"].dt.total_seconds() * FPS).round().astype(int)

STRONG = {"PROGRESSION": 1, "BUILD UP": 1, "LONG BALL": 1,                       # tag names the possessing team
          "DEFENDING IN MIDDLE THIRD": -1, "DEFENDING IN ATTACKING THIRD": -1}   # tag names the defending team
SIDE = {"Barcelona": 1, "Atletic Club": -1}                                      # +1 means home

votes = np.zeros(len(feat))
count = np.zeros(len(feat))
for _, r in ev.iterrows():
    code = str(r["code"]).upper()
    if code in STRONG and r["Team"] in SIDE:
        votes[r["f0"]:r["f1"] + 1] += STRONG[code] * SIDE[r["Team"]]
        count[r["f0"]:r["f1"] + 1] += 1
feat["tag_vote"] = np.divide(votes, count, out=np.zeros_like(votes), where=count > 0)

# %% 4. Dead-ball thresholds, LABEL-FREE: where a two-component mixture over log speed crosses 50%
def gmm_threshold(s):
    v = s.loc[gi].dropna().to_numpy()
    g2 = GaussianMixture(2, random_state=0).fit(np.log(v + 0.05).reshape(-1, 1))
    slow = g2.means_.ravel().argmin()
    grid = np.linspace(v.min() + 0.01, np.percentile(v, 99), 600)
    post = g2.predict_proba(np.log(grid + 0.05).reshape(-1, 1))[:, slow]
    thr = grid[np.argmin(np.abs(post - 0.5))]
    print("slow mode ~", round(float(np.exp(g2.means_.ravel()[slow]) - 0.05), 2), "m/s | share",
          g2.weights_[slow].round(2), "| threshold", round(thr, 2))
    return thr

THR_NEAR = gmm_threshold(feat["near_s"])
THR_PLAY = gmm_threshold(feat["play_s"])

# %% 5. Per-frame costs (lower = more plausible) for H, A, D
def softplus(x):
    return np.logaddexp(0, x)

dmin = feat[["d_home", "d_away"]].min(axis=1)                # nearest player of either team
gate = ((8 - dmin) / 6).clip(0.25, 1.0).fillna(1.0)          # evidence weakens when nobody is near the ball

L = K_GAP * feat["gap"].clip(-15, 15).fillna(0.0) * gate + TAG_W * feat["tag_vote"]     # log-odds of H over A
LD = np.where(feat["near_s"].notna(), A_NEAR * (THR_NEAR - feat["near_s"]),
                                       A_PLAY * (THR_PLAY - feat["play_s"]))
LD = np.nan_to_num(LD, nan=0.0)                              # log-odds of dead ball

C = pd.DataFrame({"H": softplus(LD) + softplus(-L),
                  "A": softplus(LD) + softplus(L),
                  "D": softplus(-LD)}, index=feat.index)

# %% 6. Viterbi per half
def viterbi(cost_per_frame, S):
    T = len(cost_per_frame)
    back = np.zeros((T, 3), dtype=np.int8)
    cost = cost_per_frame[0].copy()
    for t in range(1, T):
        tot = cost[:, None] + S                              # tot[i, j]: was in i, move to j
        back[t] = tot.argmin(axis=0)
        cost = tot.min(axis=0) + cost_per_frame[t]
    path = np.zeros(T, dtype=np.int8)
    path[-1] = cost.argmin()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path

def decode(C, switch_cost):
    S = np.full((3, 3), float(switch_cost))
    np.fill_diagonal(S, 0)
    out = []
    for _, a, b in half_video_blocks(GAME):
        path = viterbi(LAM * C.loc[a:b].to_numpy(), S)
        out.append(pd.Series(np.array(["H", "A", "D"])[path], index=range(a, b + 1)))
    return out

def drop_short_d(lab, min_len=MIN_D):
    lab = lab.copy()
    idx = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    ends = np.r_[idx[1:], len(lab)]
    for s, e in zip(idx, ends):
        if lab[s] == "D" and e - s < min_len and s > 0:
            lab[s:e] = lab[s - 1]                            # give the frames to the previous possession
    return lab

parts = [pd.Series(drop_short_d(s.to_numpy()), index=s.index) for s in decode(C, SWITCH)]

# %% 7. Write the file (frames outside the two halves are never scored: X)
labels = np.array(["X"] * len(feat), dtype=object)
for s in parts:
    labels[s.index[0]:s.index[-1] + 1] = s.to_numpy()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame({"frame": feat.index, "label": labels}).to_csv(OUT, index=False)

# %% 8. Label-free summary of what was written
def runs_of(lab):                                            # possessions = maximal runs of H or A
    start = np.r_[True, lab[1:] != lab[:-1]]
    sizes = np.bincount(np.cumsum(start))[1:]
    return [n for l, n in zip(lab[start], sizes) if l in ("H", "A")]

poss = [n for s in parts for n in runs_of(s.to_numpy())]
share = pd.Series(labels[gi]).value_counts(normalize=True).round(3).to_dict()
print("wrote", OUT)
print("label share over graded frames:", share)
print(len(poss), "possessions, mean", round(float(np.mean(poss)) / FPS, 1), "s")

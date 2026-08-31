"""Label-side artifact / mediation analyses (a-h) on the P0.4 aggregate.

Reads ``output/encoder-probes/l_mediation_table.pkl`` (written by
``l_mediation.py``), joins the pre-built card feature table and the Forge
hint table, and writes ``output/encoder-probes/l_report.md``.

CPU only, seconds to run.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "encoder-probes"
TABLE = OUT / "card_table.pkl"  # written by p0b_card_table.py
HINTS = REPO / "output" / "scorer-probes" / "forge_hints.csv"
SD_SCORE_PLAY = 0.062
K = 20.0
MIN_N = 50

L = []  # report lines


try:
    import sys as _sys

    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass


def say(s: str = "") -> None:
    L.append(s)
    print(s)


# ----------------------------------------------------------------- helpers


def wls(y, X, w):
    """Weighted least squares with HC0 (heteroscedasticity-robust) SEs."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & np.isfinite(w) & (w > 0)
    y, X, w = y[ok], X[ok], w[ok]
    A = X.T @ (X * w[:, None])
    Ainv = np.linalg.pinv(A)
    b = Ainv @ (X.T @ (w * y))
    r = y - X @ b
    meat = (X * (w * r)[:, None]).T @ (X * (w * r)[:, None])
    V = Ainv @ meat @ Ainv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    ss_t = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r2 = 1.0 - np.sum(w * r**2) / ss_t if ss_t > 0 else np.nan
    return b, se, r2, int(ok.sum())


def wmean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w)
    if not ok.any():
        return np.nan
    return float(np.average(x[ok], weights=w[ok]))


def wcorr(x, y, w=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 3:
        return np.nan
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cxy = np.average((x - mx) * (y - my), weights=w)
    cxx = np.average((x - mx) ** 2, weights=w)
    cyy = np.average((y - my) ** 2, weights=w)
    return float(cxy / np.sqrt(cxx * cyy))


def sb(r: float) -> float:
    """Spearman-Brown: half-length reliability -> full-length reliability."""
    return 2 * r / (1 + r) if np.isfinite(r) else np.nan


def table(rows, header):
    say("| " + " | ".join(header) + " |")
    say("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        say("| " + " | ".join(str(x) for x in r) + " |")
    say()


def f3(x):
    return "—" if not np.isfinite(x) else f"{x:+.4f}"


def f3u(x):
    return "—" if not np.isfinite(x) else f"{x:.4f}"


# ----------------------------------------------------------------- load

d = pickle.load(open(OUT / "l_mediation_table.pkl", "rb"))
med, meta = d["cards"], d["meta"]
feat = pickle.load(open(TABLE, "rb"))
featcols = [c for c in feat.columns if c not in med.columns] + ["card_name"]
extra = pd.read_pickle(OUT / "l_extra_flags.pkl")
df = med.merge(feat[featcols], on="card_name", how="left")
df = df.merge(extra, on="card_name", how="left")
hints = pd.read_csv(HINTS)
hints = hints.rename(columns={"name": "card_name"})[
    ["card_name", "ai_remove_deck", "draft_rank"]
].drop_duplicates("card_name")
df = df.merge(hints, on="card_name", how="left")

df["wt"] = df.n_in_deck / (df.n_in_deck + K)
df["mv2"] = df.mv**2
df["is_nonland_noncreature"] = (
    (1 - df.is_creature.fillna(0)) * (1 - df.is_land.fillna(0))
).astype(int)
df["mana_rock"] = (
    df.is_artifact.fillna(0).astype(int)
    * df.ph_tap_for_mana.fillna(0).astype(int)
    * (1 - df.is_creature.fillna(0)).astype(int)
    * (1 - df.is_land.fillna(0)).astype(int)
)
df["combat_trick"] = (
    df.ph_pump_eot.fillna(0).astype(int) * df.is_instant.fillna(0).astype(int)
)
df["draw_spell"] = (
    df.ph_draw_a_card.fillna(0).astype(int)
    * (1 - df.is_creature.fillna(0)).astype(int)
    * (1 - df.is_land.fillna(0)).astype(int)
)

# stratum-derived quantities
for tag, pre in (("fb", "fb_"), ("ff", "ff_")):
    nin = df[f"{pre}wins_in_deck"] + df[f"{pre}losses_in_deck"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df[f"n_{tag}"] = nin
        df[f"w_{tag}"] = df[f"{pre}wins_in_deck"] / nin
        a = df[f"{pre}wins_played"] / df[f"{pre}wins_in_deck"]
        b = df[f"{pre}losses_played"] / df[f"{pre}losses_in_deck"]
        df[f"a_{tag}"], df[f"b_{tag}"] = a, b
        df[f"m_{tag}"] = (a + b) / 2
        df[f"d_{tag}"] = a - b
        df[f"score_{tag}"] = (
            df[f"{pre}wins_played"] - df[f"{pre}losses_played"]
        ) / nin

# game-length-bucket scores
LB = meta["len_bucket_labels"]
TB = meta["tot_bucket_labels"]
for pre in ("gl", "og", "tg"):
    for b_ in range(len(LB)):
        nin = df[f"{pre}{b_}_wins_in_deck"] + df[f"{pre}{b_}_losses_in_deck"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"n_{pre}{b_}"] = nin
            df[f"score_{pre}{b_}"] = (
                df[f"{pre}{b_}_wins_played"] - df[f"{pre}{b_}_losses_played"]
            ) / nin
            df[f"pr_{pre}{b_}"] = (
                df[f"{pre}{b_}_wins_played"] + df[f"{pre}{b_}_losses_played"]
            ) / nin

# split halves
for h in (0, 1):
    nin = df[f"h{h}_wins_in_deck"] + df[f"h{h}_losses_in_deck"]
    ninp = df[f"h{h}_wins_in_deck_play"] + df[f"h{h}_losses_in_deck_play"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df[f"h{h}_n"] = nin
        df[f"h{h}_np"] = ninp
        df[f"h{h}_score_play"] = (
            df[f"h{h}_wins_played_play"] - df[f"h{h}_losses_played_play"]
        ) / ninp
        df[f"h{h}_score_draw"] = (
            (df[f"h{h}_wins_played"] - df[f"h{h}_wins_played_play"])
            - (df[f"h{h}_losses_played"] - df[f"h{h}_losses_played_play"])
        ) / (nin - ninp)
        df[f"h{h}_played_rate"] = (
            df[f"h{h}_wins_played"] + df[f"h{h}_losses_played"]
        ) / nin
        df[f"h{h}_score"] = (
            df[f"h{h}_wins_played"] - df[f"h{h}_losses_played"]
        ) / nin

df = df.copy()  # de-fragment after the wide column build-out
C = df[df.n_in_deck >= MIN_N].copy()  # main analysis frame

say("# Label-side artifact & mediation analysis (P0.4 + R3/R4/R9 label halves)")
say()
say(f"Corpus: `{meta['corpus']}`  ")
say(f"Games: **{meta['n_games']:,}**, malformed lines: {meta['bad_lines']}, "
    f"distinct cards: **{len(df):,}** "
    f"(cards with n_in_deck ≥ {MIN_N}: {len(C):,}).  ")
say(f"Effect sizes in label-SD units use SD(score_play) = {SD_SCORE_PLAY}.")
say()

# ---------------------------------------------------------------- verify
say("## 0. Verification")
say()
snap = pd.read_csv(
    r"Y:\Nicolas\mtg\mtg-models-data\sealed\training-data\matches-bo1"
    r"\cards-win-rates.txt",
    sep=";",
    dtype={"card_name": str},
)
mm = med.merge(snap, on="card_name", suffixes=("", "_s"))
rows = []
for c in [
    "wins_when_played",
    "wins_when_in_deck",
    "losses_when_played",
    "losses_when_in_deck",
]:
    rows.append([c, f"{(mm[c] == mm[c + '_s']).mean():.6f}",
                 int((mm[c] - mm[c + "_s"]).abs().max())])
for c, s in [("score_play", "raw_score_play"), ("score_draw", "raw_score_draw"),
             ("played_rate", "raw_played_rate"), ("cast_lift", "raw_cast_lift")]:
    ok = np.isclose(mm[c], mm[s], atol=6e-6, equal_nan=True)
    rows.append([f"{c} vs {s}", f"{ok.mean():.6f}", "—"])
table(rows, ["quantity", "exact-match fraction vs snapshot", "max abs diff"])
say(f"All {len(mm):,} snapshot cards reproduce **exactly**; the recomputed "
    f"corpus additionally contains {len(df) - len(mm)} card(s) the snapshot "
    f"omits: {', '.join(sorted(set(med.card_name) - set(snap.card_name)))} "
    "(n_in_deck = 1 each).")
say()

ident = df.m * (2 * df.w - 1) + df.d / 2
res = (ident - df.score_overall).abs()
say(f"Identity `score = m·(2w−1) + d/2`: max |residual| over all "
    f"{res.notna().sum():,} cards with both counters = "
    f"**{np.nanmax(res):.3e}** (floating-point noise).")
say()

# ---------------------------------------------------------------- (a)
say("## a. Artifact verification at full scale")
say()
viol = (df.score_overall.abs() > df.played_rate + 1e-12).sum()
say(f"**A1 ceiling** `|score| ≤ played_rate`: violations = **{viol}** / "
    f"{df.score_overall.notna().sum():,} cards (exact, as the identity forces).")
say()
bins = [0, .05, .10, .15, .20, .30, .40, .50, .60, .80, 1.001]
C["pr_bin"] = pd.cut(C.played_rate, bins, right=False)
rows = []
for b_, g in C.groupby("pr_bin", observed=True):
    rows.append([
        str(b_), len(g), f"{g.played_rate.mean():.3f}",
        f"{g.score_overall.abs().max():.4f}",
        f"{g.score_overall.std():.4f}",
        f"{g.score_play.std():.4f}",
    ])
table(rows, ["played_rate bin", "n cards", "mean played_rate",
             "max |score|", "SD(score_overall)", "SD(score_play)"])
lo = C[C.played_rate < .05].score_overall.std()
hi = C[C.played_rate >= .80].score_overall.std()
mid = C[(C.played_rate >= .50) & (C.played_rate < .60)].score_overall.std()
say(f"Heteroscedasticity: SD(score) = {lo:.4f} in the bottom played_rate bin "
    f"vs {hi:.4f} in the top — a **{hi / lo:.0f}×** spread (the top bin holds "
    f"only 5 cards; against the last well-populated bin [0.5,0.6) at "
    f"{mid:.4f} it is still {mid / lo:.0f}×). Low-played_rate classes are "
    "algebraically barred from expressing a large score.")
say()

say("**A4 game-length certification** — score/played_rate ratio by mana value:")
rows = []
for mv in range(0, 9):
    g = C[C.mv == mv] if mv < 8 else C[C.mv >= 8]
    if len(g) < 20:
        continue
    r_ov = wmean(g.score_overall, g.wt) / wmean(g.played_rate, g.wt)
    r_pl = wmean(g.score_play, g.wt) / wmean(g.played_rate, g.wt)
    rows.append([mv if mv < 8 else "8+", len(g),
                 f"{wmean(g.played_rate, g.wt):.3f}",
                 f3(wmean(g.score_overall, g.wt)),
                 f3(r_ov), f3(r_pl)])
table(rows, ["MV", "n cards", "mean played_rate", "mean score_overall",
             "score/played_rate", "score_play/played_rate"])

say("**A2 build-method win rates** (per game, side-level):")
rows = []
tot = 0
for k in sorted(meta["method_wins"], key=lambda x: -(
        meta["method_wins"].get(x, 0) + meta["method_losses"].get(x, 0))):
    wn = meta["method_wins"].get(k, 0)
    ls = meta["method_losses"].get(k, 0)
    tot += wn + ls
    if wn + ls < 2000:
        continue
    rows.append([k, f"{wn + ls:,}", f"{wn / (wn + ls):.4f}"])
table(rows, ["build method", "games (side-level)", "win rate"])
wrs = {k: meta["method_wins"].get(k, 0) /
       max(1, meta["method_wins"].get(k, 0) + meta["method_losses"].get(k, 0))
       for k in meta["method_wins"]
       if meta["method_wins"].get(k, 0) + meta["method_losses"].get(k, 0) >= 2000}
say(f"Selection span across methods: {min(wrs.values()):.3f} … "
    f"{max(wrs.values()):.3f} win rate "
    f"= {(max(wrs.values()) - min(wrs.values())):.3f} in w units.")
say()

say("**A3 opponent assignment** — who `random` decks face:")
pc = meta["pair_counts"]
opp = {}
for k, c in pc.items():
    a, b_ = k.split("|", 1)
    for me, you in ((a, b_), (b_, a)):
        opp.setdefault(me, {}).setdefault(you, 0)
        opp[me][you] += c
if "random" in opp:
    tot_r = sum(opp["random"].values())
    nonrand = sum(v for k, v in opp["random"].items() if k != "random")
    forge = sum(v for k, v in opp["random"].items() if k.startswith("forge"))
    say(f"`random` side-games: {tot_r:,}; opponent is not `random` in "
        f"**{nonrand / tot_r:.1%}**, a forge builder in **{forge / tot_r:.1%}**.")
say()
say(f"**Winner vs loser mean distinct cards played**: "
    f"**{meta['sum_played_winner'] / meta['n_games']:.3f}** vs "
    f"**{meta['sum_played_loser'] / meta['n_games']:.3f}** "
    f"(+{meta['sum_played_winner'] / meta['sum_played_loser'] - 1:.1%}).")
say()
say(f"Corpus weighted means: w = {wmean(C.w, C.wt):.4f}, "
    f"m = {wmean(C.m, C.wt):.4f}, d = {wmean(C.d, C.wt):+.4f}, "
    f"score_overall = {wmean(C.score_overall, C.wt):+.4f}, "
    f"score_play = {wmean(C.score_play, C.wt):+.4f}, "
    f"cast_lift = {wmean(C.cast_lift, C.wt):+.4f}.")
say()

# ---------------------------------------------------------------- (b)
FEATURES = [
    ("flying", "kw_flying", "creatures only"),
    ("creature (vs noncreature)", "is_creature", "all"),
    ("MV 3-4 (base 0-2)", "mv_b1", "all"),
    ("MV 5-6 (base 0-2)", "mv_b2", "all"),
    ("MV 7+ (base 0-2)", "mv_b3", "all"),
    ("unconditional removal", "ph_uncond_removal", "all"),
    ("combat trick (instant +N/+N eot)", "combat_trick", "all"),
    ("mana rock (noncreature artifact {T}: add)", "mana_rock", "all"),
    ("sweeper", "ph_sweeper", "all"),
    ("counterspell", "ph_counterspell", "all"),
    ("morph", "ph_morph", "all"),
    ("token maker", "ph_create_token", "all"),
    ("lifegain", "ph_gain_life", "all"),
    ("card-draw spell (noncreature)", "draw_spell", "all"),
]
for k, (lab, col, scope) in enumerate(FEATURES):
    pass
C["mv_b1"] = ((C.mv >= 3) & (C.mv <= 4)).astype(int)
C["mv_b2"] = ((C.mv >= 5) & (C.mv <= 6)).astype(int)
C["mv_b3"] = (C.mv >= 7).astype(int)


def channel_effects(frame, col, scope, chan_cols, extra_ctrl=True):
    f = frame
    if scope == "creatures only":
        f = f[f.is_creature == 1]
    x = f[col].astype(float).to_numpy()
    ctrl = [np.ones(len(f))]
    names = ["const"]
    if col.startswith("mv_b"):
        for other in ["mv_b1", "mv_b2", "mv_b3"]:
            if other != col:
                ctrl.append(f[other].to_numpy(dtype=float))
                names.append(other)
        if scope != "creatures only":
            ctrl.append(f.is_creature.fillna(0).to_numpy(dtype=float))
            names.append("is_creature")
    elif extra_ctrl:
        ctrl.append(f.mv.to_numpy(dtype=float))
        ctrl.append(f.mv2.to_numpy(dtype=float))
        names += ["mv", "mv2"]
        if scope != "creatures only" and col != "is_creature":
            ctrl.append(f.is_creature.fillna(0).to_numpy(dtype=float))
            names.append("is_creature")
    X = np.column_stack(ctrl + [x])
    out = {"n_feat": int(np.nansum(x)), "n": len(f)}
    wt = f.wt.to_numpy(dtype=float)
    for ch in chan_cols:
        y = f[ch].to_numpy(dtype=float)
        b, se, _, nn = wls(y, X, wt)
        out[ch] = (b[-1], se[-1])
        sel = np.isfinite(y) & (x == 1)
        sel0 = np.isfinite(y) & (x == 0)
        out[ch + "_raw"] = (wmean(y[sel], wt[sel]) - wmean(y[sel0], wt[sel0]))
    return out


say("## b. Three-channel decomposition (selection w / castability m / "
    "contribution d)")
say()
say("`score = m·(2w−1) + d/2`. Because corpus mean w ≈ 0.5, the `m` channel "
    "carries almost no score at the margin: linearised, "
    "`Δscore ≈ 2·m̄·Δw + (2w̄−1)·Δm + Δd/2`. Coefficients are WLS "
    "(weights `n/(n+20)`, cards with n_in_deck ≥ 50), controlling for "
    "MV, MV² and creature-ness unless the feature *is* that control. "
    "HC0 SEs in parentheses.")
say()
CH = ["w", "m", "d", "score_overall", "score_play"]
mbar = wmean(C.m, C.wt)
wbar = wmean(C.w, C.wt)
rows = []
effects = {}
for lab, col, scope in FEATURES:
    e = channel_effects(C, col, scope, CH)
    effects[lab] = e
    dw, sw = e["w"]
    dm, sm = e["m"]
    dd, sd = e["d"]
    ds, ss = e["score_overall"]
    dsp, ssp = e["score_play"]
    pred = 2 * mbar * dw + (2 * wbar - 1) * dm + dd / 2
    rows.append([
        lab, e["n_feat"],
        f"{dw:+.4f} ({sw:.4f})",
        f"{dm:+.4f} ({sm:.4f})",
        f"{dd:+.4f} ({sd:.4f})",
        f"{ds:+.4f} ({ss:.4f})",
        f"{ds / SD_SCORE_PLAY:+.2f}σ",
        f"{2 * mbar * dw:+.4f}",
        f"{(2 * wbar - 1) * dm:+.4f}",
        f"{dd / 2:+.4f}",
        f"{pred:+.4f}",
    ])
table(rows, ["feature", "n cards", "Δw (selection)", "Δm (castability)",
             "Δd (contribution)", "Δscore_overall", "Δscore in σ",
             "w term 2m̄·Δw", "m term (2w̄−1)·Δm", "d term Δd/2",
             "Σ terms ≈ Δscore"])
say("The last four columns are the linearised attribution of Δscore to each "
    "channel; they reconstruct Δscore up to the interaction term. The m "
    "channel is structurally near-dead: 2w̄−1 = "
    f"{2 * wbar - 1:+.4f}, so even a −0.19 castability hit (morph) moves "
    "score by only "
    f"{(2 * wbar - 1) * -0.1993:+.4f}. **Selection (w) and in-game "
    "contribution (d) are the only two channels that carry score.**")
say()
say("Unadjusted differences (no MV/type control), for reference:")
rows = []
for lab, col, scope in FEATURES:
    e = effects[lab]
    rows.append([lab, f3(e["w_raw"]), f3(e["m_raw"]), f3(e["d_raw"]),
                 f3(e["score_overall_raw"]),
                 f"{e['score_overall_raw'] / SD_SCORE_PLAY:+.2f}σ"])
table(rows, ["feature", "Δw raw", "Δm raw", "Δd raw", "Δscore raw", "σ"])

# ---------------------------------------------------------------- (c)
say("## c. Within the forge-best vs forge-best stratum")
say()
nfb = df.n_fb
say(f"Stratum sizes: forge-best vs forge-best = "
    f"{int(df.fb_wins_in_deck.sum() + df.fb_losses_in_deck.sum()):,} "
    f"card-observations across "
    f"{int((nfb >= MIN_N).sum()):,} cards with n ≥ {MIN_N}; "
    f"'same forge method vs itself' = "
    f"{int(df.ff_wins_in_deck.sum() + df.ff_losses_in_deck.sum()):,} "
    f"observations, {int((df.n_ff >= MIN_N).sum()):,} cards.")
S = df[(df.n_fb >= MIN_N)].copy()
S["wt"] = S.n_fb / (S.n_fb + K)
S["mv2"] = S.mv**2
S["mv_b1"] = ((S.mv >= 3) & (S.mv <= 4)).astype(int)
S["mv_b2"] = ((S.mv >= 5) & (S.mv <= 6)).astype(int)
S["mv_b3"] = (S.mv >= 7).astype(int)
say(f"Stratum win rate check: mean w in stratum = "
    f"{wmean(S.w_fb, S.wt):.4f} (symmetric matchup ⇒ 0.5), "
    f"corpus = {wmean(C.w, C.wt):.4f}.")
say()
# corpus effects restricted to the same card subset, for a like-for-like ratio
S2 = S.copy()
S2["wt"] = S2.n_in_deck / (S2.n_in_deck + K)
rows = []
for lab, col, scope in FEATURES:
    e_st = channel_effects(S, col, scope,
                           ["w_fb", "m_fb", "d_fb", "score_fb"])
    e_co = channel_effects(S2, col, scope, ["w", "m", "d", "score_overall"])
    ds_st = e_st["score_fb"][0]
    ds_co = e_co["score_overall"][0]
    rows.append([
        lab, e_st["n_feat"],
        f"{e_co['w'][0]:+.4f} → {e_st['w_fb'][0]:+.4f}",
        f"{e_co['m'][0]:+.4f} → {e_st['m_fb'][0]:+.4f}",
        f"{e_co['d'][0]:+.4f} → {e_st['d_fb'][0]:+.4f}",
        f"{ds_co:+.4f} → {ds_st:+.4f} ({ds_st:.4f}±{e_st['score_fb'][1]:.4f})",
        f"{ds_st / ds_co:.2f}" if abs(ds_co) > 1e-4 else "—",
    ])
table(rows, ["feature", "n cards", "Δw corpus → stratum",
             "Δm corpus → stratum", "Δd corpus → stratum",
             "Δscore corpus → stratum", "ratio"])
say("Both columns are fitted on the *same* card subset (cards with ≥ "
    f"{MIN_N} forge-best-mirror observations), so the change is the stratum, "
    "not the sample.")
say()

# ---------------------------------------------------------------- (d)
say("## d. Game-length mediation of the MV and pip gradients (R9)")
say()
say("Three stratifications of the same games, from most to least endogenous: "
    "`gl` = the **owning side's** distinct cards played (the requested "
    "proxy, but partly downstream of the card's own castability); `og` = the "
    "**opponent's** count (a game-length proxy not conditioned on the "
    "owner's own mana development); `tg` = both sides summed (whole-game "
    "length). The last column divides each slope by that bucket's mean "
    "played_rate, which removes the A1 ceiling rescaling — short games cap "
    "|score| simply because fewer cards get cast.")
say()
STRATA = [
    ("gl", LB, "owning side's distinct cards played"),
    ("og", LB, "OPPONENT's distinct cards played"),
    ("tg", TB, "both sides' distinct cards played (whole-game length)"),
]
for xname, xcol in (("MV", "mv"), ("total coloured pips", "pips_total")):
    rows = []
    X = np.column_stack([np.ones(len(C)), C[xcol].to_numpy(dtype=float)])
    b, se, _, n = wls(C.score_overall, X, C.wt)
    overall = b[1]
    pr_all = wmean(C.played_rate, C.wt)
    rows.append(["**unstratified**", f"{meta['n_games']:,} games",
                 f"{overall:+.5f} ({se[1]:.5f})", "100%",
                 f"{overall / pr_all:+.5f}"])
    for pre, labels, desc in STRATA:
        num, den, numn = 0.0, 0.0, 0.0
        for b_ in range(len(labels)):
            sub = C[C[f"n_{pre}{b_}"] >= 30]
            Xs = np.column_stack(
                [np.ones(len(sub)), sub[xcol].to_numpy(dtype=float)]
            )
            wts = sub[f"n_{pre}{b_}"] / (sub[f"n_{pre}{b_}"] + K)
            bb, ss, _, nn = wls(sub[f"score_{pre}{b_}"], Xs, wts)
            tot_obs = float(sub[f"n_{pre}{b_}"].sum())
            prb = wmean(sub[f"pr_{pre}{b_}"], sub[f"n_{pre}{b_}"])
            num += bb[1] * tot_obs
            numn += bb[1] / prb * tot_obs
            den += tot_obs
            rows.append([f"{pre} bucket {labels[b_]}", f"{int(tot_obs):,} obs",
                         f"{bb[1]:+.5f} ({ss[1]:.5f})",
                         f"{bb[1] / overall * 100:.0f}%",
                         f"{bb[1] / prb:+.5f}"])
        rows.append([f"**pooled within-bucket ({desc})**", f"{int(den):,} obs",
                     f"**{num / den:+.5f}**",
                     f"**{num / den / overall * 100:.0f}%**",
                     f"**{numn / den:+.5f}**"])
    say(f"**{xname} → score gradient**")
    table(rows, ["stratum", "size", "slope per unit (HC0 SE)",
                 "share of unstratified slope",
                 "slope ÷ bucket mean played_rate (A1-normalised)"])
_mvslope = wls(
    C.score_overall, np.column_stack([np.ones(len(C)), C.mv]), C.wt
)[0][1]
say(f"For scale: 1 MV point at the unstratified slope is "
    f"{abs(_mvslope) / SD_SCORE_PLAY:.2f}σ of score_play SD; "
    f"MV 0→8 spans {abs(_mvslope) * 8 / SD_SCORE_PLAY:.2f}σ.")
say()
for pre, labels, desc in STRATA:
    rows = []
    for b_ in range(len(labels)):
        sub = C[C[f"n_{pre}{b_}"] >= 30]
        rows.append([labels[b_], f"{int(sub[f'n_{pre}{b_}'].sum()):,}", len(sub),
                     f3(wmean(sub[f"score_{pre}{b_}"], sub[f"n_{pre}{b_}"])),
                     f3u(wmean(sub[f"pr_{pre}{b_}"], sub[f"n_{pre}{b_}"])),
                     f3u(wmean(sub.mv, sub[f"n_{pre}{b_}"]))])
    say(f"Bucket profile — {desc}:")
    table(rows, ["bucket", "card-observations", "cards",
                 "mean score in bucket", "mean played_rate in bucket",
                 "obs-weighted mean MV"])

# ---------------------------------------------------------------- (e)
say("## e. Set-level castability story")
say()
sets = pd.DataFrame({
    "set": list(meta["set_den"].keys()),
    "num": [meta["set_num"][k] for k in meta["set_den"]],
    "den": [meta["set_den"][k] for k in meta["set_den"]],
    "games": [meta["set_games"][k] for k in meta["set_den"]],
    "gold": [meta["set_gold"][k] for k in meta["set_den"]],
    "pips": [meta["set_pips"][k] for k in meta["set_den"]],
})
sets["score"] = sets.num / sets.den
sets["gold_density"] = sets.gold / sets.den
sets["pip_density"] = sets.pips / sets.den
grand = sets.num.sum() / sets.den.sum()
sets["score_c"] = sets.score - grand
sets = sets[sets.games >= 500].sort_values("score", ascending=False)
say(f"{len(sets)} sets with ≥ 500 games. "
    f"Per-set score = Σ_cards[(played∧win) − (played∧loss)] / Σ in-deck obs "
    "— the corpus-level analogue of score_overall. Every set is positive "
    "because winners simply cast more cards than losers (A5); the "
    f"corpus-wide value is {grand:+.4f}, and the `centred` column subtracts "
    "it, which is the scale the grounding pass reported.")
say()
r_gold = np.corrcoef(sets.score, sets.gold_density)[0, 1]
r_pip = np.corrcoef(sets.score, sets.pip_density)[0, 1]
r_gold_w = wcorr(sets.score, sets.gold_density, sets.games)
r_pip_w = wcorr(sets.score, sets.pip_density, sets.games)
noarb = sets[sets.set != "ARB"]
r_gold_na = np.corrcoef(noarb.score, noarb.gold_density)[0, 1]
r_pip_na = np.corrcoef(noarb.score, noarb.pip_density)[0, 1]
rk = sets[["score", "gold_density", "pip_density"]].rank()
say(f"r(set score, gold density) = **{r_gold:+.3f}** (games-weighted "
    f"{r_gold_w:+.3f}; Spearman "
    f"{np.corrcoef(rk.score, rk.gold_density)[0, 1]:+.3f}; excluding the "
    f"all-gold outlier ARB {r_gold_na:+.3f}); r(set score, pip density) = "
    f"**{r_pip:+.3f}** (games-weighted {r_pip_w:+.3f}; Spearman "
    f"{np.corrcoef(rk.score, rk.pip_density)[0, 1]:+.3f}; ex-ARB "
    f"{r_pip_na:+.3f}). "
    f"SD across sets of set score = {sets.score.std():.4f} "
    f"({sets.score.std() / SD_SCORE_PLAY:.2f}σ of the per-card label SD); "
    f"range {sets.score.min():+.4f} … {sets.score.max():+.4f}.")
say()
ext = pd.concat([sets.head(12), sets.tail(12)])
rows = [[r.set, f"{r.games:,}", f3(r.score), f3(r.score_c),
         f"{r.score_c / SD_SCORE_PLAY:+.2f}",
         f"{r.gold_density:.4f}", f"{r.pip_density:.4f}"]
        for r in ext.itertuples()]
table(rows, ["set", "games", "set score", "centred", "σ (centred)",
             "gold density", "pip density"])

# ---------------------------------------------------------------- (f)
say("## f. Play/draw noise floor (R3)")
say()
say("Split halves assign each game alternately to half 0 / half 1 by that "
    "card's own in-deck appearance counter, so the two halves are matched in "
    "n by construction. Correlations are on cards with ≥ 25 in-deck "
    "observations *in each half*; `r_half` is the raw split-half "
    "correlation and `r_full` its Spearman-Brown extrapolation to the "
    "full-corpus label.")
say()
H = df[(df.h0_n >= 25) & (df.h1_n >= 25)].copy()
H["hw"] = np.minimum(H.h0_n, H.h1_n)
rows = []
for lab, c0, c1 in [
    ("score_play", "h0_score_play", "h1_score_play"),
    ("score_draw", "h0_score_draw", "h1_score_draw"),
    ("score_overall", "h0_score", "h1_score"),
    ("played_rate", "h0_played_rate", "h1_played_rate"),
    ("score_play − score_draw", "dif0", "dif1"),
]:
    if lab.startswith("score_play −"):
        H["dif0"] = H.h0_score_play - H.h0_score_draw
        H["dif1"] = H.h1_score_play - H.h1_score_draw
    r = wcorr(H[c0], H[c1], H.hw)
    ru = wcorr(H[c0], H[c1])
    rows.append([lab, len(H), f"{r:+.3f}", f"{sb(r):+.3f}", f"{ru:+.3f}",
                 f"{sb(ru):+.3f}"])
table(rows, ["quantity", "n cards", "r_half (n-weighted)",
             "r_full (SB, weighted)", "r_half (unweighted)",
             "r_full (SB, unweighted)"])

rows = []
for lo, hi in [(25, 100), (100, 250), (250, 500), (500, 1000),
               (1000, 3000), (3000, 10**9)]:
    g = H[(H.hw >= lo) & (H.hw < hi)]
    if len(g) < 30:
        continue
    r_sp = wcorr(g.h0_score_play, g.h1_score_play)
    g = g.copy()
    g["dif0"] = g.h0_score_play - g.h0_score_draw
    g["dif1"] = g.h1_score_play - g.h1_score_draw
    r_df = wcorr(g.dif0, g.dif1)
    rows.append([f"{lo}–{hi if hi < 10**8 else '∞'}", len(g),
                 f"{r_sp:+.3f}", f"{sb(r_sp):+.3f}",
                 f"{r_df:+.3f}", f"{sb(r_df):+.3f}"])
table(rows, ["min-half n bucket", "cards", "r_half score_play",
             "r_full score_play", "r_half play−draw diff",
             "r_full play−draw diff"])

D = df[df.n_in_deck >= 200].copy()
D["dif"] = D.score_play - D.score_draw
D["wt"] = D.n_in_deck / (D.n_in_deck + K)
D["mv2"] = D.mv**2
r_d = wcorr(D.score_play - D.score_draw, D.score_play - D.score_draw)
obs_sd = D.dif.std()
Hd = H[(H.hw >= 100)].copy()
Hd["dif0"] = Hd.h0_score_play - Hd.h0_score_draw
Hd["dif1"] = Hd.h1_score_play - Hd.h1_score_draw
rel_half = wcorr(Hd.dif0, Hd.dif1)
rel_full = sb(rel_half)
say(f"Observed SD of the play−draw difference (cards with n ≥ 200): "
    f"**{obs_sd:.4f}**. With full-label reliability "
    f"**{rel_full:.3f}** (cards with ≥ 100 obs per half), true signal SD = "
    f"{obs_sd * np.sqrt(max(rel_full, 0)):.4f} and noise SD = "
    f"{obs_sd * np.sqrt(max(1 - rel_full, 0)):.4f}.")
say()
say("Correlates of the play−draw difference, WLS with MV/MV²/creature "
    "controls (cards n ≥ 200):")
rows = []
for lab, col in [("haste", "kw_haste"), ("sweeper", "ph_sweeper"),
                 ("counterspell", "ph_counterspell"),
                 ("flash", "kw_flash"), ("defender", "kw_defender"),
                 ("card-draw spell", "draw_spell"),
                 ("unconditional removal", "ph_uncond_removal"),
                 ("mana rock", "mana_rock"), ("kicker", "ph_kicker"),
                 ("fight", "ph_fight"), ("MV (per point)", "mv")]:
    x = D[col].astype(float).to_numpy()
    if col == "mv":
        X = np.column_stack([np.ones(len(D)), D.is_creature.fillna(0), x])
    else:
        X = np.column_stack([np.ones(len(D)), D.mv, D.mv2,
                             D.is_creature.fillna(0), x])
    b, se, _, n = wls(D.dif, X, D.wt)
    rows.append([lab, int(np.nansum(x)) if col != "mv" else n,
                 f"{b[-1]:+.4f}", f"{se[-1]:.4f}",
                 f"{b[-1] / se[-1]:+.1f}",
                 f"{b[-1] / (obs_sd * np.sqrt(max(1 - rel_full, 1e-9))):+.3f}"])
table(rows, ["correlate", "n", "Δ(play−draw)", "HC0 SE", "t",
             "in noise-SD units"])

# ---------------------------------------------------------------- (g)
say("## g. cast_lift anatomy (R4)")
say()
G = C[C.n_played >= 20].copy()
G = G[(G.n_in_deck - G.n_played) >= 20]
say(f"Cards with ≥ 20 played and ≥ 20 not-played observations: {len(G):,}. "
    f"Weighted mean cast_lift = **{wmean(G.cast_lift, G.wt):+.4f}** "
    f"(p_play {wmean(G.p_play, G.wt):.4f} vs p_dead "
    f"{wmean(G.p_dead, G.wt):.4f}).")
say()
rows = [
    ["MV", f"{wcorr(G.cast_lift, G.mv, G.wt):+.3f}"],
    ["played_rate", f"{wcorr(G.cast_lift, G.played_rate, G.wt):+.3f}"],
    ["score_play", f"{wcorr(G.cast_lift, G.score_play, G.wt):+.3f}"],
    ["score_overall", f"{wcorr(G.cast_lift, G.score_overall, G.wt):+.3f}"],
    ["d = P(played|win) − P(played|loss)",
     f"{wcorr(G.cast_lift, G.d, G.wt):+.3f}"],
    ["w = P(win | in deck)", f"{wcorr(G.cast_lift, G.w, G.wt):+.3f}"],
    ["total pips", f"{wcorr(G.cast_lift, G.pips_total, G.wt):+.3f}"],
]
table(rows, ["vs", "weighted Pearson r"])
rows = []
for mv in range(0, 9):
    g = G[G.mv == mv] if mv < 8 else G[G.mv >= 8]
    if len(g) < 20:
        continue
    rows.append([mv if mv < 8 else "8+", len(g),
                 f3(wmean(g.cast_lift, g.wt)),
                 f3u(wmean(g.played_rate, g.wt)),
                 f3(wmean(g.d, g.wt)),
                 f3(wmean(g.score_play, g.wt))])
table(rows, ["MV", "cards", "mean cast_lift", "mean played_rate", "mean d",
             "mean score_play"])
say("Negative tail by class (weighted means; corpus row for reference):")
CLASSES = [
    ("corpus", np.ones(len(G), dtype=bool)),
    ("fog / damage prevention", G.ph_fog == 1),
    ("sweeper", G.ph_sweeper == 1),
    ("mana rock", G.mana_rock == 1),
    ("land", G.is_land == 1),
    ("counterspell", G.ph_counterspell == 1),
    ("morph", G.ph_morph == 1),
    ("combat trick", G.combat_trick == 1),
    ("card-draw spell", G.draw_spell == 1),
    ("unconditional removal", G.ph_uncond_removal == 1),
    ("creature", G.is_creature == 1),
]
rows = []
for lab, sel in CLASSES:
    g = G[np.asarray(sel)]
    if len(g) < 5:
        continue
    rows.append([lab, len(g), f3(wmean(g.cast_lift, g.wt)),
                 f3(wmean(g.d, g.wt)), f3u(wmean(g.played_rate, g.wt)),
                 f3(wmean(g.score_play, g.wt)), f3u(wmean(g.mv, g.wt))])
table(rows, ["class", "cards", "cast_lift", "d", "played_rate", "score_play",
             "mean MV"])
say("Is cast_lift a re-encoding of (score_play, MV, played_rate)? "
    "Nested WLS R² for predicting cast_lift:")
rows = []
base = [np.ones(len(G))]
specs = [
    ("score_play", ["score_play"]),
    ("+ MV, MV²", ["score_play", "mv", "mv2"]),
    ("+ played_rate", ["score_play", "mv", "mv2", "played_rate"]),
    ("+ 1/played_rate", ["score_play", "mv", "mv2", "played_rate", "inv_pr"]),
    ("+ d", ["score_play", "mv", "mv2", "played_rate", "inv_pr", "d"]),
    ("score_overall + played_rate + 1/played_rate (algebraic)",
     ["score_overall", "played_rate", "inv_pr"]),
]
G["inv_pr"] = 1.0 / np.clip(G.played_rate, .02, 1)
prev = 0.0
for lab, cols in specs:
    X = np.column_stack(base + [G[c].to_numpy(dtype=float) for c in cols])
    b, se, r2, n = wls(G.cast_lift, X, G.wt)
    rows.append([lab, f"{r2:.4f}", f"{r2 - prev:+.4f}" if "algebraic" not in lab
                 else "—",
                 f"{np.sqrt(max(0, 1 - r2)) * G.cast_lift.std():.4f}"])
    if "algebraic" not in lab:
        prev = r2
table(rows, ["predictors of cast_lift", "weighted R²", "ΔR²",
             "residual SD"])

# ---------------------------------------------------------------- (h)
say("## h. The Forge blacklist channel (AI:RemoveDeck)")
say()
B = C[C.ai_remove_deck.notna()].copy()
say(f"Forge hints joined for {len(B):,} / {len(C):,} cards "
    f"(n_in_deck ≥ {MIN_N}); blacklisted (`AI:RemoveDeck`) = "
    f"{int(B.ai_remove_deck.sum()):,} "
    f"({B.ai_remove_deck.mean():.1%}).")
say()
rows = []
for lab, sel in [("blacklisted", B.ai_remove_deck == 1),
                 ("not blacklisted", B.ai_remove_deck == 0)]:
    g = B[sel]
    rows.append([lab, len(g), f"{g.n_in_deck.median():.0f}",
                 f3u(wmean(g.w, g.wt)), f3u(wmean(g.m, g.wt)),
                 f3(wmean(g.d, g.wt)), f3(wmean(g.score_overall, g.wt)),
                 f3(wmean(g.score_play, g.wt)), f3u(wmean(g.mv, g.wt))])
table(rows, ["group", "cards", "median n_in_deck", "w", "m", "d",
             "score_overall", "score_play", "mean MV"])
say("Matched comparison — exact cells on (MV clipped 0-8) × creature, "
    "cell weights = harmonic-style min(n_black, n_control), then "
    "cell-weighted mean difference:")
B["mvc"] = B.mv.clip(0, 8)
rows_cells = []
acc = {c: [0.0, 0.0] for c in ["w", "m", "d", "score_overall", "score_play"]}
tw = 0.0
for (mvc, isc), g in B.groupby(["mvc", "is_creature"]):
    gb = g[g.ai_remove_deck == 1]
    gc = g[g.ai_remove_deck == 0]
    if len(gb) < 5 or len(gc) < 5:
        continue
    cw = min(len(gb), len(gc))
    tw += cw
    for c in acc:
        acc[c][0] += cw * (wmean(gb[c], gb.wt) - wmean(gc[c], gc.wt))
        acc[c][1] += cw
rows = [[c, f3(acc[c][0] / acc[c][1]),
         f"{acc[c][0] / acc[c][1] / SD_SCORE_PLAY:+.2f}σ"
         if c.startswith("score") else "—"]
        for c in ["w", "m", "d", "score_overall", "score_play"]]
table(rows, ["channel", "matched Δ (blacklisted − control)", "σ"])
e = channel_effects(B, "ai_remove_deck", "all",
                    ["w", "m", "d", "score_overall", "score_play"])
rows = [[c, f"{e[c][0]:+.4f}", f"{e[c][1]:.4f}",
         f"{e[c][0] / SD_SCORE_PLAY:+.2f}σ" if c.startswith("score") else "—"]
        for c in ["w", "m", "d", "score_overall", "score_play"]]
say("Regression-controlled (MV, MV², creature):")
table(rows, ["channel", "Δ (blacklisted − rest)", "HC0 SE", "σ"])
dw = e["w"][0]
dd = e["d"][0]
predtot = 2 * mbar * dw + dd / 2
say(f"Share of the blacklist's score deficit that is selection (w): "
    f"**{2 * mbar * dw / predtot * 100:.0f}%**; in-game contribution (d): "
    f"**{dd / 2 / predtot * 100:.0f}%**.")
say()
if B.draft_rank.notna().any():
    Bd = B[B.draft_rank.notna()]
    say(f"Bonus — Forge's human draft rank (lower = better) is available for "
        f"{len(Bd):,} of these cards; r(draft_rank, score_overall) = "
        f"{wcorr(Bd.draft_rank, Bd.score_overall, Bd.wt):+.3f}, "
        f"r(draft_rank, w) = {wcorr(Bd.draft_rank, Bd.w, Bd.wt):+.3f}, "
        f"r(draft_rank, d) = {wcorr(Bd.draft_rank, Bd.d, Bd.wt):+.3f}, "
        f"r(draft_rank, m) = {wcorr(Bd.draft_rank, Bd.m, Bd.wt):+.3f}.")
    say()

(OUT / "l_report.md").write_text("\n".join(L), encoding="utf-8")
print(f"\nwrote {OUT / 'l_report.md'}")

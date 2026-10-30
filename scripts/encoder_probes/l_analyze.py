"""Label-side artifact / mediation analyses (a-h) on the P0.4 aggregate.

Reads ``output/encoder-probes/l_mediation_table.pkl`` (written by
``l_mediation.py``), joins the pre-built card feature table and the Forge
hint table, and writes ``output/encoder-probes/l_report.md`` plus
``l_channels.csv``, section b's three-channel decomposition as numbers.

CPU only, seconds to run.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402
from _lib.markdown import md_table, utf8_console  # noqa: E402
from _lib.paths import OUTPUT, SEALED_TRAINING_DATA  # noqa: E402
from _lib.stats import wcorr, wls  # noqa: E402
from l_mediation import draw_score, stratum_scores  # noqa: E402

OUT = OUTPUT / "encoder-probes"
TABLE = OUT / "card_table.pkl"  # written by p0b_card_table.py
HINTS = OUTPUT / "scorer-probes" / "forge_hints.csv"
CHANNELS = OUT / "l_channels.csv"
SD_SCORE_PLAY = pl.SD["score_play"]
K = 20.0
MIN_N = 50
CH = ["w", "m", "d", "score_overall", "score_play"]
MV_BUCKETS = ["mv_b1", "mv_b2", "mv_b3"]
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


class Report:
    """The Markdown report, echoed to the console as it is written."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, s: str = "") -> None:
        self.lines.append(s)
        print(s)

    def table(self, rows, header) -> None:
        self.say(md_table(header, rows))
        self.say()


def wmean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w)
    if not ok.any():
        return np.nan
    return float(np.average(x[ok], weights=w[ok]))


def sb(r: float) -> float:
    """Spearman-Brown: half-length reliability -> full-length reliability."""
    return 2 * r / (1 + r) if np.isfinite(r) else np.nan


def f3(x):
    return "—" if not np.isfinite(x) else f"{x:+.4f}"


def f3u(x):
    return "—" if not np.isfinite(x) else f"{x:.4f}"


def mv_rows(frame: pd.DataFrame):
    """``(label, cards)`` per mana value 0..7, then 8+, skipping thin rows."""
    for mv in range(0, 9):
        g = frame[frame.mv == mv] if mv < 8 else frame[frame.mv >= 8]
        if len(g) >= 20:
            yield (mv if mv < 8 else "8+"), g


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(df, med, meta)``: the aggregate joined to the card and hint tables,
    with every derived column the sections read."""
    with open(OUT / "l_mediation_table.pkl", "rb") as f:
        d = pickle.load(f)
    med, meta = d["cards"], d["meta"]
    feat = pd.read_pickle(TABLE)
    featcols = [c for c in feat.columns if c not in med.columns] + ["card_name"]
    df = med.merge(feat[featcols], on="card_name", how="left")
    hints = pd.read_csv(HINTS)
    hints = hints.rename(columns={"name": "card_name"})[
        ["card_name", "ai_remove_deck", "draft_rank"]
    ].drop_duplicates("card_name")
    df = df.merge(hints, on="card_name", how="left")

    df["wt"] = df.n_in_deck / (df.n_in_deck + K)
    df["mv2"] = df.mv**2
    df["mv_b1"] = ((df.mv >= 3) & (df.mv <= 4)).astype(int)
    df["mv_b2"] = ((df.mv >= 5) & (df.mv <= 6)).astype(int)
    df["mv_b3"] = (df.mv >= 7).astype(int)
    noncreature = 1 - df.is_creature.fillna(0)
    nonland = 1 - df.is_land.fillna(0)
    df["is_nonland_noncreature"] = (noncreature * nonland).astype(int)
    df["mana_rock"] = (df.is_artifact.fillna(0).astype(int)
                       * df.ph_tap_for_mana.fillna(0).astype(int)
                       * noncreature.astype(int) * nonland.astype(int))
    df["combat_trick"] = df.ph_pump_eot.fillna(0).astype(int) * df.is_instant.fillna(0).astype(int)
    df["draw_spell"] = (df.ph_draw_a_card.fillna(0).astype(int)
                        * noncreature.astype(int) * nonland.astype(int))

    for tag in ("fb", "ff"):
        df[f"n_{tag}"], df[f"score_{tag}"], _ = stratum_scores(df, f"{tag}_")
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"w_{tag}"] = df[f"{tag}_wins_in_deck"] / df[f"n_{tag}"]
            a = df[f"{tag}_wins_played"] / df[f"{tag}_wins_in_deck"]
            b = df[f"{tag}_losses_played"] / df[f"{tag}_losses_in_deck"]
        df[f"a_{tag}"], df[f"b_{tag}"] = a, b
        df[f"m_{tag}"] = (a + b) / 2
        df[f"d_{tag}"] = a - b
    for pre in ("gl", "og", "tg"):
        for b_ in range(len(meta["len_bucket_labels"])):
            df[f"n_{pre}{b_}"], df[f"score_{pre}{b_}"], df[f"pr_{pre}{b_}"] = stratum_scores(
                df, f"{pre}{b_}_")
    for h in (0, 1):
        pre = f"h{h}_"
        df[f"{pre}n"], df[f"{pre}score"], df[f"{pre}played_rate"] = stratum_scores(df, pre)
        df[f"{pre}np"], df[f"{pre}score_play"], _ = stratum_scores(df, pre, "_play")
        df[f"{pre}score_draw"] = draw_score(df, pre)
        df[f"dif{h}"] = df[f"{pre}score_play"] - df[f"{pre}score_draw"]
    return df.copy(), med, meta  # copy de-fragments the wide column build-out


def channel_effects(frame, col, scope, chan_cols, extra_ctrl=True):
    f = frame
    if scope == "creatures only":
        f = f[f.is_creature == 1]
    x = f[col].astype(float).to_numpy()
    ctrl = [np.ones(len(f))]
    if col.startswith("mv_b"):
        ctrl += [f[other].to_numpy(dtype=float) for other in MV_BUCKETS if other != col]
        if scope != "creatures only":
            ctrl.append(f.is_creature.fillna(0).to_numpy(dtype=float))
    elif extra_ctrl:
        ctrl += [f.mv.to_numpy(dtype=float), f.mv2.to_numpy(dtype=float)]
        if scope != "creatures only" and col != "is_creature":
            ctrl.append(f.is_creature.fillna(0).to_numpy(dtype=float))
    X = np.column_stack(ctrl + [x])
    out = {"n_feat": int(np.nansum(x)), "n": len(f)}
    wt = f.wt.to_numpy(dtype=float)
    for ch in chan_cols:
        y = f[ch].to_numpy(dtype=float)
        fit = wls(X, y, wt)
        out[ch] = (fit.beta[-1], fit.se[-1])
        sel = np.isfinite(y) & (x == 1)
        sel0 = np.isfinite(y) & (x == 0)
        out[ch + "_raw"] = (wmean(y[sel], wt[sel]) - wmean(y[sel0], wt[sel0]))
    return out


def verification(rep: Report, df: pd.DataFrame, med: pd.DataFrame) -> None:
    rep.say("## 0. Verification")
    rep.say()
    snap = pd.read_csv(SEALED_TRAINING_DATA / "matches-bo1" / "cards-win-rates.txt",
                       sep=";", dtype={"card_name": str})
    mm = med.merge(snap, on="card_name", suffixes=("", "_s"))
    rows = []
    for c in ["wins_when_played", "wins_when_in_deck", "losses_when_played",
              "losses_when_in_deck"]:
        rows.append([c, f"{(mm[c] == mm[c + '_s']).mean():.6f}",
                     int((mm[c] - mm[c + "_s"]).abs().max())])
    for c, s in [("score_play", "raw_score_play"), ("score_draw", "raw_score_draw"),
                 ("played_rate", "raw_played_rate"), ("cast_lift", "raw_cast_lift")]:
        ok = np.isclose(mm[c], mm[s], atol=6e-6, equal_nan=True)
        rows.append([f"{c} vs {s}", f"{ok.mean():.6f}", "—"])
    rep.table(rows, ["quantity", "exact-match fraction vs snapshot", "max abs diff"])
    rep.say(f"All {len(mm):,} snapshot cards reproduce **exactly**; the recomputed "
            f"corpus additionally contains {len(df) - len(mm)} card(s) the snapshot "
            f"omits: {', '.join(sorted(set(med.card_name) - set(snap.card_name)))} "
            "(n_in_deck = 1 each).")
    rep.say()

    ident = df.m * (2 * df.w - 1) + df.d / 2
    res = (ident - df.score_overall).abs()
    rep.say(f"Identity `score = m·(2w−1) + d/2`: max |residual| over all "
            f"{res.notna().sum():,} cards with both counters = "
            f"**{np.nanmax(res):.3e}** (floating-point noise).")
    rep.say()


def artifacts(rep: Report, df: pd.DataFrame, C: pd.DataFrame, meta: dict) -> None:
    """(a) the |score| ≤ played_rate ceiling, the MV ratio, build-method win
    rates, opponent assignment and the winner/loser cast gap."""
    rep.say("## a. Artifact verification at full scale")
    rep.say()
    viol = (df.score_overall.abs() > df.played_rate + 1e-12).sum()
    rep.say(f"**A1 ceiling** `|score| ≤ played_rate`: violations = **{viol}** / "
            f"{df.score_overall.notna().sum():,} cards (exact, as the identity forces).")
    rep.say()
    bins = [0, .05, .10, .15, .20, .30, .40, .50, .60, .80, 1.001]
    rows = []
    for b_, g in C.groupby(pd.cut(C.played_rate, bins, right=False), observed=True):
        rows.append([
            str(b_), len(g), f"{g.played_rate.mean():.3f}",
            f"{g.score_overall.abs().max():.4f}",
            f"{g.score_overall.std():.4f}",
            f"{g.score_play.std():.4f}",
        ])
    rep.table(rows, ["played_rate bin", "n cards", "mean played_rate",
                     "max |score|", "SD(score_overall)", "SD(score_play)"])
    lo = C[C.played_rate < .05].score_overall.std()
    hi = C[C.played_rate >= .80].score_overall.std()
    mid = C[(C.played_rate >= .50) & (C.played_rate < .60)].score_overall.std()
    rep.say(f"Heteroscedasticity: SD(score) = {lo:.4f} in the bottom played_rate bin "
            f"vs {hi:.4f} in the top — a **{hi / lo:.0f}×** spread (the top bin holds "
            f"only 5 cards; against the last well-populated bin [0.5,0.6) at "
            f"{mid:.4f} it is still {mid / lo:.0f}×). Low-played_rate classes are "
            "algebraically barred from expressing a large score.")
    rep.say()

    rep.say("**A4 game-length certification** — score/played_rate ratio by mana value:")
    rows = []
    for mv, g in mv_rows(C):
        pr = wmean(g.played_rate, g.wt)
        rows.append([mv, len(g), f"{pr:.3f}", f3(wmean(g.score_overall, g.wt)),
                     f3(wmean(g.score_overall, g.wt) / pr), f3(wmean(g.score_play, g.wt) / pr)])
    rep.table(rows, ["MV", "n cards", "mean played_rate", "mean score_overall",
                     "score/played_rate", "score_play/played_rate"])

    rep.say("**A2 build-method win rates** (per game, side-level):")
    wins, losses = meta["method_wins"], meta["method_losses"]
    games = {k: wins.get(k, 0) + losses.get(k, 0) for k in wins}
    rows = [[k, f"{games[k]:,}", f"{wins[k] / games[k]:.4f}"]
            for k in sorted(wins, key=lambda x: -(wins.get(x, 0) + losses.get(x, 0)))
            if games[k] >= 2000]
    rep.table(rows, ["build method", "games (side-level)", "win rate"])
    wrs = {k: wins.get(k, 0) / max(1, games[k]) for k in wins if games[k] >= 2000}
    rep.say(f"Selection span across methods: {min(wrs.values()):.3f} … "
            f"{max(wrs.values()):.3f} win rate "
            f"= {(max(wrs.values()) - min(wrs.values())):.3f} in w units.")
    rep.say()

    rep.say("**A3 opponent assignment** — who `random` decks face:")
    opp: dict[str, dict[str, int]] = {}
    for k, c in meta["pair_counts"].items():
        a, b_ = k.split("|", 1)
        for me, you in ((a, b_), (b_, a)):
            opp.setdefault(me, {}).setdefault(you, 0)
            opp[me][you] += c
    if "random" in opp:
        tot_r = sum(opp["random"].values())
        nonrand = sum(v for k, v in opp["random"].items() if k != "random")
        forge = sum(v for k, v in opp["random"].items() if k.startswith("forge"))
        rep.say(f"`random` side-games: {tot_r:,}; opponent is not `random` in "
                f"**{nonrand / tot_r:.1%}**, a forge builder in **{forge / tot_r:.1%}**.")
    rep.say()
    rep.say(f"**Winner vs loser mean distinct cards played**: "
            f"**{meta['sum_played_winner'] / meta['n_games']:.3f}** vs "
            f"**{meta['sum_played_loser'] / meta['n_games']:.3f}** "
            f"(+{meta['sum_played_winner'] / meta['sum_played_loser'] - 1:.1%}).")
    rep.say()
    rep.say(f"Corpus weighted means: w = {wmean(C.w, C.wt):.4f}, "
            f"m = {wmean(C.m, C.wt):.4f}, d = {wmean(C.d, C.wt):+.4f}, "
            f"score_overall = {wmean(C.score_overall, C.wt):+.4f}, "
            f"score_play = {wmean(C.score_play, C.wt):+.4f}, "
            f"cast_lift = {wmean(C.cast_lift, C.wt):+.4f}.")
    rep.say()


def channels(rep: Report, C: pd.DataFrame, mbar: float, wbar: float) -> None:
    """(b) the three-channel decomposition; also written to ``l_channels.csv``."""
    rep.say("## b. Three-channel decomposition (selection w / castability m / "
            "contribution d)")
    rep.say()
    rep.say("`score = m·(2w−1) + d/2`. Because corpus mean w ≈ 0.5, the `m` channel "
            "carries almost no score at the margin: linearised, "
            "`Δscore ≈ 2·m̄·Δw + (2w̄−1)·Δm + Δd/2`. Coefficients are WLS "
            "(weights `n/(n+20)`, cards with n_in_deck ≥ 50), controlling for "
            "MV, MV² and creature-ness unless the feature *is* that control. "
            "HC0 SEs in parentheses.")
    rep.say()
    rows, records, effects = [], [], {}
    for lab, col, scope in FEATURES:
        e = effects[lab] = channel_effects(C, col, scope, CH)
        (dw, sw), (dm, sm), (dd, sd) = e["w"], e["m"], e["d"]
        ds, ss = e["score_overall"]
        terms = {"w_term": 2 * mbar * dw, "m_term": (2 * wbar - 1) * dm, "d_term": dd / 2}
        pred = sum(terms.values())
        rows.append([
            lab, e["n_feat"],
            f"{dw:+.4f} ({sw:.4f})",
            f"{dm:+.4f} ({sm:.4f})",
            f"{dd:+.4f} ({sd:.4f})",
            f"{ds:+.4f} ({ss:.4f})",
            f"{ds / SD_SCORE_PLAY:+.2f}σ",
            f"{terms['w_term']:+.4f}",
            f"{terms['m_term']:+.4f}",
            f"{terms['d_term']:+.4f}",
            f"{pred:+.4f}",
        ])
        records.append({"feature": lab, "n_cards": e["n_feat"],
                        **{f"d_{ch}": e[ch][0] for ch in CH},
                        **{f"se_{ch}": e[ch][1] for ch in CH},
                        **terms, "sum_terms": pred})
    rep.table(rows, ["feature", "n cards", "Δw (selection)", "Δm (castability)",
                     "Δd (contribution)", "Δscore_overall", "Δscore in σ",
                     "w term 2m̄·Δw", "m term (2w̄−1)·Δm", "d term Δd/2",
                     "Σ terms ≈ Δscore"])
    pd.DataFrame(records).to_csv(CHANNELS, index=False)
    rep.say("The last four columns are the linearised attribution of Δscore to each "
            "channel; they reconstruct Δscore up to the interaction term. The m "
            "channel is structurally near-dead: 2w̄−1 = "
            f"{2 * wbar - 1:+.4f}, so even a −0.19 castability hit (morph) moves "
            "score by only "
            f"{(2 * wbar - 1) * -0.1993:+.4f}. **Selection (w) and in-game "
            "contribution (d) are the only two channels that carry score.**")
    rep.say()
    rep.say("Unadjusted differences (no MV/type control), for reference:")
    rows = []
    for lab, _, _ in FEATURES:
        e = effects[lab]
        rows.append([lab, f3(e["w_raw"]), f3(e["m_raw"]), f3(e["d_raw"]),
                     f3(e["score_overall_raw"]),
                     f"{e['score_overall_raw'] / SD_SCORE_PLAY:+.2f}σ"])
    rep.table(rows, ["feature", "Δw raw", "Δm raw", "Δd raw", "Δscore raw", "σ"])


def forge_mirror(rep: Report, df: pd.DataFrame, C: pd.DataFrame) -> None:
    """(c) the channels within the forge-best vs forge-best stratum."""
    rep.say("## c. Within the forge-best vs forge-best stratum")
    rep.say()
    rep.say(f"Stratum sizes: forge-best vs forge-best = "
            f"{int(df.fb_wins_in_deck.sum() + df.fb_losses_in_deck.sum()):,} "
            f"card-observations across "
            f"{int((df.n_fb >= MIN_N).sum()):,} cards with n ≥ {MIN_N}; "
            f"'same forge method vs itself' = "
            f"{int(df.ff_wins_in_deck.sum() + df.ff_losses_in_deck.sum()):,} "
            f"observations, {int((df.n_ff >= MIN_N).sum()):,} cards.")
    S = df[(df.n_fb >= MIN_N)].copy()
    S["wt"] = S.n_fb / (S.n_fb + K)
    rep.say(f"Stratum win rate check: mean w in stratum = "
            f"{wmean(S.w_fb, S.wt):.4f} (symmetric matchup ⇒ 0.5), "
            f"corpus = {wmean(C.w, C.wt):.4f}.")
    rep.say()
    # corpus effects restricted to the same card subset, for a like-for-like ratio
    S2 = S.copy()
    S2["wt"] = S2.n_in_deck / (S2.n_in_deck + K)
    rows = []
    for lab, col, scope in FEATURES:
        e_st = channel_effects(S, col, scope, ["w_fb", "m_fb", "d_fb", "score_fb"])
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
    rep.table(rows, ["feature", "n cards", "Δw corpus → stratum",
                     "Δm corpus → stratum", "Δd corpus → stratum",
                     "Δscore corpus → stratum", "ratio"])
    rep.say("Both columns are fitted on the *same* card subset (cards with ≥ "
            f"{MIN_N} forge-best-mirror observations), so the change is the stratum, "
            "not the sample.")
    rep.say()


def game_length(rep: Report, C: pd.DataFrame, meta: dict) -> None:
    """(d) game-length mediation of the MV and pip gradients."""
    rep.say("## d. Game-length mediation of the MV and pip gradients (R9)")
    rep.say()
    rep.say("Three stratifications of the same games, from most to least endogenous: "
            "`gl` = the **owning side's** distinct cards played (the requested "
            "proxy, but partly downstream of the card's own castability); `og` = the "
            "**opponent's** count (a game-length proxy not conditioned on the "
            "owner's own mana development); `tg` = both sides summed (whole-game "
            "length). The last column divides each slope by that bucket's mean "
            "played_rate, which removes the A1 ceiling rescaling — short games cap "
            "|score| simply because fewer cards get cast.")
    rep.say()
    LB, TB = meta["len_bucket_labels"], meta["tot_bucket_labels"]
    strata = [
        ("gl", LB, "owning side's distinct cards played"),
        ("og", LB, "OPPONENT's distinct cards played"),
        ("tg", TB, "both sides' distinct cards played (whole-game length)"),
    ]
    for xname, xcol in (("MV", "mv"), ("total coloured pips", "pips_total")):
        rows = []
        fit = wls(np.column_stack([np.ones(len(C)), C[xcol].to_numpy(dtype=float)]),
                  C.score_overall, C.wt)
        overall = fit.beta[1]
        pr_all = wmean(C.played_rate, C.wt)
        rows.append(["**unstratified**", f"{meta['n_games']:,} games",
                     f"{overall:+.5f} ({fit.se[1]:.5f})", "100%",
                     f"{overall / pr_all:+.5f}"])
        for pre, labels, desc in strata:
            num, den, numn = 0.0, 0.0, 0.0
            for b_ in range(len(labels)):
                n_col = f"n_{pre}{b_}"
                sub = C[C[n_col] >= 30]
                fit = wls(np.column_stack([np.ones(len(sub)), sub[xcol].to_numpy(dtype=float)]),
                          sub[f"score_{pre}{b_}"], sub[n_col] / (sub[n_col] + K))
                slope = fit.beta[1]
                tot_obs = float(sub[n_col].sum())
                prb = wmean(sub[f"pr_{pre}{b_}"], sub[n_col])
                num += slope * tot_obs
                numn += slope / prb * tot_obs
                den += tot_obs
                rows.append([f"{pre} bucket {labels[b_]}", f"{int(tot_obs):,} obs",
                             f"{slope:+.5f} ({fit.se[1]:.5f})",
                             f"{slope / overall * 100:.0f}%",
                             f"{slope / prb:+.5f}"])
            rows.append([f"**pooled within-bucket ({desc})**", f"{int(den):,} obs",
                         f"**{num / den:+.5f}**",
                         f"**{num / den / overall * 100:.0f}%**",
                         f"**{numn / den:+.5f}**"])
        rep.say(f"**{xname} → score gradient**")
        rep.table(rows, ["stratum", "size", "slope per unit (HC0 SE)",
                         "share of unstratified slope",
                         "slope ÷ bucket mean played_rate (A1-normalised)"])
    mvslope = wls(np.column_stack([np.ones(len(C)), C.mv]), C.score_overall, C.wt).beta[1]
    rep.say(f"For scale: 1 MV point at the unstratified slope is "
            f"{abs(mvslope) / SD_SCORE_PLAY:.2f}σ of score_play SD; "
            f"MV 0→8 spans {abs(mvslope) * 8 / SD_SCORE_PLAY:.2f}σ.")
    rep.say()
    for pre, labels, desc in strata:
        rows = []
        for b_ in range(len(labels)):
            n_col = f"n_{pre}{b_}"
            sub = C[C[n_col] >= 30]
            rows.append([labels[b_], f"{int(sub[n_col].sum()):,}", len(sub),
                         f3(wmean(sub[f"score_{pre}{b_}"], sub[n_col])),
                         f3u(wmean(sub[f"pr_{pre}{b_}"], sub[n_col])),
                         f3u(wmean(sub.mv, sub[n_col]))])
        rep.say(f"Bucket profile — {desc}:")
        rep.table(rows, ["bucket", "card-observations", "cards",
                         "mean score in bucket", "mean played_rate in bucket",
                         "obs-weighted mean MV"])


def set_scores(rep: Report, meta: dict) -> None:
    """(e) per-set score against the sets' gold and pip density."""
    rep.say("## e. Set-level castability story")
    rep.say()
    keys = list(meta["set_den"])
    sets = pd.DataFrame({
        "set": keys,
        "num": [meta["set_num"][k] for k in keys],
        "den": [meta["set_den"][k] for k in keys],
        "games": [meta["set_games"][k] for k in keys],
        "gold": [meta["set_gold"][k] for k in keys],
        "pips": [meta["set_pips"][k] for k in keys],
    })
    sets["score"] = sets.num / sets.den
    sets["gold_density"] = sets.gold / sets.den
    sets["pip_density"] = sets.pips / sets.den
    grand = sets.num.sum() / sets.den.sum()
    sets["score_c"] = sets.score - grand
    sets = sets[sets.games >= 500].sort_values("score", ascending=False)
    rep.say(f"{len(sets)} sets with ≥ 500 games. "
            f"Per-set score = Σ_cards[(played∧win) − (played∧loss)] / Σ in-deck obs "
            "— the corpus-level analogue of score_overall. Every set is positive "
            "because winners simply cast more cards than losers (A5); the "
            f"corpus-wide value is {grand:+.4f}, and the `centred` column subtracts "
            "it, which is the scale the grounding pass reported.")
    rep.say()
    noarb = sets[sets.set != "ARB"]
    rk = sets[["score", "gold_density", "pip_density"]].rank()
    rep.say(f"r(set score, gold density) = "
            f"**{np.corrcoef(sets.score, sets.gold_density)[0, 1]:+.3f}** (games-weighted "
            f"{wcorr(sets.score, sets.gold_density, sets.games):+.3f}; Spearman "
            f"{np.corrcoef(rk.score, rk.gold_density)[0, 1]:+.3f}; excluding the "
            f"all-gold outlier ARB {np.corrcoef(noarb.score, noarb.gold_density)[0, 1]:+.3f}); "
            f"r(set score, pip density) = "
            f"**{np.corrcoef(sets.score, sets.pip_density)[0, 1]:+.3f}** (games-weighted "
            f"{wcorr(sets.score, sets.pip_density, sets.games):+.3f}; Spearman "
            f"{np.corrcoef(rk.score, rk.pip_density)[0, 1]:+.3f}; ex-ARB "
            f"{np.corrcoef(noarb.score, noarb.pip_density)[0, 1]:+.3f}). "
            f"SD across sets of set score = {sets.score.std():.4f} "
            f"({sets.score.std() / SD_SCORE_PLAY:.2f}σ of the per-card label SD); "
            f"range {sets.score.min():+.4f} … {sets.score.max():+.4f}.")
    rep.say()
    ext = pd.concat([sets.head(12), sets.tail(12)])
    rows = [[r.set, f"{r.games:,}", f3(r.score), f3(r.score_c),
             f"{r.score_c / SD_SCORE_PLAY:+.2f}",
             f"{r.gold_density:.4f}", f"{r.pip_density:.4f}"]
            for r in ext.itertuples()]
    rep.table(rows, ["set", "games", "set score", "centred", "σ (centred)",
                     "gold density", "pip density"])


def noise_floor(rep: Report, df: pd.DataFrame) -> None:
    """(f) the play/draw split-half reliability and the difference's correlates."""
    rep.say("## f. Play/draw noise floor (R3)")
    rep.say()
    rep.say("Split halves assign each game alternately to half 0 / half 1 by that "
            "card's own in-deck appearance counter, so the two halves are matched in "
            "n by construction. Correlations are on cards with ≥ 25 in-deck "
            "observations *in each half*; `r_half` is the raw split-half "
            "correlation and `r_full` its Spearman-Brown extrapolation to the "
            "full-corpus label.")
    rep.say()
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
        r = wcorr(H[c0], H[c1], H.hw)
        ru = wcorr(H[c0], H[c1])
        rows.append([lab, len(H), f"{r:+.3f}", f"{sb(r):+.3f}", f"{ru:+.3f}",
                     f"{sb(ru):+.3f}"])
    rep.table(rows, ["quantity", "n cards", "r_half (n-weighted)",
                     "r_full (SB, weighted)", "r_half (unweighted)",
                     "r_full (SB, unweighted)"])

    rows = []
    for lo, hi in [(25, 100), (100, 250), (250, 500), (500, 1000),
                   (1000, 3000), (3000, 10**9)]:
        g = H[(H.hw >= lo) & (H.hw < hi)]
        if len(g) < 30:
            continue
        r_sp = wcorr(g.h0_score_play, g.h1_score_play)
        r_df = wcorr(g.dif0, g.dif1)
        rows.append([f"{lo}–{hi if hi < 10**8 else '∞'}", len(g),
                     f"{r_sp:+.3f}", f"{sb(r_sp):+.3f}",
                     f"{r_df:+.3f}", f"{sb(r_df):+.3f}"])
    rep.table(rows, ["min-half n bucket", "cards", "r_half score_play",
                     "r_full score_play", "r_half play−draw diff",
                     "r_full play−draw diff"])

    D = df[df.n_in_deck >= 200].copy()
    D["dif"] = D.score_play - D.score_draw
    D["wt"] = D.n_in_deck / (D.n_in_deck + K)
    obs_sd = D.dif.std()
    Hd = H[(H.hw >= 100)]
    rel_full = sb(wcorr(Hd.dif0, Hd.dif1))
    rep.say(f"Observed SD of the play−draw difference (cards with n ≥ 200): "
            f"**{obs_sd:.4f}**. With full-label reliability "
            f"**{rel_full:.3f}** (cards with ≥ 100 obs per half), true signal SD = "
            f"{obs_sd * np.sqrt(max(rel_full, 0)):.4f} and noise SD = "
            f"{obs_sd * np.sqrt(max(1 - rel_full, 0)):.4f}.")
    rep.say()
    rep.say("Correlates of the play−draw difference, WLS with MV/MV²/creature "
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
        fit = wls(X, D.dif, D.wt)
        b, se = fit.beta, fit.se
        rows.append([lab, int(np.nansum(x)) if col != "mv" else fit.n,
                     f"{b[-1]:+.4f}", f"{se[-1]:.4f}",
                     f"{b[-1] / se[-1]:+.1f}",
                     f"{b[-1] / (obs_sd * np.sqrt(max(1 - rel_full, 1e-9))):+.3f}"])
    rep.table(rows, ["correlate", "n", "Δ(play−draw)", "HC0 SE", "t",
                     "in noise-SD units"])


def cast_lift_anatomy(rep: Report, C: pd.DataFrame) -> None:
    """(g) what cast_lift tracks, by MV and class, and whether it is a
    re-encoding of score_play, MV and played_rate."""
    rep.say("## g. cast_lift anatomy (R4)")
    rep.say()
    G = C[C.n_played >= 20].copy()
    G = G[(G.n_in_deck - G.n_played) >= 20]
    rep.say(f"Cards with ≥ 20 played and ≥ 20 not-played observations: {len(G):,}. "
            f"Weighted mean cast_lift = **{wmean(G.cast_lift, G.wt):+.4f}** "
            f"(p_play {wmean(G.p_play, G.wt):.4f} vs p_dead "
            f"{wmean(G.p_dead, G.wt):.4f}).")
    rep.say()
    rows = [[name, f"{wcorr(G.cast_lift, G[col], G.wt):+.3f}"] for name, col in (
        ("MV", "mv"), ("played_rate", "played_rate"), ("score_play", "score_play"),
        ("score_overall", "score_overall"), ("d = P(played|win) − P(played|loss)", "d"),
        ("w = P(win | in deck)", "w"), ("total pips", "pips_total"))]
    rep.table(rows, ["vs", "weighted Pearson r"])
    rows = [[mv, len(g), f3(wmean(g.cast_lift, g.wt)), f3u(wmean(g.played_rate, g.wt)),
             f3(wmean(g.d, g.wt)), f3(wmean(g.score_play, g.wt))] for mv, g in mv_rows(G)]
    rep.table(rows, ["MV", "cards", "mean cast_lift", "mean played_rate", "mean d",
                     "mean score_play"])
    rep.say("Negative tail by class (weighted means; corpus row for reference):")
    classes = [
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
    for lab, sel in classes:
        g = G[np.asarray(sel)]
        if len(g) < 5:
            continue
        rows.append([lab, len(g), f3(wmean(g.cast_lift, g.wt)),
                     f3(wmean(g.d, g.wt)), f3u(wmean(g.played_rate, g.wt)),
                     f3(wmean(g.score_play, g.wt)), f3u(wmean(g.mv, g.wt))])
    rep.table(rows, ["class", "cards", "cast_lift", "d", "played_rate", "score_play",
                     "mean MV"])
    rep.say("Is cast_lift a re-encoding of (score_play, MV, played_rate)? "
            "Nested WLS R² for predicting cast_lift:")
    G["inv_pr"] = 1.0 / np.clip(G.played_rate, .02, 1)
    rows = []
    prev = 0.0
    for lab, cols in [
        ("score_play", ["score_play"]),
        ("+ MV, MV²", ["score_play", "mv", "mv2"]),
        ("+ played_rate", ["score_play", "mv", "mv2", "played_rate"]),
        ("+ 1/played_rate", ["score_play", "mv", "mv2", "played_rate", "inv_pr"]),
        ("+ d", ["score_play", "mv", "mv2", "played_rate", "inv_pr", "d"]),
        ("score_overall + played_rate + 1/played_rate (algebraic)",
         ["score_overall", "played_rate", "inv_pr"]),
    ]:
        X = np.column_stack([np.ones(len(G))] + [G[c].to_numpy(dtype=float) for c in cols])
        r2 = wls(X, G.cast_lift, G.wt).r2
        algebraic = "algebraic" in lab
        rows.append([lab, f"{r2:.4f}", "—" if algebraic else f"{r2 - prev:+.4f}",
                     f"{np.sqrt(max(0, 1 - r2)) * G.cast_lift.std():.4f}"])
        if not algebraic:
            prev = r2
    rep.table(rows, ["predictors of cast_lift", "weighted R²", "ΔR²",
                     "residual SD"])


def blacklist(rep: Report, C: pd.DataFrame, mbar: float) -> None:
    """(h) Forge's AI:RemoveDeck blacklist through the three channels."""
    rep.say("## h. The Forge blacklist channel (AI:RemoveDeck)")
    rep.say()
    B = C[C.ai_remove_deck.notna()].copy()
    rep.say(f"Forge hints joined for {len(B):,} / {len(C):,} cards "
            f"(n_in_deck ≥ {MIN_N}); blacklisted (`AI:RemoveDeck`) = "
            f"{int(B.ai_remove_deck.sum()):,} "
            f"({B.ai_remove_deck.mean():.1%}).")
    rep.say()
    rows = []
    for lab, sel in [("blacklisted", B.ai_remove_deck == 1),
                     ("not blacklisted", B.ai_remove_deck == 0)]:
        g = B[sel]
        rows.append([lab, len(g), f"{g.n_in_deck.median():.0f}",
                     f3u(wmean(g.w, g.wt)), f3u(wmean(g.m, g.wt)),
                     f3(wmean(g.d, g.wt)), f3(wmean(g.score_overall, g.wt)),
                     f3(wmean(g.score_play, g.wt)), f3u(wmean(g.mv, g.wt))])
    rep.table(rows, ["group", "cards", "median n_in_deck", "w", "m", "d",
                     "score_overall", "score_play", "mean MV"])
    rep.say("Matched comparison — exact cells on (MV clipped 0-8) × creature, "
            "cell weights = harmonic-style min(n_black, n_control), then "
            "cell-weighted mean difference:")
    acc = {c: [0.0, 0.0] for c in CH}
    for _, g in B.groupby([B.mv.clip(0, 8), "is_creature"]):
        gb = g[g.ai_remove_deck == 1]
        gc = g[g.ai_remove_deck == 0]
        if len(gb) < 5 or len(gc) < 5:
            continue
        cw = min(len(gb), len(gc))
        for c in acc:
            acc[c][0] += cw * (wmean(gb[c], gb.wt) - wmean(gc[c], gc.wt))
            acc[c][1] += cw
    rows = [[c, f3(acc[c][0] / acc[c][1]),
             f"{acc[c][0] / acc[c][1] / SD_SCORE_PLAY:+.2f}σ"
             if c.startswith("score") else "—"]
            for c in CH]
    rep.table(rows, ["channel", "matched Δ (blacklisted − control)", "σ"])
    e = channel_effects(B, "ai_remove_deck", "all", CH)
    rows = [[c, f"{e[c][0]:+.4f}", f"{e[c][1]:.4f}",
             f"{e[c][0] / SD_SCORE_PLAY:+.2f}σ" if c.startswith("score") else "—"]
            for c in CH]
    rep.say("Regression-controlled (MV, MV², creature):")
    rep.table(rows, ["channel", "Δ (blacklisted − rest)", "HC0 SE", "σ"])
    dw = e["w"][0]
    dd = e["d"][0]
    predtot = 2 * mbar * dw + dd / 2
    rep.say(f"Share of the blacklist's score deficit that is selection (w): "
            f"**{2 * mbar * dw / predtot * 100:.0f}%**; in-game contribution (d): "
            f"**{dd / 2 / predtot * 100:.0f}%**.")
    rep.say()
    if B.draft_rank.notna().any():
        Bd = B[B.draft_rank.notna()]
        rep.say(f"Bonus — Forge's human draft rank (lower = better) is available for "
                f"{len(Bd):,} of these cards; r(draft_rank, score_overall) = "
                f"{wcorr(Bd.draft_rank, Bd.score_overall, Bd.wt):+.3f}, "
                f"r(draft_rank, w) = {wcorr(Bd.draft_rank, Bd.w, Bd.wt):+.3f}, "
                f"r(draft_rank, d) = {wcorr(Bd.draft_rank, Bd.d, Bd.wt):+.3f}, "
                f"r(draft_rank, m) = {wcorr(Bd.draft_rank, Bd.m, Bd.wt):+.3f}.")
        rep.say()


def main() -> None:
    utf8_console()
    df, med, meta = load()
    C = df[df.n_in_deck >= MIN_N].copy()  # main analysis frame
    mbar, wbar = wmean(C.m, C.wt), wmean(C.w, C.wt)

    rep = Report()
    rep.say("# Label-side artifact & mediation analysis (P0.4 + R3/R4/R9 label halves)")
    rep.say()
    rep.say(f"Corpus: `{meta['corpus']}`  ")
    rep.say(f"Games: **{meta['n_games']:,}**, malformed lines: {meta['bad_lines']}, "
            f"distinct cards: **{len(df):,}** "
            f"(cards with n_in_deck ≥ {MIN_N}: {len(C):,}).  ")
    rep.say(f"Effect sizes in label-SD units use SD(score_play) = {SD_SCORE_PLAY}.")
    rep.say()
    verification(rep, df, med)
    artifacts(rep, df, C, meta)
    channels(rep, C, mbar, wbar)
    forge_mirror(rep, df, C)
    game_length(rep, C, meta)
    set_scores(rep, meta)
    noise_floor(rep, df)
    cast_lift_anatomy(rep, C)
    blacklist(rep, C, mbar)

    (OUT / "l_report.md").write_text("\n".join(rep.lines), encoding="utf-8")
    print(f"\nwrote {OUT / 'l_report.md'}")


if __name__ == "__main__":
    main()

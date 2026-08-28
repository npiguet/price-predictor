"""Assemble ``output/encoder-probes/c_report.md`` from the C-series artifacts.

Also computes the encoder-vs-label divergence columns: every counterfactual
value is put on the same centred scale as the matching label-side WLS
coefficient from ``c7_labelside.csv``, so "what the encoder charges for the
edit" and "what the labels pay the feature" can be subtracted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402

S = cc.SCRATCH


def load(name: str):
    if name.endswith(".json"):
        return json.loads((S / name).read_text(encoding="utf-8"))
    return pd.read_csv(S / name)


def centre(series: pd.Series) -> pd.Series:
    return series - series.mean()


def md(df: pd.DataFrame, cols: dict[str, str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    keys = list(cols)
    lines = ["| " + " | ".join(cols[k] for k in keys) + " |",
             "| " + " | ".join("---" for _ in keys) + " |"]
    for _, r in df[keys].iterrows():
        cells = []
        for i, orig in enumerate(keys):
            v = r.iloc[i]
            f = fmt.get(orig, "{:+.3f}")
            if isinstance(v, str):
                cells.append(v)
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("—")
            else:
                cells.append(f.format(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ci(df: pd.DataFrame, lo="ci_lo", hi="ci_hi") -> pd.Series:
    return df.apply(lambda r: f"[{r[lo]:+.3f}, {r[hi]:+.3f}]", axis=1)


def main() -> None:
    ls = load("c7_labelside.csv")
    ls_json = load("c7_labelside.json")
    out: list[str] = []
    A = out.append

    A("# C-series — the counterfactual content battery (R10-R16)\n")
    A("""Checkpoint `full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d.pt`; harness
`scripts/encoder_probes/probe_lib.py` + `c_common.py`; 27,980 joined primary
cards. Scripts: `c1_keywords.py`, `c1b_interactions.py`, `c2_statlines.py`,
`c3_removal.py`, `c4_body_vs_spell.py`, `c5_types.py`, `c6_spot.py`,
`c7_labelside.py`, `c_report.py`. Roughly 155,000 GPU encodes in total.

Every number is a **paired, layout-matched, same-base-card difference** read
off the fidelity/weighted ridge probes, quoted in label-SD units
(SD(shrunk_score_play) = 0.06181, SD(shrunk_played_rate) = 0.1247). The
design rule comes from R1c: positional edits are a large, negatively biased
null (0.11-0.49 SD for an inert line move) while within-line token
substitutions are clean and mean-zero at ~0.09 SD. So wherever a line has to
be added, the comparison arm adds a line in the same slot and the artifact
cancels in the difference; wherever a substitution will do, nothing moves at
all. Bootstrap CIs are 95%, 1,500-4,000 resamples, clustered on the base card.
The off-manifold gate is the real-card 95th-percentile nearest-neighbour
cosine distance, 0.353.

## How to read the divergence flags

Three quantities recur, and they are not the same thing:

* **encoder-counterfactual** — what the encoder's prediction does when the
  edit is actually made. This is what the C battery measures.
* **label-correlational** — the MV/statline-controlled WLS coefficient of the
  feature on the *labels* (`c7_labelside.csv`). What Forge's eyes actually pay.
* **encoder-correlational** — the identical regression run on the encoder's
  *predictions*. This is how the encoder's outputs are distributed, not how
  they respond.

The encoder-correlational column tracks the label column almost everywhere
(that is the R2 fidelity result restated). Divergences are therefore between
**counterfactual and correlational**: cases where the encoder reproduces a
feature's average association without having learned it as a cause.
""")

    # ────────────────────────────────────────────────────────────────────
    A("\n---\n\n## C1 — the keyword ladder (R10)\n")
    scale = load("c1_scale.csv")
    c1 = load("c1_summary.json")
    dele = load("c1_delete.csv")
    A(f"""**Design A, the ranking.** {c1['n_base_A']} creatures whose only keyword
`static:` line is one of sixteen keywords; that line's body is substituted for
each of the other fifteen ({c1['n_arms_A']:,} encodes). The full 240-cell
ordered pairwise matrix is reduced to a one-dimensional additive scale by
weighted least squares on `delta(K->K') = v(K') - v(K)`.

The reduction is a good fit: **weighted R² = {c1['scale_fit_r2']:.3f}**,
residual SD {c1['scale_fit_resid_sd']:.3f} SD. Only
{100 * c1['off_manifold_A']:.2f}% of the {c1['n_arms_A']:,} edited cards leave
the real-card manifold. The scale is centred (Σv = 0), so a value is the
keyword's worth **relative to the average of the sixteen**.
""")
    sc = scale.copy()
    sc["ci"] = ci(sc)
    sc["pr_ci"] = ci(sc, "pr_ci_lo", "pr_ci_hi")
    A("\n" + md(sc, {
        "keyword": "keyword", "n_carriers": "n", "value_sp": "value (score_play, SD)",
        "ci": "95% CI", "value_pr": "value (played_rate, SD)", "pr_ci": "95% CI",
    }, {"n_carriers": "{:.0f}"}))

    kwmap = ls[ls["family"] == "keyword"].set_index("feature")
    comp = scale.set_index("keyword").join(kwmap[["label_sp", "label_se", "pred_sp", "n"]])
    comp["label_c"] = centre(comp["label_sp"])
    comp["predcorr_c"] = centre(comp["pred_sp"])
    comp["gap"] = comp["value_sp"] - comp["label_c"]
    comp = comp.sort_values("gap")
    comp["flag"] = np.where(comp["gap"].abs() > 0.15, "**yes**", "")
    A(f"""

### Encoder-counterfactual vs label-correlational

The label column is the WLS coefficient of the keyword on `shrunk_score_play`
among creatures, controlling MV, MV², power, toughness and keyword count
(so it is the same "this keyword rather than another" comparison the
substitution makes), re-centred over the same sixteen keywords. `n` is the
label-side carrier count, which is larger than the counterfactual base set
because it does not require the keyword to be the card's only one.
""")
    A("\n" + md(comp.reset_index(), {
        "keyword": "keyword", "value_sp": "encoder-counterfactual",
        "label_c": "label-correlational", "predcorr_c": "encoder-correlational",
        "gap": "gap (cf − label)", "label_se": "label SE", "flag": "flag",
    }, {"label_se": "{:.3f}"}))
    r_cf = comp["value_sp"].corr(comp["label_c"])
    r_corr = comp["predcorr_c"].corr(comp["label_c"])
    A(f"""

Pearson r(encoder-counterfactual, label) = **{r_cf:.2f}**;
r(encoder-correlational, label) = **{r_corr:.2f}**. The encoder's *outputs* are
distributed almost exactly as the labels are; its *response to the edit* is a
noticeably different function of the keyword.

Where they part company:

* **haste** (+0.16 counterfactual vs −0.05 label), **double strike** (+0.14 vs
  −0.03), **reach** (+0.09 vs −0.04) and **menace** (0.00 vs −0.13) are all
  priced above what the labels pay. Aggressive, combat-relevant keywords read
  as good to the encoder whether or not they win games for Forge.
* **hexproof** (−0.24 vs +0.03), **ward {{2}}** (−0.19 vs +0.08) and
  **shroud** (−0.12 vs +0.10) are the three protection keywords, and the
  encoder charges each of them a penalty the labels do not. This is the
  largest coherent divergence family in the battery. Their label SEs are the
  loosest in the table (0.07-0.12, n = 46-152), so treat the individual cells
  as soft; the family-level sign agreement across all three is the finding.
* **flying** lands on the label prior to two decimals (+0.271 vs +0.371
  centred, +0.403 absolute below) — the same calibration R1b found.

`played_rate` runs on its own axis: **haste** is the biggest castability
signal in the set (+0.137 SD), **defender** the biggest negative (−0.103),
and **double strike** — third on the quality scale — is *negative* on
castability (−0.046). These are not the same ordering, which is R3's
low-rank-but-not-identical picture at the level of a single edit.
""")

    A(f"""

### Design B — deleting the keyword line (absolute premium)

The same base cards with the keyword line removed. This is the absolute
premium over a body with no keyword at all, and it carries the line-deletion
artifact, so it is reported but does not drive the ranking. Rank agreement
with Design A: **Spearman {c1['spearman_A_vs_B']:.2f}**; the mean deletion
premium is {c1['mean_delete_premium']:+.3f} SD, which is very nearly the
offset between the two tables.
""")
    dl = dele.copy()
    dl["ci"] = ci(dl)
    A("\n" + md(dl, {
        "keyword": "keyword", "n": "n", "premium_sp": "premium (score_play, SD)",
        "ci": "95% CI", "frac_pos": "% correct direction",
        "premium_pr": "premium (played_rate, SD)", "off_manifold": "off-manifold",
    }, {"n": "{:.0f}", "frac_pos": "{:.1%}", "off_manifold": "{:.3f}"}))
    A(f"""

**The own-keyword bonus.** Summing the two directions of every pair,
`delta(a->b) + delta(b->a)`, gives {c1['antisymmetry_mean']:+.3f} SD rather
than zero. A keyword is worth systematically more on the cards that actually
carry it than on cards that carry a different one — about 0.029 SD per
direction. The encoder has learned that the *rest* of a real flier is
flier-shaped, which is the same distributional asymmetry R1b saw as the 0.54
add/remove ratio, measured here without any line moving.
""")

    # ── C1 interactions ─────────────────────────────────────────────────
    A("\n### Interaction (i) — flying x body size\n")
    fs_sub = load("c1b_flying_size_sub.csv")
    fs_add = load("c1b_flying_size_add.csv")
    c1b = load("c1b_summary.json")
    fs_sub["ci"] = ci(fs_sub)
    fs_add["ci"] = ci(fs_add)
    A("""Two independent readings of the same premium. *Substitution* replaces the
existing single keyword with `flying` vs `vigilance` on the C1 base cards
(nothing moves). *Addition* adds `static: flying` vs `static: vigilance` to
keywordless creatures (both arms add one line in the same slot).
""")
    A("\n**Substitution (n = 2,857 solo-keyword creatures)**\n")
    A(md(fs_sub, {"bucket": "P+T", "n": "n",
                  "flying_minus_vigilance_sp": "flying − vigilance (SD)",
                  "ci": "95% CI", "frac_pos": "% positive"},
         {"n": "{:.0f}", "frac_pos": "{:.1%}"}))
    A("\n**Addition (n = 1,200 keywordless creatures)**\n")
    A(md(fs_add, {"bucket": "P+T", "n": "n", "mean_PT_total": "mean P+T",
                  "flying_minus_vigilance_sp": "flying − vigilance (SD)",
                  "ci": "95% CI"},
         {"n": "{:.0f}", "mean_PT_total": "{:.2f}"}))
    addk = load("c1b_add_keyword.csv")
    addk["ci"] = ci(addk)
    addk["nci"] = ci(addk, "nl_ci_lo", "nl_ci_hi")
    A("""
The add design also gives the six keywords' absolute add-premium on a
keywordless body, with `vigilance` as the layout-matched control and the
un-edited card as the (artifact-carrying) zero:
""")
    A("\n" + md(addk, {"keyword": "added keyword", "n": "n",
                       "vs_vigilance_sp": "vs adding vigilance (SD)", "ci": "95% CI",
                       "vs_no_line_sp": "vs adding no line (SD)", "nci": "95% CI",
                       "vs_vigilance_pr": "Δ played_rate vs vigilance",
                       "off_manifold": "off-manifold"},
                {"n": "{:.0f}", "off_manifold": "{:.3f}"}))
    A("""
The line-add artifact for a `static:` line on a creature is small and
*positive* (+0.022 SD for the control keyword), unlike the `triggered:`
line-add artifact C6 measures (−0.063 SD) — one more reason to difference
rather than to read an add-arm raw.
""")
    A(f"""
**The interaction is an inverted U, not a slope.** Both designs agree: the
flying premium climbs from ~0.21-0.26 SD on tiny bodies to a peak of
0.35-0.37 SD around P+T = 7-8, then *falls back* to 0.27-0.28 SD on the
biggest creatures. A straight line through the substitution data has slope
{c1b['flying_vs_vigilance_slope_per_PT_point']:+.4f} SD per point of P+T —
essentially zero, because the two halves of the curve cancel.

This contradicts the label prior, which is a monotone **+0.045 SD per point
of P+T**. Over the observed range that prior predicts a premium roughly three
times larger on a 6/6 than on a 1/1; the encoder's is 1.4x larger and then
shrinks again. **Flagged divergence.** The most likely reading is a
distributional one: 9/9 fliers barely exist, so the encoder has no
well-populated neighbourhood in which a huge flier is priced as such.
""")

    A("\n### Interaction (ii) — deathtouch x trample\n")
    tt = load("c1b_dt_trample.csv")
    tt["ci"] = ci(tt)
    A(f"""A 2x2 in which *every* arm adds exactly two `static:` lines in the same two
slots, on 400 keywordless creatures with P+T in [3,10] and at most one other
ability line. Slot 1 holds `vigilance` or `deathtouch`, slot 2 holds `reach`
or `trample`; the interaction term cancels both controls and the layout
artifact at once. Off-manifold across all four cells:
{100 * c1b['dt_trample_off_manifold']:.1f}%.
""")
    A("\n" + md(tt, {"term": "term", "n": "n", "mean_sd": "mean Δ (SD)",
                     "ci": "95% CI", "frac_pos": "% positive"},
                {"n": "{:.0f}", "frac_pos": "{:.1%}"}))
    A("""
**The encoder knows the combo.** Deathtouch alone is worth +0.202 SD over the
control keyword and trample alone −0.139 SD (trample is *below* `reach` on
the C1 scale, so this sign is internally consistent). Their sum is +0.063 SD,
but the joint cell is **+0.203 SD** — an interaction of **+0.139 SD
[+0.127, +0.152]**, positive on 91% of base cards. Deathtouch-plus-trample is
one of Magic's canonical two-keyword synergies and only four real cards in the
corpus carry both, so this is an encoder-only claim: the model has generalized
a combination it has almost no direct evidence for.
""")

    A("\n### Interaction (iii) — the keyword-count ladder\n")
    lad = load("c1b_ladder.csv")
    lad["ci"] = ci(lad)
    A("""There is no neutral `static:` line to control an added keyword against, so
this rung ladder is reported as raw marginals against the previous rung, on
500 keywordless creatures, under two orderings.
""")
    A("\n" + md(lad, {"order": "order", "rung": "rung", "added": "added keyword",
                      "marginal_sp": "marginal (SD)", "ci": "95% CI",
                      "cumulative_sp": "cumulative (SD)",
                      "off_manifold": "off-manifold"},
                {"rung": "{:.0f}", "off_manifold": "{:.3f}"}))
    A("""
**No saturation.** The label prior is a saturating ladder (+0.35 / +0.28 /
+0.13 for the first three keywords). The encoder's strong-first ladder runs
+0.36 / +0.20 / +0.16 / +0.29 and its weak-first ladder +0.06 / +0.05 / +0.13
/ +0.07 — in neither ordering does the marginal decay, and the strong-first
fourth rung is the *second* largest. Cumulatively a four-keyword creature is
+1.01 SD over its keywordless self. **Flagged divergence**: the encoder adds
keyword values roughly linearly where the labels saturate.

Rungs 3-4 are a mechanism demo, not an in-distribution claim: off-manifold
fraction rises from 0.2% at rung 1 to 7.8% (strong-first) and 14.0%
(weak-first) at rung 4, and four-keyword creatures are rare in the corpus.
""")

    # ────────────────────────────────────────────────────────────────────
    A("\n---\n\n## C2 — statlines and costs (R11)\n")
    A("Nothing in C2 adds or removes a line; every arm edits digits inside an "
      "existing `power toughness:` or `mana cost:` line.\n")

    A("\n### (i) Power vs toughness at fixed total\n")
    pt = load("c2_pt_asymmetry.csv")
    pts = load("c2_pt_by_size.csv")
    c2 = load("c2_summary.json")
    pt["ci"] = ci(pt)
    pts["ci"] = ci(pts)
    A("\n" + md(pt, {"contrast": "contrast", "n": "n", "mean_sd": "mean Δ (SD)",
                     "ci": "95% CI", "frac_pos": "% positive"},
                {"n": "{:.0f}", "frac_pos": "{:.1%}"}))
    A("\n" + md(pts, {"P+T": "P+T", "n": "n",
                      "gradient_per_point": "gradient per point of P−T (SD)",
                      "ci": "95% CI"}, {"n": "{:.0f}"}))
    lab_g = ls_json["pt_gradient_label_sp"]
    prd_g = ls_json["pt_gradient_pred_sp"]
    A(f"""
900 creatures with P+T in [4,8], swapped to (P+1, T−1) and (P−1, T+1).
The causal gradient is **+0.038 SD per point of P−T [+0.035, +0.041]**,
correct-signed on 78% of cards.

The move is strongly asymmetric: taking a point off power costs −0.116 SD
while adding one gains only +0.037 SD, so the encoder is much more sure that
*low* power is bad than that high power is good.

Against the label side: the same P−T gradient fitted on the labels (controlling
P+T, MV, MV², keyword count, same P+T window) is
**{lab_g['beta_per_point_of_P_minus_T']:+.3f} ± {lab_g['se']:.3f}**, and on the
encoder's predictions {prd_g['beta_per_point_of_P_minus_T']:+.3f}. The
counterfactual is 12% below the label coefficient — **agreement**, and a
correction to the v1 prior of +0.11 SD/point, which was fitted without a P+T
control and so partly measured "bigger is better".

The size interaction survives: **+0.061 SD/point at P+T 4-5** falling to
**+0.011 at P+T 6** and **+0.014 at P+T 7-8**. Power is priced on small
statlines and free on large ones, exactly as the MTG-domain prior said.
`played_rate` moves in the same direction but a third as far
({c2['pt_asymmetry_pr']['gradient_per_point']:+.3f} SD/point).
""")

    A("\n### (ii) The N/N integer sweep — where monotonicity breaks\n")
    nn = load("c2_nn_sweep.csv")
    nn["ci"] = ci(nn)
    nn["pr_delta"] = nn["played_rate_sd"] - nn["played_rate_sd"].iloc[0]
    A("""40 real vanilla creatures (no ability lines, plain costs) with their
statline overwritten as `power toughness: N/N` for N = 0…12. The mana cost is
held at the base card's own, so from N ≈ 6 upward this is a **mechanism
demo**: the statline no longer matches the cost.
""")
    A("\n" + md(nn, {"N": "N/N", "score_play_sd": "score_play (SD)", "ci": "95% CI",
                     "marginal": "marginal", "pr_delta": "Δ played_rate vs 0/0 (SD)",
                     "off_manifold": "off-manifold"},
                {"N": "{:.0f}", "off_manifold": "{:.3f}"}))
    A(f"""
**The encoder does not read integers as magnitudes.** The sweep is
non-monotone at N = {', '.join(str(x) for x in c2['nn_monotone_breaks'])} — eight
of the twelve steps go the wrong way. Three specific failures:

1. **1/1 is scored below 0/0** (−0.090 vs −0.037).
2. **7/7 is scored below 4/4, 5/5, 6/6 and 8/8** (+0.252 vs +0.507 / +0.474 /
   +0.482 / +0.543) — a 0.29 SD hole in the middle of the range.
3. **The curve turns over after 8/8 and collapses**: 12/12 (+0.037) is scored
   essentially the same as **0/0** (−0.037), and below 2/2.

Note that these are *not* off-manifold artifacts — the gate fires on 0-2.5% of
these cards, because a 12/12 body still sits near the corpus's real fatties.
The integer tokens simply carry no ordinal structure past the range where they
are common; `8` beats `7` because the cards printed with an 8 are better, not
because 8 > 7. This is the sharpest single demonstration in the battery that
the encoder is a text classifier and not a card evaluator, and it is the
directly actionable one for scoring hypothetical cards: **do not trust a
statline outside roughly 2/2-8/8**.

`played_rate` is better behaved and monotone downward from N = 4 (a bigger
body reads as harder to cast at a fixed cost), which is the correct sign.
""")

    A("\n### (iii) The generic-cost sweep and the {X} question\n")
    mv = load("c2_mv_sweep.csv")
    mv["ci"] = ci(mv)
    mv["pr_delta"] = mv["played_rate_sd"] - mv["played_rate_sd"].iloc[0]
    A("The same 40 bodies with the whole mana cost overwritten as `{k}` for "
      "k = 0…9 (also a mechanism demo — the body no longer matches the cost).\n")
    A("\n" + md(mv, {"cost": "cost", "score_play_sd": "score_play (SD)", "ci": "95% CI",
                     "marginal": "marginal", "pr_delta": "Δ played_rate vs {0} (SD)",
                     "off_manifold": "off-manifold"},
                {"off_manifold": "{:.3f}"}))
    A("""
Monotone and roughly linear from `{0}` to `{4}` (+0.109 SD per generic point,
close to the label-side MV slope of +0.15 SD/point), then it turns over: `{6}`
through `{9}` all land back at or below the `{2}` level. `played_rate` falls
essentially monotonically throughout (one 0.03 SD reversal at `{2}`), which is
right. So the encoder's cost channel is a
mana-development-certification effect that only exists in the range where a
statline-cost pair is realistic; past MV 5 the mismatch dominates and the
score reverts. Off-manifold fraction is high at the cheap end (40% at `{0}`,
37.5% at `{1}`) — a 4/4 for `{0}` is genuinely not a card — so read the top
rows as extrapolation.
""")
    xc = load("c2_x_cost.csv")
    xc["ci"] = ci(xc)
    A("\n**{X}, on 250 real noncreature X-spells and 250 non-X spells**\n")
    A(md(xc, {"contrast": "contrast", "n": "n", "mean_sd": "mean Δ (SD)",
              "ci": "95% CI", "frac_pos": "% positive",
              "delta_pr": "Δ played_rate (SD)", "off_manifold": "off-manifold"},
         {"n": "{:.0f}", "frac_pos": "{:.1%}", "off_manifold": "{:.3f}"}))
    A("""
**{X} reads as a cheap generic, not as its own thing.** Substituting `{X}` for
`{1}` is a null (−0.013); for `{3}` it gains +0.067 and for `{6}` +0.141 — the
same shape as the generic sweep, so the encoder is placing `{X}` at roughly
`{1}`-to-`{2}` on the cost axis. Deleting `{X}` outright is also a null
(−0.034 [−0.081, +0.015]). And the reverse edit — turning a real generic pip
into `{X}` on non-X spells — moves nothing at all (+0.006 [−0.022, +0.033]).

The `played_rate` column is where `{X}` behaves like a real cost: `{X}` → `{6}`
costs −0.345 SD of castability while `{X}` → `{1}` costs almost nothing. So the
encoder has learned that an X-spell is castable early, and has *not* learned
that it scales.
""")

    A("\n### (iv) Pips — the colour fee, causally\n")
    pips = load("c2_pips.csv")
    pips["ci"] = ci(pips)
    A(md(pips, {"base": "base cost", "arm": "→ arm", "n": "n",
                "mean_sd": "Δ score_play (SD)", "ci": "95% CI",
                "frac_pos": "% positive", "delta_pr": "Δ played_rate (SD)",
                "off_manifold": "off-manifold"},
         {"n": "{:.0f}", "frac_pos": "{:.1%}", "off_manifold": "{:.3f}"}))
    A("""
Two clean, opposite effects, replicated across all five colours:

* **Intensifying colour is rewarded on quality and punished on castability.**
  `{1}{W}` → `{W}{W}` gains +0.100 SD of score_play but loses **−0.369 SD** of
  played_rate; `{2}{W}` → `{1}{W}{W}` gains +0.194 and loses −0.280. Red is the
  most rewarded (+0.156), blue the least (+0.063).
* **Going colourless is a large penalty.** `{1}{W}` → `{2}` costs **−0.297 SD**
  and `{1}{G}` → `{2}` **−0.241 SD**, while *gaining* castability (+0.059,
  +0.156). The fee is smallest for red and blue (−0.093, −0.116).

This is the scorer study's colour fee, measured causally on the encoder for
the first time, and it separates cleanly into the two heads: pips buy score
and cost castability, colourlessness does the reverse. It also explains the
mana-rock result in C4 — a colourless artifact is paying this fee before it
says anything at all.
""")

    # ────────────────────────────────────────────────────────────────────
    A("\n---\n\n## C3 — the removal / spell-effect ladder (R12)\n")
    lad3 = load("c3_spell_ladder.csv")
    c3 = load("c3_summary.json")
    lad3["ci"] = ci(lad3)
    lad3["rci"] = ci(lad3, "ref_ci_lo", "ref_ci_hi")
    A(f"""200 real single-line instants and sorceries of MV 1-5 have the *content* of
their `spell[1]:` line replaced by each of fifteen templates. Every arm is one
line in the same slot on the same base card, so all 105 pairwise contrasts are
layout-matched and the base card's memorized offset cancels. `value` is
centred on the fifteen-arm mean. Off-manifold across the whole family:
{100 * c3['off_manifold_ladder']:.1f}%.
""")
    A("\n" + md(lad3, {"effect": "effect text", "n": "n", "value_sp": "value (SD)",
                       "ci": "95% CI", "vs_destroy_creature": "vs destroy target creature",
                       "rci": "95% CI", "value_pr": "value (played_rate, SD)",
                       "off_manifold": "off-manifold"},
                {"n": "{:.0f}", "off_manifold": "{:.3f}"}))
    A("""
The ladder spans **1.84 SD** end to end — an order of magnitude more than the
keyword ladder's 0.51 SD. Text on a spell is where the encoder's opinions live.

Reading it:

* **Burn beats removal.** `deals 3 damage to any target` (+0.62) outranks
  `destroy target creature` (+0.29) by +0.32 SD [+0.28, +0.36]. Face damage is
  the single most valuable spell text in the set. The label side agrees
  (+0.547 vs +0.359 in the WLS table) — this one is not a divergence, it is
  the encoder reproducing a real Forge fact.
* **Restricting a removal spell is priced correctly and steeply.**
  `destroy target creature` → `with power 4 or greater` −0.169, → `with flying`
  −0.339. Conditional removal loses about a third of a SD. Note that
  `with flying` costs *more* than `power 4 or greater`, which is the right
  ordering for a sealed pool.
* **Exile > destroy** (+0.078 [+0.058, +0.099]), a small but clean effect that
  no keyword-level feature could produce.
* **Fight is not discounted.** `target creature fights…` sits second at +0.41,
  *above* `destroy target creature`, matching the label prior (fight +0.462 in
  the WLS table) and confirming that the scorer's dislike of fight effects is a
  search-level phenomenon, not a label or encoder one.
* **Sweepers are the second-worst text in the ladder** (−0.56, i.e. −0.85 vs
  spot removal), agreeing with the labels (−0.372) and with the A4 mechanism
  running the other way.
* **`tap target creature` (−0.40) and `counter target spell` (−0.14)** are both
  well below spot removal, both matching the label signs.
* **Card draw is priced at par with doing nothing to the board.**
  `draw two cards` (−0.045) sits below `return target creature to its owner's
  hand` and barely above `destroy target creature with flying`. Label side:
  −0.153. Agreement.

**The one large divergence: lifegain.** `you gain 4 life.` is
**−1.22 SD**, dead last by 0.67 SD, and 1.51 SD below `destroy target
creature`. The label-side coefficient for lifegain spells is
**−0.034 ± 0.062 — a null**, and the l_report three-channel table has lifegain
at *+0.15σ*. The encoder is charging more than a full label SD for a text the
labels treat as neutral-to-slightly-good. This is the largest
encoder-vs-label divergence in the whole C battery. **Flagged.**
""")
    au = load("c3_aura.csv")
    au["ci"] = ci(au)
    A("\n### The aura family\n")
    A(f"""The same design on 200 real auras: the second `static:` line (the first is
`enchant creature`) has its content replaced. Reference arm is
`enchanted creature gets +2/+2.`; off-manifold
{100 * c3['off_manifold_aura']:.1f}%.
""")
    A("\n" + md(au, {"aura_text": "aura text", "n": "n",
                     "vs_plus2_sp": "vs +2/+2 (SD)", "ci": "95% CI",
                     "frac_pos": "% positive", "vs_plus2_pr": "Δ played_rate (SD)",
                     "off_manifold": "off-manifold"},
                {"n": "{:.0f}", "frac_pos": "{:.1%}", "off_manifold": "{:.3f}"}))
    A("""
**Lockdown is the top of the noncreature text ladder, as the label prior said**
(+0.40 to +0.50 there; +0.33 SD over a pump aura here, and it is the only arm
above the pump reference). The internal ordering is the interesting part:
`can't attack or block` (+0.33) ≫ `can't block` (−0.44), a **0.77 SD** gap for
adding two words. The encoder reads *how much* of the creature the aura turns
off, which is a genuinely compositional distinction.

`gets -2/-2` sits at par with `gets +2/+2` (−0.040 [−0.084, +0.005]) — the
R1c finding that the sign channel is thin, reproduced here on a family where
the sign is the whole card.
""")

    # ────────────────────────────────────────────────────────────────────
    A("\n---\n\n## C4 — body vs spell (R14)\n")
    c4 = load("c4_summary.json")
    anch = load("c4_class_anchors.csv")
    agg = load("c4_body_premium_agg.csv")
    rock = load("c4_rock_dork.csv")
    A(f"""**This section is a mechanism probe**: the shells are synthetic. They are
however unusually well-behaved as synthetic cards — median nearest-real-card
cosine distance **{c4['shell_dist_p50']:.3f}** (corpus median for a real card
is 0.170), p95 {c4['shell_dist_p95']:.3f}, and only
{100 * c4['shell_off_manifold']:.1f}% off-manifold. Several land on a real card
exactly (the `{{2}}{{G}}` instant `destroy target artifact.` reproduces
*Verdigris* bit for bit).

Eighteen effect texts are placed on four matched shells at two costs
(`{{2}}{{c}}`, `{{4}}{{c}}`), colour held fixed within an effect:
a sorcery, an instant, a 2/2 creature's ETB trigger, and the bare 2/2.

### Real-card anchors (MV 1-4), for calibration
""")
    A("\n" + md(anch, {"class": "class", "n": "n", "mean_mv": "mean MV",
                       "pred_sp": "predicted score_play (SD)",
                       "label_sp": "label score_play (SD)",
                       "pred_pr": "predicted played_rate (SD)"},
                {"n": "{:.0f}", "mean_mv": "{:.2f}"}))
    A("""
The predicted and label columns agree to within 0.12 SD on every class, which
is what licenses reading the synthetic shells against these anchors. Note the
class ordering the labels give and the encoder reproduces: **token creature
(+0.67) > vanilla creature (+0.35) > token spell (+0.16) > mana dork (+0.08) >
ramp sorcery (−0.37) > mana rock (−0.69)**.
""")
    A(f"""
### The body premium

Mean over all 18 effects x 2 costs: **{c4['body_premium_mean_sd']:+.3f} SD**
[{c4['body_premium_ci'][0]:+.3f}, {c4['body_premium_ci'][1]:+.3f}], positive on
{100 * c4['body_premium_frac_pos']:.0f}% of effect-cost cells. Putting an effect
on a 2/2 body instead of a sorcery is worth about six tenths of a label SD.
""")
    A("\n" + md(agg, {
        "effect": "effect text",
        "body_premium_etb_minus_sorcery": "body premium (ETB − sorcery)",
        "effect_on_body_etb_minus_bare": "effect's marginal on a body (ETB − bare 2/2)",
        "effect_on_spell_vs_control": "effect's marginal on a spell (vs gain-1-life)",
        "instant_minus_sorcery": "instant − sorcery",
    }))
    A(f"""
The body premium is **largest for the weakest effects** and **negative for the
strongest**: `you gain 7 life.` gains +1.49 SD from being stapled to a 2/2,
while `destroy target creature` *loses* 0.18 SD and `deals 2 damage to any
target` loses 0.17. The two columns explain why — an effect's marginal on a
*spell* runs from +0.41 (gain 7 life) to +2.14 (deal 2 damage / two tokens),
while its marginal on a *body* runs only −0.78 to +0.98. A creature's score is
mostly its body; a spell's score is entirely its text. So the body is a floor:
it rescues a weak effect and it caps a strong one.

`instant` over `sorcery` on the identical text is
{c4['instant_minus_sorcery_mean_sd']:+.3f} SD
[{c4['instant_minus_sorcery_ci'][0]:+.3f}, {c4['instant_minus_sorcery_ci'][1]:+.3f}]
— small, positive, and consistent with the label-side coefficient
({ls[ls.feature == 'instant (vs sorcery)']['label_sp'].iloc[0]:+.3f}).

### Mana rocks vs mana dorks
""")
    A("\n" + md(rock, {"shell": "shell", "score_play_sd": "score_play (SD)",
                       "played_rate_sd": "played_rate (SD)", "dist": "dist to nearest real",
                       "nearest": "nearest real card"},
                {"dist": "{:.3f}", "played_rate_sd": "{:+.2f}"}))
    A("\n(`played_rate` here is an absolute predicted level in SD units, not a "
      "difference; the corpus mean sits at about +2.64 on this scale.)\n")
    A("""
Holding the ability, the cost and everything else fixed, moving `{T}: add {C}.`
from an artifact onto a creature body is worth **+0.333 SD** (0.022 → 0.354).
An enchantment shell scores the same as the artifact (+0.006), and an
*artifact creature* scores the same as the artifact too (+0.024) — so it is
not the word "artifact" that is being punished, it is the absence of a body
that a plain creature type-line supplies. Making the mana coloured
(`add {G}` instead of `add {C}`) costs 0.18 SD, the colour fee from C2(iv)
arriving from the other side.

Against the anchors: the synthetic dork (+0.35) sits well above the real dork
class mean (+0.08) and the synthetic rock (+0.02) well above the real rock
class mean (−0.69). Real rocks are worse than a minimal rock shell — their
extra text (bigger costs, ETB-tapped clauses, activated sinks) makes them
worse still, and the population effect the l_report measured (−0.79σ) is about
three times the pure shell effect. The **direction** of the body-vs-spell
master axis, however, is exactly reproduced: dork > rock, token creature >
token spell, and body-mounted ramp > ramp sorcery (−0.87 for the ramp sorcery
shell).
""")

    # ────────────────────────────────────────────────────────────────────
    A("\n---\n\n## C5 — types and flavour (R16)\n")
    tri = load("c5_tribe_scale.csv")
    c5 = load("c5_summary.json")
    tri["ci"] = ci(tri)
    trimap = ls[ls["family"] == "tribe"].set_index("feature")
    tc = tri.set_index("tribe").join(trimap[["label_sp", "label_se", "pred_sp"]])
    tc["label_c"] = centre(tc["label_sp"])
    tc["gap"] = tc["value_sp"] - tc["label_c"]
    tc = tc.sort_values("value_sp", ascending=False)
    A(f"""### (i) The tribal-noun scale

Creatures whose types line carries exactly one subtype, and that subtype is
one of seventeen tribes, have the noun substituted for each of the others —
the same pairwise-matrix-to-additive-scale reduction as C1, on
{int(tri['n_carriers'].sum()):,} base cards. Nothing but the noun changes: cost,
statline, abilities and every other word are held fixed. Off-manifold
{100 * c5['tribe_off_manifold']:.2f}%. The label column is the same tribe's WLS
coefficient among creatures controlling MV, MV², power, toughness and keyword
count, re-centred over the same seventeen.
""")
    A("\n" + md(tc.reset_index(), {
        "tribe": "tribe", "n_carriers": "n", "corpus_freq": "corpus freq",
        "value_sp": "encoder-counterfactual (SD)", "ci": "95% CI",
        "label_c": "label-correlational (SD)", "gap": "gap",
    }, {"n_carriers": "{:.0f}", "corpus_freq": "{:.0f}"}))
    rt = tc["value_sp"].corr(tc["label_c"], method="spearman")
    A(f"""
**A tribal noun is worth real score at fixed everything else.** The scale spans
**{c5['tribe_spread_sd']:.3f} SD** from angel (+0.131) to lizard (−0.183), and the
headline substitutions are decisive: `dragon` → `lizard` costs
**{c5['dragon->lizard']['mean_sd']:+.3f} SD**
[{c5['dragon->lizard']['ci'][0]:+.3f}, {c5['dragon->lizard']['ci'][1]:+.3f}],
correct-signed on {100 * c5['dragon->lizard']['frac_neg']:.0f}% of the 224 dragons;
the reverse upgrade `human` → `dragon` on plain humans gains
**{c5['human->dragon']['mean_sd']:+.3f} SD**; `angel` → `bird` costs
{c5['angel->bird']['mean_sd']:+.3f}.

Against the label side, two things are true at once. The **rank order agrees**
(Spearman {rt:.2f}): angel, hydra, sphinx and dragon at the top, lizard,
goblin, wall and beast at the bottom, in both columns. But the **magnitude is
about a third**: the label spread over these tribes is 0.94 SD, the encoder's
counterfactual spread 0.31 SD. So the v1 prior — "the dragon premium is
essentially all statline" — is **two-thirds right and one-third wrong**. Note
the label column here already controls power and toughness, so the residual it
reports is *not* statline; the missing two thirds is whatever else correlates
with being an angel (rarity, cost structure, the rest of the text) and which
the encoder correctly refuses to transfer to a re-typed common.

Corpus frequency does not explain the scale (`human` at 4,400 carriers sits
mid-table, `hydra` at 70 sits near the top), so this is not a
frequency-of-token effect.
""")

    A("\n### (ii) Type-line swaps\n")
    tsw = load("c5_type_swaps.csv")
    tsw["ci"] = ci(tsw)
    A(md(tsw, {"contrast": "contrast", "n": "n", "mean_sd": "mean Δ (SD)",
               "ci": "95% CI", "frac_pos": "% positive",
               "delta_pr": "Δ played_rate (SD)", "off_manifold": "off-manifold"},
         {"n": "{:.0f}", "frac_pos": "{:.1%}", "off_manifold": "{:.3f}"}))
    art_lab = ls[ls.feature == "artifact creature (vs creature)"]["label_sp"].iloc[0]
    A(f"""
**Flash speed is not in the type token.** `instant` → `sorcery` on 400
single-line instants costs only −0.014 SD [−0.022, −0.007] and the reverse
gains +0.007 [−0.001, +0.015]. Both are an order of magnitude below the
label-side instant coefficient
({ls[ls.feature == 'instant (vs sorcery)']['label_sp'].iloc[0]:+.3f}) and two
orders below the `flash` keyword penalty C1 measures on creatures (−0.148). The
encoder has learned that instants and sorceries *say different things*, not
that one token makes a card faster.

**The artifact penalty is not in the type token either — and its sign flips.**
Prepending `artifact` to a creature's type line **gains +0.068 SD**
[+0.030, +0.105], while the label-side coefficient for artifact creatures is
**{art_lab:+.3f}** and the l_report calls artifact the most negative type. The
encoder's own *correlational* coefficient reproduces the label
({ls[ls.feature == 'artifact creature (vs creature)']['pred_sp'].iloc[0]:+.3f})
almost exactly, so this is a clean counterfactual-vs-correlational split:
the encoder knows artifact creatures are worse **without having attached that
knowledge to the word "artifact"**. **Flagged divergence.** (Prepending
`enchantment` gains more, +0.124, against a label coefficient of +0.106 — that
one agrees.)
""")

    A("\n### (iii) Taplands\n")
    lands = load("c5_taplands.csv")
    lands["ci"] = ci(lands)
    A(md(lands, {"contrast": "contrast", "n": "n", "mean_sd": "mean Δ (SD)",
                 "ci": "95% CI", "frac_pos": "% positive",
                 "delta_pr": "Δ played_rate (SD)", "off_manifold": "off-manifold"},
         {"n": "{:.0f}", "frac_pos": "{:.1%}", "off_manifold": "{:.3f}"}))
    tl = ls_json["tapland_label_sp"]
    A(f"""
332 lands carrying exactly `replacement: CARDNAME enters tapped.`. The clean
edit is the one-token substitution `tapped` → `untapped`, which **costs**
−0.054 SD [−0.062, −0.046]; deleting the line outright costs the same
(−0.055). Both directions say the encoder prefers the tapland.

This confirms the prior. The label side agrees in sign and magnitude: among
lands, the enters-tapped flag is worth
**{tl['beta']:+.3f} ± {tl['se']:.3f}** on `shrunk_score_play`. In a sealed pool
built by Forge, a land that fixes colours and enters tapped beats a land that
does neither, and the encoder has learned it as a *causal* property of the
clause, not just an association.
""")

    # ────────────────────────────────────────────────────────────────────
    A("\n---\n\n## C6 — riders and trigger timing (R15)\n")
    rid = load("c6_riders.csv")
    rid["ci"] = ci(rid)
    rid["cci"] = ci(rid, "c_ci_lo", "c_ci_hi")
    A("""### (i) The cantrip rider — and a wordiness artifact

500 real single-line instants and sorceries, with a rider **appended inside**
the existing `spell[1]:` line. No line is added, so all five arms are exactly
layout-matched and the correct read is the `vs control rider` column.
""")
    A("\n" + md(rid, {"rider": "appended rider", "n": "n",
                      "vs_no_rider_sd": "vs no rider (SD)", "ci": "95% CI",
                      "vs_control_rider_sd": "vs control rider (SD)", "cci": "95% CI",
                      "delta_pr": "Δ played_rate (SD)"}, {"n": "{:.0f}"}))
    A("""
**The cantrip rider is worth exactly nothing**: `draw a card.` against
`you gain 2 life.` is **−0.001 SD [−0.018, +0.016]**, one of the tightest nulls
in the battery, and `draw two cards.` is +0.006 [−0.010, +0.023]. This confirms
the v1 prior (cantrip riders −0.04, no bonus) with a causal design.

But the `vs no rider` column contains a finding of its own. **Every** rider is
worth about the same +0.09 SD regardless of what it says — gain 2 life +0.093,
draw a card +0.092, draw two cards +0.099, and *lose 2 life* +0.095. Only
`scry 1.` differs (+0.033). Appending a clause to a spell line is worth roughly
a tenth of a label SD **for being there**, and the encoder does not read the
sign of the clause it appended: a strict drawback pays the same premium as a
strict upside. This is the R1c thin-negation channel showing up as a pure
wordiness artifact, and it is a direct warning for hypothetical cards: adding
text to a spell raises its score whatever the text says.

### (iii) The drawback rider on a creature
""")
    trg = load("c6_end_triggers.csv")
    trg["ci"] = ci(trg)
    trg["cci"] = ci(trg, "c_ci_lo", "c_ci_hi")
    A(md(trg, {"added_trigger": "added `triggered:` line", "n": "n",
               "vs_no_line_sd": "vs no line (SD)", "ci": "95% CI",
               "vs_control_line_sd": "vs control line (SD)", "cci": "95% CI",
               "delta_pr": "Δ played_rate (SD)", "off_manifold": "off-manifold"},
         {"n": "{:.0f}", "off_manifold": "{:.3f}"}))
    sac = ls_json["sacrifice_self_label_sp"]
    A(f"""
500 creatures with at most one ability line, each arm adding exactly one
`triggered: at the beginning of your end step, …` line in the same slot, so
the `vs control line` column is the priced content of the clause.

**On a creature, the sign channel works.** `sacrifice CARDNAME.` costs
**−0.202 SD** against the control clause and `you lose 1 life.` −0.127, while
`draw a card.` gains +0.106 and `put a +1/+1 counter on CARDNAME.` +0.197.
That is a 0.40 SD spread between a self-sacrifice and a growth trigger written
into the same slot — an order of magnitude more sign sensitivity than the
spell-line riders above showed.

The label side agrees on the self-sacrifice: among creatures, controlling MV
and statline, `ph_sacrifice_self` is worth
**{sac['beta']:+.3f} ± {sac['se']:.3f}**. Against the v1 prior that "drawbacks
are ≈ unpunished" this is a **correction rather than a divergence** — the
prior came from `echo`, a cost-style drawback, and a sacrifice clause is not
one.

Note also that the control clause itself is worth −0.063 SD against no line at
all: the line-add artifact for a `triggered:` line on a creature is *negative*,
which is why it must be differenced out and why the raw `vs no line` column
would have made every arm look worse than it is.

### (iv) Trigger timing
""")
    tim = load("c6_timing.csv")
    tim["ci"] = ci(tim)
    A(md(tim, {"contrast": "contrast", "n": "n", "mean_sd": "mean Δ (SD)",
               "ci": "95% CI", "frac_pos": "% positive",
               "delta_pr": "Δ played_rate (SD)", "off_manifold": "off-manifold"},
         {"n": "{:.0f}", "frac_pos": "{:.1%}", "off_manifold": "{:.3f}"}))
    dt_l = ls_json["dies_trigger_label_sp"]
    et_l = ls_json["etb_trigger_label_sp"]
    A(f"""
456 real creatures with a `triggered: when CARDNAME dies,` line, with the two
words substituted in place. **Moving a trigger from death to entry is worth
+0.091 SD** [+0.075, +0.107], correct-signed on 69%; moving it to attack costs
−0.047. Nothing is off-manifold and no line moves.

The label side has no such gradient: among creatures the dies-trigger flag is
{dt_l['beta']:+.3f} ± {dt_l['se']:.3f} and the ETB flag
{et_l['beta']:+.3f} ± {et_l['se']:.3f}, i.e. both indistinguishable from zero
and from each other. So the encoder prices a timing premium the labels do not
pay — **a mild flagged divergence** (0.09 SD, well inside the range where the
label estimate's own noise could hide it, so it is a soft flag).
""")

    # ────────────────────────────────────────────────────────────────────
    A("\n---\n\n## Divergence summary\n")
    A("""Everything the C battery found where the encoder's *causal* response
parts company with what the labels pay, ordered by size. A positive gap means
the encoder charges more than the labels do; a negative gap, less.

| # | finding | encoder-counterfactual | label-correlational | gap | confidence |
| --- | --- | --- | --- | --- | --- |
| 1 | `you gain 4 life.` as a spell's whole text (C3) | −1.22 SD (centred on the ladder) | −0.03 ± 0.06 (null) | ≈ −1.2 | high — n=200, tight CI, label null is well estimated |
| 2 | tribal-noun spread compressed to a third (C5) | 0.31 SD spread | 0.94 SD spread | −0.63 spread | high |
| 3 | integer statlines carry no ordinal structure past ~8 (C2) | 12/12 ≈ 0/0 | monotone in P+T | n/a | high — 8 of 12 steps non-monotone |
| 4 | keyword-count ladder does not saturate (C1) | +0.36/+0.20/+0.16/+0.29 | +0.35/+0.28/+0.13 (saturating) | +0.16 at rung 4 | medium — rungs 3-4 partly off-manifold |
| 5 | protection keywords (hexproof / ward / shroud) (C1) | −0.24 / −0.19 / −0.12 | +0.03 / +0.08 / +0.10 | ≈ −0.27 each | medium — small label n, but three-for-three sign agreement |
| 6 | wordiness: any appended spell-line clause pays +0.09 (C6) | +0.09 regardless of content | 0 by construction | +0.09 | high — four riders of opposite sign, all +0.09 |
| 7 | `artifact` prepended to a creature type line (C5) | **+0.068** | **−0.212** | +0.28 | high — n=400 and n=1000, both tight |
| 8 | flying x size is an inverted U, not a slope (C1) | peak at P+T 7-8, falls after | +0.045 SD/point, monotone | n/a | high — replicated in two designs |
| 9 | haste / double strike / reach / menace overpriced (C1) | +0.16 / +0.14 / +0.09 / 0.00 | −0.05 / −0.03 / −0.04 / −0.13 | ≈ +0.15 each | medium |
| 10 | death → entry trigger timing premium (C6) | +0.091 | ≈ 0 (both flags null) | +0.09 | low-medium — inside label noise |
| 11 | deathtouch+trample superadditivity (C1) | +0.139 interaction | untestable (4 real cards) | n/a | encoder-only claim |

And the places where the counterfactual **confirms** the label prior, which
are worth as much:

* flying, +0.27 centred / +0.40 absolute, against a label prior of +0.37
  centred (C1) — and this is the calibration anchor for the whole battery.
* the P−T gradient, +0.038 SD/point against a label +0.043 ± 0.006, including
  its size dependence (C2).
* the colour fee, both signs, on all five colours (C2).
* the removal ladder's entire internal ordering — burn > exile > destroy >
  conditional > edict > tap > sweeper — against the label WLS column (C3).
* fight not discounted, confirming the scorer's dislike is search-level (C3).
* lockdown auras at the top of noncreature text (C3).
* taplands ≥ untapped lands (C5).
* cantrip riders worth zero (C6).
* the body-vs-spell master axis and every class ordering inside it (C4).

## What this means for scoring hypothetical cards

1. **Keyword and effect vocabulary is trustworthy; magnitudes are not.** The
   encoder ranks fifteen spell effects and sixteen keywords the way the labels
   do, but it cannot read `7/7` as bigger than `4/4`, cannot read `{X}` as
   scaling, and prices a fourth keyword as highly as a second.
2. **Adding words to a spell raises its score by ~0.09 SD whatever they say.**
   Any hypothetical whose novelty is a *restriction* written into the spell
   line will be over-scored. On a creature's own triggered line the sign
   channel does work, so drawbacks written as triggers are priced.
3. **Type words are not levers.** `artifact` is not a penalty token and
   `instant` is not a speed token, even though the encoder's outputs correlate
   correctly with both. Do not expect a re-typing edit to move a score the way
   the corpus statistics suggest.
4. **Tribal nouns are levers, at about a third of their apparent worth.**
   Naming a hypothetical an angel rather than a lizard is worth ~0.31 SD of
   predicted score for no game reason at all.
5. **Bodies are a floor and a cap.** Stapling an effect to a 2/2 is worth
   +0.60 SD when the effect is weak and slightly *negative* when it is strong.
   Compare a hypothetical creature against creatures and a hypothetical spell
   against spells, never across.

## Artifacts

`c1_pairwise.csv`, `c1_pairwise_raw.npz`, `c1_scale.csv`, `c1_delete.csv`,
`c1_summary.json`, `c1b_flying_size_sub.csv`, `c1b_flying_size_add.csv`,
`c1b_add_keyword.csv`, `c1b_dt_trample.csv`, `c1b_ladder.csv`,
`c1b_summary.json`, `c2_pt_asymmetry.csv`, `c2_pt_by_size.csv`,
`c2_nn_sweep.csv`, `c2_mv_sweep.csv`, `c2_x_cost.csv`, `c2_pips.csv`,
`c2_summary.json`, `c3_spell_ladder.csv`, `c3_ladder_by_type.csv`,
`c3_aura.csv`, `c3_summary.json`, `c4_shells.csv`, `c4_body_premium.csv`,
`c4_body_premium_agg.csv`, `c4_rock_dork.csv`, `c4_class_anchors.csv`,
`c4_summary.json`, `c5_tribe_pairwise.csv`, `c5_tribe_scale.csv`,
`c5_type_swaps.csv`, `c5_taplands.csv`, `c5_summary.json`, `c6_riders.csv`,
`c6_end_triggers.csv`, `c6_timing.csv`, `c7_labelside.csv`,
`c7_labelside.json` — all under `output/encoder-probes/`.
""")

    (S / "c_report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {S / 'c_report.md'}")
    # echo the derived comparison tables for the caller
    comp.reset_index()[["keyword", "value_sp", "label_c", "predcorr_c", "gap"]] \
        .to_csv(S / "c_divergence_keywords.csv", index=False)
    tc.reset_index()[["tribe", "value_sp", "label_c", "gap"]] \
        .to_csv(S / "c_divergence_tribes.csv", index=False)
    print(comp.reset_index()[["keyword", "value_sp", "label_c", "predcorr_c", "gap"]]
          .to_string(index=False))
    print(tc.reset_index()[["tribe", "value_sp", "label_c", "gap"]].to_string(index=False))


if __name__ == "__main__":
    main()

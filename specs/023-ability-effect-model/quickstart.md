# Quickstart: Ability effect model (stage one)

**Feature**: `023-ability-effect-model` | **Date**: 2026-09-06

Stage one end to end, against a **stock, unpatched** Forge checkout. This is the walkthrough that
produces the first `e` vectors and all three gate verdicts, and it is the acceptance path for User
Story 1 in [spec.md](spec.md).

## Prerequisites

```bash
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
cd forge-connector && mvn package -DskipTests   # fat JAR the workers run from
```

Sibling `../forge` checkout built with `mvn install -DskipTests`. Nothing is patched at this stage.

## 1. Convert, with sidecars and token scripts

```bash
python -m price_predictor convert
```

Now writes three things instead of one: converted card text under `output/cardsfolder/`, a
`<name>.provenance.json` beside each of them, and converted token scripts under `output/tokenscripts/`
with sidecars of their own.

**Checks**: every converted `.txt` has a sidecar beside it, every sidecar line's `line_index` points
inside the file it describes, and a card whose converter deduplicated a trait lists that trait in
`dropped_keys`. A sub-ability or mode rendered as its own line carries no key — it is reached through
its parent's `sub_ability_links` — so a line with an empty `provenance` is expected rather than a
fault.

## 2. Keyword definitions and vocabulary

```bash
python -m effects extract-keyword-definitions
python -m effects build-vocab
```

The first writes `output/effects/keyword-definitions.json` (reminder-text templates for every keyword;
captured scripts arrive at stage four). The second writes `models/effects/vocab.txt`, scanning the
converted cards, the token scripts, and the keyword-definition file.

**Check**: the vocabulary contains `[PAD]`, `[UNK]`, `cardname`, `[MASK]`, `[CLS]` and is ≤ 5000 tokens.

## 3. Collect, riding ordinary self-play

```bash
python -m sealed match-outcomes --effect-records output/effects/records/
```

Run it as long as you would run it anyway — the instrumentation adds no simulation cost. Ctrl-C to stop.

**Checks**:
- `output/effects/records/` fills with `{run_id}.{worker}.jsonl` shards holding `resolution` and
  `combat` records.
- Every record carries `"mode": "degraded"` — the workers probed for the patch hooks, found none, and
  fell back to bracket attribution.
- `output/sealed/match-outcomes.txt` and `cards-played.txt` are unchanged in format and content by the
  flag's presence.
- A first-strike combat produced two `combat` records, one per damage step.

## 4. Train

```bash
python -m effects train-effect-model --withhold-keyword cascade
```

`--withhold-keyword` holds one implemented keyword's token out of training so the zero-shot check in
step 7 has something to measure; its occurrences are always expanded instead. Any implemented keyword
outside gate 2's eight damage-step keywords works; `cascade` is just an example. Withholding one of
the eight would degrade that keyword's own gate-2 verdict in the same run.

Trains both transformers jointly from random init. Stage one has three of the eight sampling
classes — `resolution-cost`, `resolution-effect`, and `combat` — so the mixture renormalizes over what
is present and the absent kinds' fields contribute no loss.

**Checks**: the checkpoint under `models/effects/effect-model/` records its held-out card list, its
`game_id` set across both strata, and the vocabulary and keyword-definition paths plus their hashes.

## 5. Train the baselines

All four are needed at stage one, because every reported check except the three that wait for later
stages runs now: gate 1 compares against `identity`, every record kind reports `state-only` as its
floor, the average-effect control is `no-state`, and the taxonomy comparison is `taxonomy`. Each
inherits the split so the comparison is honest:

```bash
for V in identity state-only no-state taxonomy; do
  python -m effects train-effect-model --variant $V \
      --split-from models/effects/effect-model/latest.pt
done
```

These write under `models/effects/effect-model/{variant}/`, never over the shipping checkpoint.
`--split-from` is required for a variant run — without it the run fails fast, rather than silently
computing its own split and making the comparison meaningless.

## 6. Encode the cache

The `e`-geometry checks (gate 3, the decodability battery, the ward canary, the scorer smoke test)
read `full`, `no-state`, and `taxonomy`. `identity` and `state-only` are prediction baselines and
need no cache, so encoding them is optional:

```bash
python -m effects encode-abilities
for V in no-state taxonomy; do
  python -m effects encode-abilities --variant $V
done
```

**Checks**: `output/effects/abilities/cardsfolder/…` and `…/tokenscripts/…` mirror their source trees;
each file is `(n_lines, e_dim)` and row-aligned with that source's sidecar. Each variant run wrote
`<name>.{variant}.npz` beside the shipping `<name>.npz` rather than replacing it. The `taxonomy`
variant has no encoder, so its file is the taxonomy lookup emitted into the same row layout — every
`e`-geometry check reads one file shape.

## 7. Evaluate

```bash
python -m effects evaluate-effect-model \
    --variant-checkpoint identity=models/effects/effect-model/identity/latest.pt \
    --variant-checkpoint state-only=models/effects/effect-model/state-only/latest.pt \
    --variant-checkpoint no-state=models/effects/effect-model/no-state/latest.pt \
    --variant-checkpoint taxonomy=models/effects/effect-model/taxonomy/latest.pt
```

Splits come from the checkpoint — never a flag, never recomputed.

**What you get, and what to do with it**:

| Result | Meaning |
|---|---|
| **Gate 1** (identity baseline) | Blocks shipping. All three margins must hold on the card-disjoint split's unique-text stratum. Failure means the encoder is not reading text. |
| **Gate 3** (collapse canaries) | Blocks shipping. Mean pairwise cosine ≤ 0.5 over 10,000 pairs, top PC ≤ 30% of variance. |
| **Gate 2** (damage-step canary), per keyword | Blocks nothing. Each of the eight keywords passes or is routed to a stage-three probe. **This is the output that decides whether stage three builds probe machinery at all.** |
| Ward canary, nearest-neighbour / UMAP, decodability battery, scorer smoke test | Reported, not gating |

Checks needing records that do not exist yet are skipped: matched real-vs-fork agreement and the
probe-diff re-check wait for stage three, the role-polarity probe for stage two.

## What stage one deliberately does not do

No Forge source is modified. There are no `continuous`, `playability`, `trigger`, `rewrite`, or mana
records, and no forks. Evasion keywords will not separate — their signal is playability records at
stage two, which is why gate 2 covers only the damage-step family.

## Where results go

Run results belong in the design record's Outcome section
([`../../experiments/2026-09-04-ability-effect-model-design.md`](../../experiments/2026-09-04-ability-effect-model-design.md)),
never in the root spec and never here.

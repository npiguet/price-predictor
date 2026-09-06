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

**Check**: a card whose converted file has N ability lines has a sidecar with N `lines` entries, and a
merged line carries more than one provenance key.

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
python -m effects train-effect-model
```

Trains both transformers jointly from random init. Stage one has only two of the eight sampling
classes, so the mixture renormalizes over what is present and the absent kinds' fields contribute no
loss.

**Checks**: the checkpoint under `models/effects/effect-model/` records its held-out card list, its
`game_id` set across both strata, and the vocabulary and keyword-definition paths plus their hashes.

## 5. Train the baselines

Gate 1 needs `identity`; every record kind reports `state-only` as its floor. Both inherit the split
so the comparison is honest:

```bash
python -m effects train-effect-model --variant identity \
    --split-from models/effects/effect-model/latest.pt
python -m effects train-effect-model --variant state-only \
    --split-from models/effects/effect-model/latest.pt
```

These write under `models/effects/effect-model/{variant}/`, never over the shipping checkpoint.

## 6. Encode the cache

```bash
python -m effects encode-abilities
python -m effects encode-abilities --variant identity
```

**Checks**: `output/effects/abilities/cardsfolder/…` and `…/tokenscripts/…` mirror their source trees;
each file is `(n_lines, e_dim)` and row-aligned with that source's sidecar. The `identity` run wrote
`<name>.identity.npz` beside the shipping `<name>.npz` rather than replacing it.

## 7. Evaluate

```bash
python -m effects evaluate-effect-model \
    --variant-checkpoint identity=models/effects/effect-model/identity/latest.pt \
    --variant-checkpoint state-only=models/effects/effect-model/state-only/latest.pt
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

# Quickstart: Ability effect model

**Feature**: `023-ability-effect-model` | **Date**: 2026-09-06

The operating procedure for the whole feature. Steps 1–7 produce the first `e` vectors and all three
gate verdicts, and are the acceptance path for User Story 1 in [spec.md](spec.md).
[After stage one](#after-stage-one) covers the later stages, each of which widens the corpus without
invalidating what came before.

They run against a patched or a stock `../forge` alike. Applying
[the engine patch](../../forge-connector/patches/) first is worth it — it is one `git am` and one
rebuild, and it takes a run from three of the eight sampling classes to six — but nothing in the
procedure requires it, and every step below says what changes either way.

| Step | Command | Wall clock | Attended |
|---|---|---|---|
| 1 | `price_predictor convert` | minutes | no |
| 2 | `effects extract-keyword-definitions`, `effects build-vocab` | minutes | no |
| 3 | `sealed match-outcomes --effect-records` | hours | you decide when to stop |
| 4 | `effects train-effect-model` | hours | no |
| 5 | four baseline runs | 4 × step 4 | no |
| 6 | `effects encode-abilities` ×3 | minutes | no |
| 7 | `effects evaluate-effect-model` | minutes | read the output |

## Prerequisites

```bash
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
cd forge-connector && mvn package -DskipTests   # fat JAR the workers run from
```

Sibling `../forge` checkout built with `mvn install -DskipTests`. The JAR carries several mains the
Python side spawns — `ConvertMain`, `KeywordDefinitionMain`, `VariantSidecarMain`, `CastabilityMain`,
`MatchWorkerMain` — so rebuild it after any Java change, and **after applying the engine patch**,
because the worker links against the freshly installed Forge jars.

Apply the patch first unless there is a reason not to:

```bash
cd ../forge
git checkout -b effect-record-hooks
git am ../price-predictor/forge-connector/patches/0001-*.patch
mvn -pl forge-core,forge-game,forge-ai -am install -DskipTests
cd ../price-predictor/forge-connector && mvn package -DskipTests
```

## 1. Convert, with sidecars and token scripts

```bash
python -m price_predictor convert
```

Writes three things: converted card text under `output/cardsfolder/`, a `<name>.provenance.json`
beside each of them, and converted token scripts under `output/tokenscripts/` with sidecars of their
own. The sidecar is the join between a runtime Forge trait and a converted line; without it every
record collected in step 3 is unjoinable.

This rewrites the converted tree. The sealed pipeline's `.npz` card embeddings sit in that same tree
and are not deleted, but they stop describing their neighbours if the rendered text moves — re-run
`python -m sealed encode-cards --clean` afterwards if the sealed pipeline reads this corpus.

**Checks**: every converted `.txt` has a sidecar beside it, every sidecar line's `line_index` points
inside the file it describes, and a card whose converter deduplicated a trait lists that trait in
`dropped_keys`. A sub-ability or mode rendered as its own line carries no key — it is reached through
its parent's `sub_ability_links` — so a line with an empty `provenance` is expected rather than a
fault. Two cards warn (`bind_liberate`, `start_fire`); more than a handful is not expected.

## 2. Keyword definitions and vocabulary

```bash
python -m effects extract-keyword-definitions
python -m effects build-vocab
```

Order matters. `build-vocab` scans three sources — converted cards, token scripts, and the
keyword-definition file — so a vocabulary built before the keyword file exists expands every keyword
to `[UNK]`, which looks like a working run.

`extract-keyword-definitions` writes `output/effects/keyword-definitions.json`: a reminder-text
template for every keyword, plus the implementation script for the keywords whose factory generates
one from a bare display name. The parameterized keywords generate nothing without the value the card
printed, and the engine-coded family generates nothing at all; both keep their template.

**Checks**: `build-vocab` reports **3 sources**, not 2. The vocabulary contains `[PAD]`, `[UNK]`,
`cardname`, `[MASK]`, `[CLS]` at ids 0–4 and is ≤ 5000 tokens.

## 3. Collect, riding ordinary self-play

```bash
python -m sealed match-outcomes --effect-records output/effects/records/ --workers 6
```

Instrumentation is an opt-in on a command that already exists. The flag costs no extra simulation —
it rides matches that were going to be played anyway — and the sealed corpora keep their exact format,
so this doubles as a sealed self-play run. Ctrl-C to stop.

**Checks**:
- `output/effects/records/` fills with `{run_id}.{worker}.jsonl.gz` shards.
- Every record carries the same `mode`, and it is the one the checkout offers: `patched` after the
  engine patch, `degraded` without it, printed by each worker at startup. A run cannot mix the two —
  the mode is probed once per worker.
- A `patched` run reaches `resolution`, `combat`, `playability`, `trigger`, `rewrite` and
  `continuous` — seven of the eight sampling classes, mana records arriving as the effect half of
  `resolution`. A `degraded` run reaches `resolution` and `combat` only, which is three.
- `output/sealed/match-outcomes.txt` and `cards-played.txt` are unchanged in format and content by the
  flag's presence.
- A first-strike combat produced two `combat` records, one per damage step.

```bash
# records collected so far
python -c "import sys;sys.path.insert(0,'src');from pathlib import Path;\
from effects.infrastructure.record_io import count_records;\
print(count_records(Path('output/effects/records')))"
```

### How long to collect

Gate 2 sets the corpus size, and it is the only judgement call in the procedure. It needs **200
qualifying combat records per keyword** across the eight damage-step keywords, and those eight differ
by more than an order of magnitude in how often sealed play produces them.

Size the run in **games**, not records. A game yields a few hundred records, but the mix shifts with
the collectors installed, so a record target moves under you while a game target does not:

| Keyword | Games for 200 qualifying records |
|---|---|
| trample | ~380 |
| first strike | ~410 |
| deathtouch | ~820 |
| lifelink | ~960 |
| double strike | ~1,300 |
| indestructible | ~2,500 |
| infect | ~3,900 |
| wither | ~7,000 |

**About 4,000 games clears seven of the eight, and about 7,000 clears all eight.** Rates measured
against a real corpus are in the design record's feasibility section; a corpus large enough for gate 2
is comfortably large enough for gate 1 and for the split.

A patched run writes roughly 320 records and 80 KB per game, so 7,000 games is about half a gigabyte.
Shards are gzip-compressed, which is where the room comes from — the same corpus uncompressed is
around 30 GB.

```bash
# what has been collected so far
python -c "import sys;sys.path.insert(0,'src');from pathlib import Path;\
from collections import Counter;from effects.infrastructure.record_io import read_records;\
rs=list(read_records(Path('output/effects/records')));\
print(f'{len({r.game_id for r in rs})} games, {len(rs)} records');\
print(Counter(r.kind.value for r in rs))"
```

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

**Read the first line it prints.** It reports the split: total records, then how many are training,
card-disjoint and game-disjoint. The split holds out cards by newest first printing until they cover
≥ 8% of the corpus, then excludes every game holding a record that names one of them — so a thin
corpus can leave most of its records untrainable. Collect more rather than training on a small
training share.

On an 8 GB card, `--context-cache` is the documented fallback: it swaps live context re-encoding for a
stop-gradient momentum cache refreshed every `--cache-refresh` batches.

**Checks**: the checkpoint under `models/effects/effect-model/` records its held-out card list, its
`game_id` set across both strata, and the vocabulary and keyword-definition paths plus their hashes.

## 5. Train the baselines

All four are needed at stage one, because every reported check except the three that wait for later
stages runs now: gate 1 compares against `identity`, every record kind reports `state-only` as its
floor, the average-effect control is `no-state`, and the taxonomy comparison is `taxonomy`. Each
inherits the split, and each holds out the same keyword, so the only difference between a baseline and
the shipping model is the input its variant masks:

```bash
for V in identity state-only no-state taxonomy; do
  python -m effects train-effect-model --variant $V \
      --split-from models/effects/effect-model/latest.pt \
      --withhold-keyword cascade
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

Splits come from the checkpoint — never a flag, never recomputed. A baseline recording a different
split, or different vocabulary hashes, fails the run rather than being compared. The command exits
non-zero when a blocking gate fails.

**What you get, and what to do with it**:

| Result | Meaning |
|---|---|
| **Gate 1** (identity baseline) | Blocks shipping. All three margins must hold on the card-disjoint split's unique-text stratum — resolution records whose acting text appears on no training card, the only slice where a free embedding per text cannot recall the answer. Failure means the encoder is not reading text. |
| **Gate 3** (collapse canaries) | Blocks shipping. Mean pairwise cosine ≤ 0.5 over 10,000 pairs, top PC ≤ 30% of variance. |
| **Gate 2** (damage-step canary), per keyword | Blocks nothing. Each of the eight keywords passes or is routed to a stage-three probe. **This is the output that decides whether stage three builds probe machinery at all.** |
| Ward canary, nearest-neighbour / UMAP, decodability battery, scorer smoke test | Reported, not gating |

Checks needing records that do not exist yet are skipped: matched real-vs-fork agreement and the
probe-diff re-check wait for stage three, the role-polarity probe for stage two, and the decodability
battery needs `output/sealed/cards-win-rates.txt` from a `train-encoder` run.

## What stage one deliberately does not do

No Forge source is modified. There are no `continuous`, `playability`, `trigger`, `rewrite`, or mana
records, and no forks. Evasion keywords will not separate — their signal is playability records at
stage two, which is why gate 2 covers only the damage-step family.

## After stage one

Each stage below widens the corpus. None of them invalidates a record collected earlier, because the
envelope is frozen before stage-one collection; re-run steps 4–7 against the widened corpus to see
what the new records buy.

### Stage two — the patch set, and the cards self-play never deals

The four hooks in `forge-connector/patches/` are **hook specifications**, applied to the sibling
checkout by hand; see that directory's README. Applying them unlocks the `rewrite`, `continuous`,
`trigger` and `playability` record kinds plus snapshot tier 3, and makes workers stamp `patched`
instead of `degraded`. Rebuild `../forge` and the fat JAR afterwards.

Sealed self-play only ever plays what sealed pools contain, so a card in no sealed-legal set never
appears in a record at all:

```bash
python -m effects collect-coverage \
    --split-from models/effects/effect-model/latest.pt \
    --target-records 50 --workers 12
```

Decks are built over the whole converted corpus, weighted toward the cards with the fewest records,
and rounds play until every card is satisfied or retires after `--no-progress-rounds` without a new
qualifying record. `--split-from` keeps held-out cards out of every deck; without it a coverage run
contaminates the split of the model it feeds. The run reports two residues — cards judged uncastable,
and castable cards short of `--target-records` — and both fall to stage three.

### Stage three — interventions and probes

Only if gate 2 routed keywords. Its per-keyword verdict names exactly which:

```bash
python -m effects collect-coverage \
    --split-from models/effects/effect-model/latest.pt \
    --interventions-per-game 2 \
    --probe-keywords first_strike,double_strike
```

`--probe-keywords` is comma-separated and empty by default, so a checkout carrying the machinery takes
no fork unless asked. An interventional resolution forces an ability that no game plays; a damage-step
probe re-runs one combat with a keyword stripped. Every fork is score-checked against the live game
before any perturbation, a discarded fork still counts against its budget, and at most two forks may
target one real resolution.

### Stage four — the script surface and synthetic variants

The script surface is a second vocabulary and a second cache, side by side with the prose ones:

```bash
python -m effects build-vocab --surface script          # models/effects/vocab-script.txt

python -m effects collect-variants \
    --split-from models/effects/effect-model/latest.pt \
    --variant-volume 0.2
```

`collect-variants` reads Forge's **source** scripts, perturbs one whitelisted parameter per variant,
writes a sidecar for each through `VariantSidecarMain`, and plays them. `--variant-volume` caps
variant records as a fraction of the real records already present, so the cap scales with the corpus.
A variant of a held-out card is held out with it.

Then re-run steps 4, 6 and 7 with the variant tree, and — for the script surface — the script
vocabulary. The surface follows the loaded vocabulary rather than a flag of its own, so the two cannot
disagree:

```bash
python -m effects train-effect-model \
    --vocab-path models/effects/vocab-script.txt \
    --variant-scripts output/effects/variant-scripts/ \
    --withhold-keyword cascade
```

`--variant-scripts` also takes on `encode-abilities` and `evaluate-effect-model`. It defaults to absent
everywhere, so a stage-one checkpoint keeps working unchanged after a stage-four rebuild exists.

## Where results go

Run results belong in the design record's Outcome section
([`../../experiments/2026-09-04-ability-effect-model-design.md`](../../experiments/2026-09-04-ability-effect-model-design.md)),
never in the root spec and never here. Record gate 1's three margins, gate 3's two geometry figures,
and gate 2's per-keyword table — the routed list is the input to the stage-three decision.

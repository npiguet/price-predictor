# Quickstart: Ability effect model

**Feature**: `023-ability-effect-model` | **Date**: 2026-09-06

The operating procedure for the whole feature. The path below collects **everything the record schema
names** and trains against it; [Collecting less](#collecting-less) says what to drop for a smaller run,
and [What is not collected](#what-is-not-collected) lists the two fields that are empty on purpose.

Steps 1–8 are the acceptance path for User Story 1 in [spec.md](spec.md).

| Step | Command | Wall clock | Attended |
|---|---|---|---|
| 0 | apply the engine patch series, rebuild Forge and the JAR | minutes | no |
| 1 | `price_predictor convert` | minutes | no |
| 2 | `effects extract-keyword-definitions`, `effects build-vocab` | minutes | no |
| 3 | `effects holdout-cards`, `sealed match-outcomes --exclude-cards --effect-records` | hours | you decide when to stop |
| 3b | the same, full strength, into the same shard directory | hours | no |
| 4 | `effects collect-coverage` | hours | no |
| 5 | `effects collect-variants` | hours | no |
| 6 | `effects train-effect-model`, then four baselines | 5 × hours | no |
| 7 | `effects encode-abilities` ×3 | minutes | no |
| 8 | `effects evaluate-effect-model` | minutes | read the output |

Steps 3–5 are three different ways to reach cards, and a full corpus wants all three: self-play plays
what sealed pools deal, coverage plays what they never deal, and variants play text that never
existed. Each writes into the same shard directory and none invalidates the others.

Step 3 collects the training corpus from pools the held-out cards were removed from, and step 3b
the validation corpus from pools at full strength. Splitting the corpus at collection time rather
than discarding held-out games afterwards is what keeps every collected game usable;
[spec.md](spec.md) FR-130 has the rules.

## Prerequisites

```bash
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
cd forge-connector && mvn package -DskipTests   # fat JAR the workers run from
```

Sibling `../forge` checkout built with `mvn install -DskipTests`. The JAR carries several mains the
Python side spawns — `ConvertMain`, `KeywordDefinitionMain`, `VariantSidecarMain`, `CastabilityMain`,
`MatchWorkerMain` — so rebuild it after any Java change, and **after applying the engine patch**,
because the worker links against the freshly installed Forge jars.

## 0. Put the sibling checkout on the hooks branch

```bash
cd ../forge
git checkout effect-record-hooks          # after a Forge upgrade: git rebase master
mvn -pl forge-core,forge-game,forge-ai -am install -DskipTests
cd ../price-predictor/forge-connector && mvn package -DskipTests
```

The hooks are commits on that branch, one per hook. This repository carries no exported patch copy —
the branch is the history — so a Forge upgrade is a rebase, and `PatchHooks.REQUIRED` is the
inventory of what the branch has to provide.

Rebuild the fat JAR afterwards, always: the worker links against the freshly installed Forge jars.
Without the hooks a run collects three of the eight sampling classes and stamps every record
`degraded`; with them, all eight, and the two fork kinds become available. Nothing in the procedure
*requires* them, but a run meant to collect everything does.

**Read the two lines each worker prints at startup**, not just the first. The mode is one hook's
answer, so a half-rebased branch still reports `patched` while a channel this run meant to collect is
quietly empty; the second line names any required hook that is missing and what it costs.

Workers run detached, so that startup banner is never on a terminal you're watching — it, and
everything else a worker prints (including the one-line reports the five latched effect-record
failure reporters give when an emitter or listener misbehaves), lands in
`{run_id}.{worker}.log` next to that worker's shards, under whichever directory you pointed
`--effect-records` at. Each is capped at 20 MiB with one rotated `.log.1` backup per worker, so a
long run cannot fill a disk with them. **Read these after every run, not just when something looks
wrong** — a reporter line prints once per worker JVM and does not repeat, so it is easy to miss if
you only check when throughput drops.

## 1. Convert, with sidecars and token scripts

```bash
python -m price_predictor convert \
    --output-path ./output/cardsfolder \
    --tokens-output-path ./output/tokenscripts
```

Both paths are named rather than defaulted. The defaults are these same two trees, but they are
**relative**, so the command run from anywhere but the repository root writes a second converted
corpus where nothing looks for it and leaves the real one stale — a failure that surfaces hours
later as records that join nothing.

Writes three things: converted card text under `output/cardsfolder/`, a `<name>.provenance.json`
beside each of them, and converted token scripts under `output/tokenscripts/` with sidecars of their
own. The sidecar is the join between a runtime Forge trait and a converted line; without it every
record collected later is unjoinable.

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
python -m effects holdout-cards --out output/effects/records/depleted/holdout-cards.txt

python -m sealed match-outcomes \
    --exclude-cards output/effects/records/depleted/holdout-cards.txt \
    --effect-records output/effects/records/depleted/ --workers 6 \
    --snapshot-tiers 1,2,3,4 \
    --playability-rate 0.1 \
    --legality-rate 0.1 \
    --interventions-per-game 2 \
    --probes-per-game 2 \
    --probe-keywords first_strike,double_strike,deathtouch,lifelink,trample,indestructible,wither,infect
```

`--snapshot-tiers 1,2,3,4` because this command opts into interventions and probes, which are
stage three, and stage three collects at that depth. Left at the `1,2,3` default it would write
stage-three forks into stage-two snapshots, and tier 4 is the one tier no later pass can add:
an absent tier means uncollected, so a reader cannot tell an empty graveyard from an
unrecorded one. It costs about 59% more entities per record — 11.8 cards sit in hands and
graveyards against the 19.9 entities a snapshot already carries.

The two sampling rates are named rather than defaulted, because both defaults are judgement
calls and the command should carry them. `0.1` is right for the same reason in both cases: the
question is not what share of the corpus a class occupies, it is whether the class holds enough
distinct records for the batch quota to draw from. A full 40-epoch training draws roughly 640,000
`playability/decision` examples (5000 steps x 32 batch x 40 epochs x the 10% quota); at `0.1` an
eight-hour run collects around 7.7 million of them, twelve times over. Raising the rate buys
nothing that the quota can spend and costs games: measured on this machine, `1.0` collected 60
million decision records and 34,900 games where `0.1` collected 44,300 games, and it is the games
that widen card coverage — which is what the card-disjoint split actually needs. Lowering it below
`0.1` only saves disk.

`--legality-rate` earns its own knob because the two playability subkinds arrive at wildly
different volumes from one priority pass; legality records are also already coalesced per game on
their rendered payload, so `0.1` samples what survives dedup rather than the raw flood.

`--mana-cap` stays at its default of 1. A Mountain taps for `R` a dozen times a game against a
board that barely moved, and the cap is keyed on the mana produced, so a dual land still records
both of its colours; the last corpus carried 154,000 `mana_produced` events at that cap.

Instrumentation is an opt-in on a command that already exists. The flag costs no extra simulation —
it rides matches that were going to be played anyway — and the sealed corpora keep their exact format,
so this doubles as a sealed self-play run. Ctrl-C to stop.

Only one of the three fork flags is off by default, and it is not the one that reads like a switch.
`--interventions-per-game` and `--probes-per-game` both default to 2, so a patched run already forks
for interventions whether or not you name them; naming them here makes the budget explicit rather
than turning it on. `--probe-keywords` defaults to empty and is what actually gates probes. Naming
all eight damage-step keywords collects a probe branch for each, which is what gate 2 checks its
model-side perturbation against; naming none takes no probe at all. Forks are the only expensive
mechanism here — each costs a game copy that re-parses every card from its script — so the per-game
budgets are the throttle, not the keyword list.

**`--probes-per-game` is a budget, not a switch.** Without `--probe-keywords` it buys nothing, and a
run that omits the keyword list collects a corpus with zero probe forks in it — which is what the
55,296-record smoke corpus was, with nothing saying so until gate 2 had no engine-side branch to
compare against. Every collecting command therefore prints its probe state before the first worker
spawns:

```
Damage-step probes DISABLED: --probe-keywords is empty, so the --probes-per-game 2 budget buys no fork at all. Pass --probe-keywords <keyword>[,<keyword>...] to take any.
```

`validate-corpus` reports the probe count as a watched number for the same reason, so a run that
*meant* to probe is caught in its first minutes rather than at evaluation.

**Checks** — run the first one *while the pass is still young*, not after it:

```bash
python -m effects validate-corpus --effect-records output/effects/records/ --limit 50000
```

Every defect the first eight-hour run produced was already visible in its first minute of shards.
Read the `[WATCH]` lines as well as the verdicts: they carry the probe count, the `zone_change`
`from_zone` share and the `attributed_to` breakdown, none of which fails a run and all of which say
whether a channel is wired.

- `output/effects/records/` fills with `{run_id}.{worker}.jsonl.gz` shards.
- The same directory fills with `{run_id}.{worker}.log` files, one per worker slot, carrying
  everything that worker printed — including any of the five latched effect-record failure
  reporters. See "Put the sibling checkout on the hooks branch" above for what they are and how
  they're capped.
- Every record carries the same `mode`, and it is the one the checkout offers: `patched` after the
  engine patch, `degraded` without it, printed by each worker at startup. A run cannot mix the two —
  the mode is probed once per worker.
- A `patched` run reaches **all eight** sampling classes: `resolution` in both halves, `combat`,
  `continuous`, `trigger`, `rewrite`, and `playability` in both its decision and legality subkinds.
  Mana records arrive as the effect half of `resolution`.
- **`rewrite` should now be a populated class, not a rounding error.** One record is written per
  replacement evaluation, `not_replaced` included, so the count should be comparable to `trigger`'s
  rather than the 34-in-1.88M the old `{incoming, outgoing}` payload produced. A `rewrite` count still
  in the tens after a full run means the hook is not passing `result`, not that replacements are rare.
  Check the `result` breakdown as well as the count: all five values reachable, `not_replaced` the
  bulk of them, and `outgoing` non-null only on the genuine in-place edits (counter counts, damage
  amounts). Every `outgoing` non-null, or every one null, is a wiring defect.
- `output/sealed/match-outcomes.txt` and `cards-played.txt` are unchanged in format and content by the
  flag's presence.
- A first-strike combat produced two `combat` records, one per damage step.

```bash
# what has been collected so far
python -c "import sys;sys.path.insert(0,'src');from pathlib import Path;\
from collections import Counter;from effects.infrastructure.record_io import read_records;\
rs=list(read_records(Path('output/effects/records')));\
print(f'{len({r.game_id for r in rs})} games, {len(rs)} records');\
print(Counter(r.kind.value for r in rs))"
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

A patched run writes roughly 410 records and 90 KB per game, so 7,000 games is under a gigabyte.
Shards are gzip-compressed, which is where the room comes from — the same corpus uncompressed is
tens of gigabytes.

### The cap flags

Seven flags govern what one worker collects, and every one is a per-worker quantity the supervisor
cannot observe, so they travel to the JVM as `-Deffect.*` properties. All three collecting commands
accept all seven: `match-outcomes`, `collect-coverage` and `collect-variants`.

| Flag | Default | What it caps |
|---|---|---|
| `--mana-cap` | 1 | Records per unique mana ability **per game**, drawn uniformly from that game's activations rather than taken first — a land's first tap is turn one against an empty board, and taking it would make every mana record describe the same early game. Keyed on the mana produced, so a dual land's colours each record. |
| `--playability-rate` | 0.1 | Share of decision points sampled. The legality subkinds are coalesced on their rendered payload instead, since the AI re-asks who may block while it evaluates. |
| `--interventions-per-game` | 2 | Forced resolutions per game. On at the default — pass `0` to stop them, since omitting the flag leaves them running. |
| `--probes-per-game` | 2 | Damage-step probes per game. A budget, not a switch. |
| `--probe-keywords` | *(empty)* | Which keywords a probe may strip. Empty means **no probe is ever taken**, whatever the budget — the state each collecting command announces at startup. |
| `--legality-rate` | 0.1 | Share of legality points kept, sampled *after* the coalescing above. Its own knob because the two playability subkinds arrive at very different volumes from one priority pass. |
| `--snapshot-tiers` | `1,2,3` | How deep every snapshot reaches: a prefix of `1,2,3,4` — 1 referenced objects, 2 core, 3 the unreferenced stack, 4 unreferenced hands and graveyards. Run-level, never per collector — a depth that varies by kind turns `state.tiers` into a proxy for how a record was collected. Stage three collects at `1,2,3,4`. |

## 3b. Collect the validation corpus at full strength

```bash
python -m sealed match-outcomes \
    --effect-records output/effects/records/full-strength/ --workers 6 \
    --snapshot-tiers 1,2,3,4
```

The same command as step 3 without `--exclude-cards`, so these pools contain the held-out cards and
these games become the card-disjoint stratum. Nothing marks the shards as validation: the trainer
reserves every shard holding a held-out card, so the two runs separate themselves.

This run is much smaller than step 3. It only has to satisfy `--min-holdout-records` unique-text
resolution records, which is the number `train-effect-model` warns about. Collect it after step 3 so
you can size it against what the holdout actually covers.

## 4. Collect the cards self-play never deals

```bash
python -m effects collect-coverage \
    --training-corpus output/effects/records/depleted/ \
    --target-records 50 --workers 12 \
    --interventions-per-game 2 \
    --probes-per-game 2 \
    --probe-keywords first_strike,double_strike,deathtouch,lifelink,trample,indestructible,wither,infect
```

Sealed self-play only ever plays what sealed pools contain, so a card in no sealed-legal set never
appears in a record at all. Decks are built over the whole converted corpus, weighted toward the cards
with the fewest records, and rounds play until every card is satisfied or retires after
`--no-progress-rounds` without a new qualifying record — which is what makes the run terminate.

`--training-corpus` names one directory holding both the corpus this run extends and the
`holdout-cards.txt` it was depleted against — shorthand for `--effect-records DIR` plus
`--exclude-cards DIR/holdout-cards.txt`. The two belong together, so a directory without a list is
refused rather than run undepleted, and passing either implied flag beside it is refused too.

Point it at the **depleted** corpus rather than the tree above it. Coverage counts records per card
to decide what still needs collecting, and shard discovery recurses, so a directory holding the
full-strength corpus as well would count validation records toward coverage — a card whose only
records are held out would read as satisfied while the model never learns from it.

Reading that list is what keeps held-out cards out of every deck; without it a coverage run puts them
into training games and contaminates the split of the model it feeds. Pass `--split-from CHECKPOINT`
instead only when adding coverage to a corpus an existing checkpoint was trained on, whose recorded
split is then the authority.

Every worker here collects effect records unconditionally (there is no plain mode to fall back to,
unlike step 3), so it writes the same `{run_id}.{worker}.log` per-worker logs described in step 3's
checks, into the same `--effect-records` directory — a fresh `run_id` per invocation, so this
command's logs and step 3's never collide even when both target `output/effects/records/`.

The run reports two residues — cards judged uncastable, and castable cards short of `--target-records`.
An **interventional resolution** is the answer to both: it forks the game at a phase boundary, puts an
ability nobody played on the fork's stack, and records what it did. It goes on the stack rather than
through the cost machinery, because the cost is usually why the record is missing — an ability the AI
never used is mostly one it could never afford, and paying for it would fail on exactly the population
the intervention exists to reach. Lands and mana abilities are skipped: every game plays them, so
forking to force one spends a game copy on the commonest event in the corpus.

Every fork is score-checked against the live game before any perturbation, a discarded fork still
counts against its budget, and at most two forks may target one real resolution.

## 5. Collect synthetic variants

The script surface is a second vocabulary and a second cache, side by side with the prose ones:

```bash
python -m effects build-vocab --surface script          # models/effects/vocab-script.txt

python -m effects collect-variants \
    --training-corpus output/effects/records/depleted/ \
    --variant-volume 0.2
```

`collect-variants` reads Forge's **source** scripts, perturbs one whitelisted parameter per variant,
writes a sidecar for each through `VariantSidecarMain`, and plays them. `--variant-volume` caps
variant records as a fraction of the real records already present, so the cap scales with the corpus.
No variant is generated from a card carrying a held-out text. Variant scripts land in
`output/effects/variant-scripts/` and are never converted to prose — the perturbation is on the script
surface only.

## Verify the corpus before training

```bash
python -m effects field-coverage --effect-records output/effects/records
```

Names every record field the corpus never varied. A field written as a fixed literal and one a run
happened not to exercise look identical from a corpus, so the report separates them by a checked-in
list: `[known ]` is expected, `[NEW   ]` is either newly broken or newly rare, and only the writer
says which. A "listed as constant but carrying data" section means a field was implemented and the
list is stale.

On a full run expect `[known ]` on exactly the two fields under
[What is not collected](#what-is-not-collected). Everything else appearing as `[NEW   ]` is worth a
look before spending hours training against it — a rare counter type absent from a short run is
normal, a whole payload channel constant is not.

## Build the curated dataset, if runs have to be comparable

Two training runs a week apart do not read the same corpus. An epoch walks the sorted shard list, a
shard's filename begins with its collection run's UUID, and every run since inserts shards at an
arbitrary position — so which half of the corpus a run reads moves under it. Rarity is counted over
whichever shard is resident, which inverts on step 4's shards: those are built dense in scarce cards,
so inside one of them a scarce card looks common. `build-corpus` freezes all of it into a dataset
that many runs can read:

```bash
python -m effects build-corpus \
    --records-dir output/effects/records/ \
    --output output/effects/corpus/
```

Two passes. A parallel survey reads every shard once, then one process folds provenance keys to
ability texts, decides the split, computes the rarity table corpus-wide, and a second parallel pass
writes `training/`, `validation/card-disjoint/`, `validation/game-disjoint/` and `manifest.json`.
Training records are selected per record; both validation strata are selected per **game**, which is
what keeps a probe and the combat record its `mirror_of` names in the same stratum.

**Read the delivered-against-requested table it prints.** The manifest records the mixture the run
asked for and the mixture the data actually holds, and they are not identical: availability is counted
over every game while admission applies only to training candidates. The gap is a few percent, and it
is in the manifest rather than left to be discovered.

The run refuses rather than writing in two cases, both of which produce a dataset no training run will
accept. Nothing held out means the card-disjoint stratum would be empty and gate 1 would have nothing
to measure — usually the relative `--cards-folder` paths failing to resolve from somewhere other than
the repository root. No game naming a held-out card means the same outcome by a different route, and
on this corpus it usually means `--records-dir` was pointed at `records/depleted/` rather than
`records/`, so the full-strength shards from step 3b were never read.

`--text-cap` (default 200) is the ceiling on records per unique ability text, and it is a ceiling and
never a floor: a text below it keeps everything it has, so step 4's tail survives curation whole. The
scarcest sampling class sets the size of the whole dataset, and that class is `rewrite` by a wide
margin; the distribution is in the [design record](../../experiments/2026-09-04-ability-effect-model-design.md). `--verify` reports the manifest's drift against
the corpus on disk and writes nothing; a corpus that has grown is rebuilt whole, because the split and
the rarity table are both corpus-wide quantities.

Build the dataset **after** step 4 finishes, not during it. The dataset freezes whatever it is given,
and what step 4 is still fixing is how thinly the tail is observed.

## 6. Train, and train the baselines

```bash
python -m effects train-effect-model \
    --variant-scripts output/effects/variant-scripts/ \
    --withhold-keyword cascade
```

Add `--corpus output/effects/corpus/` to train against a curated dataset instead of the raw corpus.
It takes the split, both validation strata and the rarity table from the manifest, so it refuses
`--records-dir`, `--reserved-shards`, `--split-from`, `--holdout-permille` and `--holdout-max-carriers`
alongside it — each of those names a decision the manifest already records, and two spellings of one
decision is a disagreement nothing would report. It also refuses a dataset built on the other encoding
surface, because the rarity table's keys are the texts of that surface and a table read on the wrong
one matches nothing while looking entirely valid.

`--withhold-keyword` holds one implemented keyword's token out of training so the zero-shot check in
step 8 has something to measure; its occurrences are always expanded instead. Any implemented keyword
outside gate 2's eight damage-step keywords works; `cascade` is just an example. Withholding one of
the eight would degrade that keyword's own gate-2 verdict in the same run.

Trains both transformers jointly from random init. The mixture renormalizes over the sampling classes
actually present, and an absent kind's fields contribute no loss — so the same command works on a
three-class stage-one corpus and an eight-class one.

**Read the first line it prints.** It reports the holdout — how many ability texts it holds out and
what share of `output/cardsfolder/` cards carry one — then the three strata's record counts. The
holdout is keyed on the text, not the card: a text is eligible when at most `--holdout-max-carriers`
cards carry it, and an eligible text is held out on a hash of its own bytes, so functional reprints
are held out together and a Forge upgrade never reassigns an existing text.

The run fails rather than trains when the card-disjoint stratum is empty, and warns when it holds
fewer than `--min-holdout-records` unique-text resolution records. Both mean the same thing: step 3b
was skipped or was too short, so there are no full-strength games to validate on.

On an 8 GB card, `--context-cache` is the documented fallback: it swaps live context re-encoding for a
stop-gradient momentum cache refreshed every `--cache-refresh` batches.

All four baselines are needed, because every reported check runs against one: gate 1 compares against
`identity`, every record kind reports `state-only` as its floor, the average-effect control is
`no-state`, and the taxonomy comparison is `taxonomy`. Each inherits the split and holds out the same
keyword, so the only difference from the shipping model is the input its variant masks:

```bash
for V in identity state-only no-state taxonomy; do
  python -m effects train-effect-model --variant $V \
      --split-from models/effects/effect-model/latest.pt \
      --variant-scripts output/effects/variant-scripts/ \
      --withhold-keyword cascade
done
```

Pass the **same** `--holdout-permille` and `--holdout-max-carriers` here as in step 3. The corpus was depleted against those values; a run that computes a different holdout would train on cards it believes are held out, and nothing would say so. The checkpoint records them.

These write under `models/effects/effect-model/{variant}/`, never over the shipping checkpoint. A
variant run must inherit the split it is a baseline for, so it fails fast without either
`--split-from` or `--corpus` rather than silently computing its own and making the comparison
meaningless. Against a curated dataset, pass `--corpus` to every baseline and drop `--split-from`:
the manifest enumerates the split directly, so every run reading that dataset trains on the same
games by construction.

**For the script surface** add `--vocab-path models/effects/vocab-script.txt` to *every* command in
steps 6, 7 and 8 — training, the four baselines, `encode-abilities` and `evaluate-effect-model` all
take it. The surface follows the loaded vocabulary rather than a flag of its own, so the two cannot
disagree; but a cache encoded under one vocabulary and evaluated under another is a mismatch nothing
catches for you.

**Checks**: the checkpoint under `models/effects/effect-model/` records its held-out card list, its
`game_id` set across both strata, and the vocabulary and keyword-definition paths plus their hashes.

## 7. Encode the cache

The `e`-geometry checks (gate 3, the decodability battery, the ward canary, the scorer smoke test)
read `full`, `no-state`, and `taxonomy`. `identity` and `state-only` are prediction baselines and
need no cache, so encoding them is optional:

```bash
python -m effects encode-abilities --variant-scripts output/effects/variant-scripts/
for V in no-state taxonomy; do
  python -m effects encode-abilities --variant $V \
      --variant-scripts output/effects/variant-scripts/
done
```

**Checks**: `output/effects/abilities/cardsfolder/…` and `…/tokenscripts/…` mirror their source trees;
each file is `(n_lines, e_dim)` and row-aligned with that source's sidecar. Each variant run wrote
`<name>.{variant}.npz` beside the shipping `<name>.npz` rather than replacing it. The `taxonomy`
variant has no encoder, so its file is the taxonomy lookup emitted into the same row layout — every
`e`-geometry check reads one file shape.

## 8. Evaluate

```bash
python -m effects evaluate-effect-model \
    --variant-scripts output/effects/variant-scripts/ \
    --variant-checkpoint identity=models/effects/effect-model/identity/latest.pt \
    --variant-checkpoint state-only=models/effects/effect-model/state-only/latest.pt \
    --variant-checkpoint no-state=models/effects/effect-model/no-state/latest.pt \
    --variant-checkpoint taxonomy=models/effects/effect-model/taxonomy/latest.pt
```

Splits come from the checkpoint — never a flag, never recomputed. A baseline recording a different
split, or different vocabulary hashes, fails the run rather than being compared. A checkpoint trained
with `--corpus` also records that dataset's digest, and `--corpus` here defaults to the path it
recorded; a dataset rebuilt since fails the run, because a rebuild is a different split and the gates
would be scored partly on games the model trained on. The command exits
non-zero when a blocking gate fails.

**What you get, and what to do with it**:

| Result | Meaning |
|---|---|
| **Gate 1** (identity baseline) | Blocks shipping. All three margins must hold on the card-disjoint split's unique-text stratum — resolution records whose acting text appears on no training card, the only slice where a free embedding per text cannot recall the answer. Failure means the encoder is not reading text. |
| **Gate 3** (collapse canaries) | Blocks shipping. Mean pairwise cosine ≤ 0.5 over 10,000 pairs, top PC ≤ 30% of variance. |
| **Gate 2** (damage-step canary), per keyword | Blocks nothing. Each of the eight keywords passes or is routed to a probe. On a corpus collected with `--probe-keywords`, each routed keyword has engine-side branches to check the model-side perturbation against. |
| Ward canary, nearest-neighbour / UMAP, decodability battery, scorer smoke test | Reported, not gating |

The decodability battery needs `output/sealed/cards-win-rates.txt` from a `train-encoder` run; without
it that check is skipped rather than failed.

## Collecting less

Every step above can be dropped, and the ones after it still run:

| Drop | Cost |
|---|---|
| Step 0, the engine patch | Three of eight sampling classes, `degraded` mode, no forks. Evasion keywords will not separate — their signal is playability records. |
| `--probe-keywords` in steps 3–4 | Gate 2 still reports a per-keyword verdict; a routed keyword has no engine-side branch to check against. |
| `--interventions-per-game 0` | Cards the AI can never afford to play keep no resolution record. Omitting the flag does not do this: it defaults to 2, so interventions run anyway. |
| Step 3b, the full-strength run | No card-disjoint stratum, so `train-effect-model` fails before its first epoch. Gate 1 has nothing to measure. |
| `--exclude-cards` in step 3 | Held-out cards reach training games, and every game holding one is discarded instead — most of the corpus at a useful holdout size. The run looks identical while it happens. |
| Step 4, `collect-coverage` | Cards in no sealed-legal set appear in no record at all. |
| Step 5, `collect-variants` | No script-surface corpus; the prose surface trains as before. |
| `build-corpus` | Training still runs against the raw corpus. Each run reads a different half of a corpus that keeps growing, counts rarity over the resident shard, and enumerates its own split — so two runs differ by more than the hyperparameters under test, and an architecture comparison cannot be read. |

## What is not collected

Two fields in the record schema are deliberately empty, and `field-coverage` lists both as `[known ]`:

- **A `continuous` contribution's `name`.** Eight cards in all of Forge write the name layer from a
  static, the head has no field for a name, and a name is open-vocabulary over 33,680 strings unlike
  every other channel.
- **A `playability`/`decision` candidate's `responsible_static`.** It would need a `cantBeCastStatic`
  hook Forge does not have, and even where the equivalent exists for attackers and blockers only about
  a quarter of forbidden creatures find a static — the rest are stopped by the rules themselves, where
  an empty list is the right answer.

## Where results go

Run results belong in the design record's Outcome section
([`../../experiments/2026-09-04-ability-effect-model-design.md`](../../experiments/2026-09-04-ability-effect-model-design.md)),
never in the root spec and never here. Record gate 1's three margins, gate 3's two geometry figures,
and gate 2's per-keyword table — the routed list is the input to the stage-three decision.

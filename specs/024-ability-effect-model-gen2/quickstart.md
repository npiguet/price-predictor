# Quickstart: Ability effect model — generation 2

**Feature**: `024-ability-effect-model-gen2`
Run order and rationale: the gen-2 record's run plan,
[`../../experiments/2026-09-18-effect-model-gen2-improvements-design.md`](../../experiments/2026-09-18-effect-model-gen2-improvements-design.md).
Base walkthrough (prerequisites, the hooks branch, cap flags): [`../023-ability-effect-model/quickstart.md`](../023-ability-effect-model/quickstart.md).

Commands below are the ones this feature adds or changes. Paths are the defaults.

## Prerequisites

- `../forge` on `effect-record-hooks`, built with `mvn install -DskipTests`; the connector JAR rebuilt
  with `cd forge-connector && mvn install -DskipTests`.
- The gen-1 checkpoint at `models/effects/runs/2026-09-17-full-textless-corpus/latest.pt`, its
  curated corpus, and its raw shards.

## Stage 0 — gen-1 artifacts (can run beside stages 1–3)

Keep gen-1's sidecars, keyword definitions and script vocabulary before stage 1 overwrites them:

```bash
cp -r output/cardsfolder output/gen1-cardsfolder
cp -r output/tokenscripts output/gen1-tokenscripts
cp output/effects/keyword-definitions.json output/effects/gen1-keyword-definitions.json
cp models/effects/vocab-script.txt models/effects/gen1-vocab-script.txt
```

A checkpoint records its vocabulary and keyword-definition paths with their content hashes, and every
loader re-hashes them. After stage 1 rewrites the originals, gen-1 and the noise-pilot checkpoints
load only with `--vocab-path models/effects/gen1-vocab-script.txt` (gen-1) and
`--keyword-definitions output/effects/gen1-keyword-definitions.json` (both).

Knowledge probes on gen-1:

```bash
python scripts/effect_knowledge_probes/run.py \
    --checkpoint models/effects/runs/2026-09-17-full-textless-corpus/latest.pt \
    --cards-folder output/gen1-cardsfolder --cards-folder output/gen1-tokenscripts \
    --freeze-probe-set
python scripts/effect_knowledge_probes/run.py \
    --checkpoint models/effects/runs/2026-09-17-full-textless-corpus/latest.pt \
    --cards-folder output/gen1-cardsfolder --cards-folder output/gen1-tokenscripts
```

Noise pilot, three short runs on gen-1's curated corpus (value, verdict, created-objects, MLM and API
heads on, one-segment chains). A pilot trains from scratch under the gen-2 tokenizer rules, so it
reads a script vocabulary rebuilt with those rules over gen-1's sidecars; gen-1's own
`vocab-script.txt` stays as the gen-1 checkpoint recorded it:

```bash
python -m effects build-vocab --surface script \
    --cards-folder output/gen1-cardsfolder --cards-folder output/gen1-tokenscripts \
    --vocab-path models/effects/vocab-script-gen1-sidecars.txt
for r in 0.05 0.1 0.2; do
  python -m effects train-effect-model --corpus output/effects/corpus/ \
      --vocab-path models/effects/vocab-script-gen1-sidecars.txt \
      --cards-folder output/gen1-cardsfolder --cards-folder output/gen1-tokenscripts \
      --e-noise $r --value-weight 0.05 --epochs 3 \
      --model-output models/effects/runs/2026-10-noise-pilot-$r/
done
```

The gen-1 corpus records the `text` holdout unit, which the trainer takes as its default. Pass the
same `--cards-folder` pair to `encode-abilities` and `evaluate-effect-model` when running them on a
pilot checkpoint.

Read the epoch lines: rarity-bucket and family shares, and each loss term. Pick the ratio for the
sweep and record it in the gen-2 record.

## Stage 1 — encoding text and holdout (before any gen-2 game)

```bash
python -m effects extract-keyword-definitions
python -m price_predictor convert
python -m effects build-vocab --surface script
python -m effects holdout-cards --holdout-unit template --out output/effects/holdout-cards.txt
```

No pools file is generated: each `match-outcomes` worker opens its own pool per match and applies
`--exclude-cards` there.

Check: `convert` reports missing SVars; `.txt` files diff clean against the previous conversion
except where an ordering fix changed a line's or a face's order;
`build-vocab` prints the length distribution and seeded counts and does not fail on `[UNK]`;
`holdout-cards` prints templates, texts, cards and the depleted share.

## Stage 2 — pilot collection

```bash
python -m sealed match-outcomes --effect-records output/effects/records-pilot/ \
    --exclude-cards output/effects/holdout-cards.txt \
    --random-seat-share 0.25 --random-seat-probability 0.5
# stop it (Ctrl-C) once the progress line shows a few hundred games
python -m effects validate-corpus --effect-records output/effects/records-pilot/
python -m effects field-coverage --effect-records output/effects/records-pilot/
python -m effects build-corpus --records-dir output/effects/records-pilot/ \
    --output scratch/corpus-pilot/ --vocab-path models/effects/vocab-script.txt
```

Check: both envelope fields vary; `match-outcomes.txt` and `cards-played.txt` gained rows only for
matches without a random seat; every real legality decision is present; a charm resolution has one
effect half per chosen mode; the legality class holds real decisions only, since what-if queries
are not collected at the default `--legality-rate` of 0; the manifest's per-family shortfalls,
signatures and on/off-policy counts look as intended. The real block decisions against the random
seat's attacks are what replaces the what-if queries, so raise `--random-seat-probability` if they
are too few.

## Stage 3 — full collection

```bash
# depleted self-play with the random seat, for training; stop at 9,860 match rows
python -m sealed match-outcomes --effect-records output/effects/records/depleted/ \
    --exclude-cards output/effects/holdout-cards.txt \
    --random-seat-share 0.25 --random-seat-probability 0.5 \
    --interventions-per-game 4 --probes-per-game 4
# full strength with the random seat, for the card-disjoint stratum; stop at 3,150 match rows
python -m sealed match-outcomes --effect-records output/effects/records/full-strength/ \
    --random-seat-share 0.25 --random-seat-probability 0.5
# coverage of uncovered training cards
python -m effects collect-coverage --training-corpus output/effects/records/depleted/ \
    --interventions-per-game 4 --probes-per-game 4
# held-out texts to their floor
python -m effects collect-coverage --only-cards output/effects/holdout-cards.txt --min-text-games 5 \
    --interventions-per-game 4 --probes-per-game 4 \
    --effect-records output/effects/records/held-out-coverage/
# synthetic variants, one round
python -m effects collect-variants --effect-records output/effects/records/variants/ \
    --exclude-cards output/effects/holdout-cards.txt \
    --corpus-records output/effects/records/depleted/ --decks-per-round 4000
```

- **Run order.** As listed. `collect-coverage --training-corpus` reads `records/depleted/holdout-cards.txt`, so
  copy the holdout list there before the first run.
- **Stopping the self-play runs.** `match-outcomes` plays until interrupted. Count rows in
  `output/sealed/match-outcomes.txt` and stop at the target. After each run, append
  `output/sealed/match-outcomes.txt` and `cards-played.txt` to the same-named files in that run's
  records directory and delete the originals, so the next run counts from zero and the sealed corpus
  never receives these rows.
- **Fork budgets.** The training-side runs (depleted self-play, both coverage passes) take 4
  interventions and 4 probes per game. Full-strength keeps the defaults of 2 and 2.
- **Flags left at their defaults.** Every record carries snapshot tiers 1–4, and the probes cover all
  eight damage-step keywords, with no flag needed for either. `--legality-rate` stays at 0.
- **Variant round size.** One `collect-variants` invocation plays one round of `--decks-per-round`
  decks, 23 variant slots each, drawn evenly over about 16,000 variants. At the default of 500 about
  half the variants reach no deck and yield no record; 4000 puts nearly every variant in several decks.

## Stage 4 — curated dataset

```bash
python -m effects build-corpus --records-dir output/effects/records/ \
    --output output/effects/corpus-gen2/ \
    --variant-scripts output/effects/variant-scripts/ \
    --vocab-path models/effects/vocab-script.txt \
    --holdout-unit template
```

Send the manifest's shortfall list back to `collect-coverage` / `collect-variants`, then rebuild into
the same `--output`. The rebuild keeps every game-disjoint game the first build placed.

## Stage 5 — per sweep arm

```bash
python -u scripts/effect_sweep_arm.py --run-dir models/effects/runs/2026-10-gen2-<arm>/ \
    --corpus output/effects/corpus-gen2/ [--freeze-probe-set] -- \
    --e-dim D --encoder-layers L --encoder-d-model W --e-noise R --withhold-keyword K
```

Every arm keeps the trainer's batch size (32) and steps per epoch (5,000).

`effect_sweep_arm.py` runs, in order: `train-effect-model` with the flags after `--`,
`encode-abilities`, `evaluate-effect-model`, the eight scripts in `scripts/effect_embedding_probes/`
(`build_texts` first), and `scripts/effect_knowledge_probes/run.py`. It passes `--variant-scripts`
to training, encoding and evaluation, and `--vocab-path` to training, defaulting to the paths
`build-corpus` is given in stage 4. Pass `--freeze-probe-set` on
the first arm only; that arm freezes the knowledge-probe set for the corpus before probing.

The run directory holds the arm's whole record:

| Path | Contents |
|---|---|
| `latest.pt`, `<timestamp>.pt` | checkpoints |
| `arm.log` | the runner's command line, each step's start, and the step a failure stopped at |
| `train.log`, `encode.log`, `evaluate.log` | each command's full output |
| `probes-<script>.log`, `knowledge-probes.log` | each probe's full output |
| `abilities/` | the ability cache this checkpoint encoded |
| `reports/embedding-probes-*/`, `reports/knowledge-probes-*/` | the probe reports and the scorecard |

Every log opens with the command line that produced it and the commit it ran at. A rerun appends to
the log under its own command line. The frozen probe set is shared by every arm and stays in
`output/effects/reports/`. After a failure, rerun with `--from-step` naming the failed step
(`encode`, `evaluate`, `embedding-probes` or `knowledge-probes`). The runner refuses to train into a
directory that already holds `latest.pt`.

Compare arms:

```bash
python scripts/effect_knowledge_probes/compare.py models/effects/runs/2026-10-gen2-*/reports/knowledge-probes-*/scorecard.json
```

Results go into the gen-2 and probes records' Outcome sections, never into a spec.

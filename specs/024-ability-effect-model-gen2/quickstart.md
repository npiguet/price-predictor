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

Keep gen-1's sidecars before stage 1 reconverts them:

```bash
cp -r output/cardsfolder output/gen1-cardsfolder
cp -r output/tokenscripts output/gen1-tokenscripts
```

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
python -m sealed generate-pools --exclude-cards output/effects/holdout-cards.txt
```

Check: `convert` reports missing SVars; `.txt` files diff clean against the previous conversion;
`build-vocab` prints the length distribution and seeded counts and does not fail on `[UNK]`;
`holdout-cards` prints templates, texts, cards and the depleted share.

## Stage 2 — pilot collection

```bash
python -m sealed match-outcomes --effect-records output/effects/records-pilot/ \
    --exclude-cards output/effects/holdout-cards.txt \
    --random-seat-share 0.125 --random-seat-probability 0.25
# stop it (Ctrl-C) once the progress line shows a few hundred games
python -m effects validate-corpus --records-dir output/effects/records-pilot/
python -m effects field-coverage --records-dir output/effects/records-pilot/
python -m effects build-corpus --records-dir output/effects/records-pilot/ \
    --output scratch/corpus-pilot/ --vocab-path models/effects/vocab-script.txt
```

Check: both envelope fields vary; `match-outcomes.txt` and `cards-played.txt` gained rows only for
matches without a random seat; every real legality decision is present; a charm resolution has one
effect half per chosen mode; the manifest's per-family shortfalls, signatures, policy and
real/what-if counts look as intended. Raise `--random-seat-probability` only if minority outcomes
are too few.

## Stage 3 — full collection

```bash
# depleted self-play with the random seat, for training
python -m sealed match-outcomes --effect-records output/effects/records/ \
    --exclude-cards output/effects/holdout-cards.txt \
    --random-seat-share 0.125 --random-seat-probability P
# full strength with the random seat, for the card-disjoint stratum
python -m sealed match-outcomes --effect-records output/effects/records/ \
    --random-seat-share 0.125 --random-seat-probability P
# coverage of uncovered training cards
python -m effects collect-coverage --training-corpus …
# held-out texts to their floor
python -m effects collect-coverage --only-cards output/effects/holdout-cards.txt --min-text-games 5
python -m effects collect-variants
```

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
python -m effects train-effect-model --corpus output/effects/corpus-gen2/ \
    --e-dim D --encoder-layers L --encoder-d-model W --e-noise R \
    --withhold-keyword K --model-output models/effects/runs/2026-10-gen2-<arm>/
python -m effects encode-abilities --checkpoint models/effects/runs/2026-10-gen2-<arm>/latest.pt
python -m effects evaluate-effect-model --checkpoint models/effects/runs/2026-10-gen2-<arm>/latest.pt
for s in build_texts effect_profiles pca_directions lexical_probes linear_probes type_merging neighbours keyword_expansion; do
  python scripts/effect_embedding_probes/$s.py \
      --checkpoint models/effects/runs/2026-10-gen2-<arm>/latest.pt --abilities-root output/effects/abilities
done
python scripts/effect_knowledge_probes/run.py --checkpoint models/effects/runs/2026-10-gen2-<arm>/latest.pt --freeze-probe-set   # first arm only
python scripts/effect_knowledge_probes/run.py --checkpoint models/effects/runs/2026-10-gen2-<arm>/latest.pt
python -m effects scorer-smoke-test --checkpoint models/effects/runs/2026-10-gen2-<arm>/latest.pt \
    --scratch-dir scratch/smoke-<arm>/
```

Compare arms:

```bash
python scripts/effect_knowledge_probes/compare.py output/effects/reports/knowledge-probes-*/scorecard.json
```

Results go into the gen-2 and probes records' Outcome sections, never into a spec.

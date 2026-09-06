# Contract: CLI surface

**Feature**: `023-ability-effect-model`
Authority: [`../../2026-09-05-ability-effect-model.md`](../../2026-09-05-ability-effect-model.md) § CLI summary.

`python -m effects <subcommand>`, plus one flag added to an existing `sealed` subcommand. Defaults
below are contract, not suggestions.

## `python -m effects build-vocab`

| Flag | Default | Meaning |
|---|---|---|
| `--surface` | `prose` | `prose` \| `script`; `script` adds the sidecars' script lines to the scan and switches the `--vocab-path` default |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` | repeatable |
| `--vocab-path` | `models/effects/vocab.txt` (`vocab-script.txt` under `--surface script`) | separate paths so a stage-four rebuild never overwrites the vocabulary stage-one-to-three checkpoints recorded |
| `--keyword-definitions` | `output/effects/keyword-definitions.json` | included in the scan |
| `--target-size` | 5000 | post-truncates the corpus-frequency vocabulary, preserving seeded specials and domain tokens |

Seeded specials: `[PAD]`, `[UNK]`, `cardname`, `[MASK]`, `[CLS]`.

## `python -m effects extract-keyword-definitions`

| Flag | Default |
|---|---|
| `--output` | `output/effects/keyword-definitions.json` |

Writes keyword → reminder-text template for every keyword; from stage four also the generated
implementation script, captured as text at the keyword factory, for the script-generated majority.

## `python -m sealed match-outcomes` (existing command, one added flag)

| Flag | Default | Meaning |
|---|---|---|
| `--effect-records` | **none** | the instrumentation opt-in. Absent, no effect records are written anywhere and the command behaves exactly as today |
| cap/budget flags | see below | shared with the other collectors |

Adding the flag must not change `match-outcomes.txt` or `cards-played.txt` in format or content.

## `python -m effects collect-coverage`

| Flag | Default | Meaning |
|---|---|---|
| `--effect-records` | `output/effects/records/` | destination shard directory |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` | repeatable; deck candidates and the coverage unit come from the `output/cardsfolder/` entry alone |
| `--split-from` | none | a checkpoint whose held-out cards are excluded from every deck |
| `--target-records` | 50 | per-card satisfaction goal |
| `--decks-per-round` | 500 | decks played as matches per round |
| `--no-progress-rounds` | 3 | consecutive rounds without a new qualifying record before a card retires |
| cap/budget flags | see below | |

## `python -m effects collect-variants` (stage four)

| Flag | Default |
|---|---|
| `--effect-records` | `output/effects/records/` |
| `--forge-cards-path` | `../forge/forge-gui/res/cardsfolder/` |
| `--variant-volume` | 0.2 (cap on variant records as a fraction of real records already present) |
| `--decks-per-round` | 500 |
| cap/budget flags | see below |

## Cap and budget flags (every collecting supervisor)

| Flag | Default | Meaning |
|---|---|---|
| `--mana-cap` | 2000 | resolution records per unique mana-ability text, per worker process |
| `--playability-rate` | 0.1 | fraction of `decision`-subkind logging points sampled; `attackers`/`blockers` always logged |
| `--interventions-per-game` | 2 | interventional resolutions per game (stage three) |
| `--probes-per-game` | 2 | damage-step probe forks per game |
| `--probe-keywords` | none (probes disabled) | comma-separated canary-failing keywords |

`continuous` records take no cap: coalescing per stable board is the cap.

## `python -m effects train-effect-model`

The full flag table is the root spec's § Training. Contract highlights:

| Flag | Default |
|---|---|
| `--records-dir` | `output/effects/records/` |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` |
| `--variant-scripts` | none (stage four: `output/effects/variant-scripts/`) |
| `--split-from` | none (compute the split); **required for variant runs** |
| `--vocab-path` | `models/effects/vocab.txt` |
| `--printings-path` | `resources/AllPrintings.json` |
| `--keyword-definitions` | `output/effects/keyword-definitions.json` |
| `--model-output` | `models/effects/effect-model/` for `--variant full`, `models/effects/effect-model/{variant}/` otherwise |
| `--variant` | `full` (\| `identity` \| `state-only` \| `no-state` \| `taxonomy`) |
| `--e-dim` / `--e-noise` | 64 / 0.05 |
| `--keyword-expand-p` / `--context-dropout` | 0.25 / 0.15 |
| `--mlm-weight` / `--mlm-mask-prob` / `--api-weight` | 0.1 / 0.15 / 0.05 |
| `--curriculum-step` | 10000 |
| `--batch-size` / `--grad-accum` | 32 / 1 |
| `--kind-mix` | the eight-class mixture |
| `--context-cache` / `--cache-refresh` | off / 500 |
| `--steps-per-epoch` / `--epochs` / `--patience` | 5000 / 40 / 5 |

Hardcoded, not flags: encoder d_model 256 / 4 layers / 4 heads; trunk d_model 256 / 6 layers / 4 heads;
`ff_dim` 4 × d_model; dropout 0.1; AdamW; lr 1e-4 constant after warmup; warmup over the first 5% of
`--epochs` × `--steps-per-epoch`; per-parameter-group gradient clip 1.0; seed 42.

## `python -m effects encode-abilities`

| Flag | Default |
|---|---|
| `--variant` | `full` — resolves both `--checkpoint` and the output suffix, so a variant is never encoded with another variant's weights |
| `--checkpoint` | the variant's `latest.pt`; an explicit value overrides |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` |
| `--variant-scripts` | stage four |
| `--vocab-path`, `--keyword-definitions` | the paths the checkpoint recorded from training |
| `--clean` | removes only files this command wrote |

Output: `output/effects/abilities/<tree>/…`, `<name>.npz` for `full` and `<name>.{variant}.npz`
otherwise, beside rather than replacing the shipping cache.

## `python -m effects evaluate-effect-model`

| Flag | Default |
|---|---|
| `--checkpoint` | `models/effects/effect-model/latest.pt` |
| `--variant-checkpoint NAME=PATH` | repeatable |
| `--records-dir`, `--cards-folder`, `--variant-scripts` | the trainer's defaults |
| `--vocab-path`, `--keyword-definitions` | the paths `--checkpoint` recorded |

**Splits are never a flag.** The held-out card list and `game_id` set come from `--checkpoint`. The
command fails fast when a `--variant-checkpoint` records a different split, or different vocabulary or
keyword-definition hashes, than `--checkpoint`.

## Exit-code and failure contract

| Condition | Behaviour |
|---|---|
| Vocabulary or keyword-definition hash mismatch | fail fast with the recorded vs actual hash |
| `--variant-checkpoint` split disagreement | fail fast naming both checkpoints |
| Variant training run without `--split-from` | fail fast |
| Missing connector JAR | the existing `forge_jvm` error naming the `mvn package` command |
| Stock (unpatched) Forge | **not** a failure — degrade, stamp `mode = degraded` |
| Trailing partial line in a shard | **not** a failure — skip it |

# Contract: CLI surface (delta)

**Feature**: `024-ability-effect-model-gen2`
Authority: [`../../2026-10-06-ability-effect-model-gen2.md`](../../2026-10-06-ability-effect-model-gen2.md) § 12.
Base: [`../../023-ability-effect-model/contracts/cli.md`](../../023-ability-effect-model/contracts/cli.md).

Only flags and behaviour this feature adds, removes or changes. Every other flag keeps the base
contract. Defaults are contract. Every refusal below happens before any game, training step or
file write, and exits non-zero with a message naming the offending flags.

## `python -m price_predictor convert` (existing)

No new flag. Behaviour:

- Sidecar `script_text` holds the chained, label-renamed encoding text (FR-001–003).
- Charm-mode `option` lines carry their mode key, chain and API type (FR-005a).
- Reports, once per run: every referenced SVar missing from its script (card, line, label).
- Converted `.txt` output is byte-identical to the previous conversion.

## `python -m effects extract-keyword-definitions` (existing)

No new flag. Each entry gains `formatter` (the Forge `Keyword.type` simple name). `generated_script`
is still written but no longer read by `build-vocab`.

## `python -m effects build-vocab` (existing)

No new flag. Under `--surface script`:

- seeds `[SEG]`, `sv1…svN` (N = the longest chain's label count), every template word after
  specifier removal, and every part of every parameter key and `$`-prefix the staged lines carry;
- stages script lines with the script-surface tokenization rules, before the prose files;
- reports: script-line length distribution in tokens (min, p50, p90, p99, max, and the count over
  512); unknown-token rate over script parameters with `*Description$` values removed; distinct
  camel-case parts; seeded-token counts by kind;
- fails when any keyword definition expands to a text containing `[UNK]`.

## `python -m effects holdout-cards` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--holdout-unit` | `template` | `template` \| `text`; `text` reproduces feature 023's rule |

Report adds: held-out templates, held-out texts, held-out cards, share of converted cards depleted.

## Records-set generation (every command that reads a records set)

`build-corpus`, `train-effect-model`, `evaluate-effect-model` and `run.py` refuse, before reading
records, a records set holding both gen-1 shards (no `random_seat`) and gen-2 shards.

## `python -m sealed match-outcomes` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--random-seat-share` | 0 | share of matches with one random seat |
| `--random-seat-probability` | none | `P`, the chance the random seat draws at random at a decision point |

Refuses: `--random-seat-share` > 0 without `--random-seat-probability`; > 0 without
`--effect-records`; either value outside [0, 1]. Passed to workers as `-Deffect.random.seat.share`
and `-Deffect.random.seat.probability`.

A random-seat match writes effect records and the progress line only.

## `python -m effects collect-coverage` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--only-cards` | none | the `holdout-cards` list; decks from listed cards plus basics; unit = held-out text; full-strength caps |
| `--min-text-games` | 5 | distinct games a held-out text must act in |

Refuses `--only-cards` beside `--exclude-cards`, `--training-corpus` or `--split-from`. Ends when
every held-out text with a castable carrier (by feature 023's castability consult) is satisfied or
retired; reports the texts under the floor and, separately, the texts with no castable carrier.

## `python -m effects build-corpus` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--holdout-unit` | `template` | as `holdout-cards` |
| `--game-disjoint-share` | 0.01 | per-game hash threshold |
| `--game-disjoint-keyword-share` | 0.15 | threshold for a game holding a qualifying combat for a listed keyword |
| `--game-disjoint-keywords` | `first strike,deathtouch,trample,indestructible,wither,infect` | comma list |
| `--reuse-cap` | 4 | maximum copies of one record toward a short family's share |
| `--text-cap` | 200 (unchanged) | maximum written records per text per cell, repeats included (meaning changed: per cell, not per text overall) |
| ~~`--game-disjoint-games`~~ | removed | argparse rejects it |

Fails when a display-name keyword expansion lands on a line whose `script_api_type` is not
`Keyword` (FR-015). The summary log adds family shortfalls and the real/what-if counts.

## `python -m effects train-effect-model` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--holdout-unit` | the manifest's unit | refused when it differs from the manifest's; the holdout is recomputed and must match the manifest's |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` | repeatable; sidecar roots, recorded in the checkpoint |
| `--e-noise` | 0.1 | noise ratio `r` relative to the running covariance of `e` (meaning changed: no longer a fixed σ) |
| `--value-weight` | 0.05 | value-head loss weight |
| `--encoder-layers` | 4 | encoder layers |
| `--encoder-d-model` | 256 | encoder width; refused when not divisible by the head count |
| `--mlm-weight`, `--mlm-mask-prob`, `--api-weight` | 0.1, 0.15, 0.05 | now wired |
| `--keyword-expand-p` | 0.25 (unchanged) | training only |

Epoch line adds: share of trained records per rarity bucket and per rule family; the value, verdict,
created-objects, MLM and API loss terms.

## `python -m effects encode-abilities` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` | repeatable; sidecar roots |

Builds the encoder from the checkpoint's recorded size; keyword expansion probability is the shared
constant (0). Reports each truncated line by provenance key.

## `python -m effects evaluate-effect-model` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--win-rates` | `output/sealed/cards-win-rates.txt` | read by the decodability battery |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` | repeatable; sidecar roots |

Report adds: per-text card-disjoint results by rarity bucket and family (also averaged over
families); per-family memorization gap; random-seat slice; real-decision legality slice; withheld
keyword zero-shot, beside trained keywords; "gate 1 skipped" when no identity variant is given.

## `python -m effects scorer-smoke-test` (new)

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | `models/effects/effect-model/latest.pt` | effect model whose pooled `e` is written |
| `--sealed-encoder-checkpoint` | `models/sealed/encoder/latest.pt` | sealed vectors to concatenate |
| `--scratch-dir` | required | where the vectors and the scorer checkpoint go |

Copies the converted `.txt` files under `--scratch-dir`, runs `python -m sealed encode-cards` there with
`--sealed-encoder-checkpoint` and the `vocab.txt` beside it, inserts each card's pooled `e` before the
trailing deterministic-feature block of its vector (zeros for a card with no ability line), and runs
`python -m sealed train-scorer` Phase A (`--embedding-lr 0`) on that directory as a subprocess, with
its checkpoint directory under `--scratch-dir`. Writes nothing under `output/cardsfolder/`. Imports
nothing from `sealed.application`.

## `scripts/effect_embedding_probes/*.py` (existing)

Every script takes `--checkpoint` (default `models/effects/effect-model/latest.pt`),
`--abilities-root` (default `output/effects/abilities`) and a repeatable `--cards-folder` (default: the
trees the checkpoint recorded, else `output/cardsfolder/` and `output/tokenscripts/`), and reads
vocabulary, cache and width of `e` from them. Output goes under
`output/effects/reports/embedding-probes-<label>-<date>/`, the label being the checkpoint stem, or
`<run dir>-latest` for a `latest.pt`.
`taxonomy` columns are omitted when that cache is absent. `pca_directions.py` adds the participation
ratio.

## `scripts/effect_knowledge_probes/run.py` (new)

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | required | any effects checkpoint, gen-1 included |
| `--corpus` | the checkpoint's recorded corpus | curated corpus the probe set belongs to |
| `--records-dir` | none | override: read the probe games from these raw shards instead of the corpus's `validation/{card-disjoint,game-disjoint}/` shards, which the probe set lists |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` | sidecar trees (gen-1: the kept-aside copy) |
| `--abilities-root` | `output/effects/abilities` | cache |
| `--freeze-probe-set` | off | enumerate and write the probe set for `--corpus`, then exit |
| `--families` | all | comma list of family names |
| `--per-layer` | off | probe every trunk layer through forward hooks |
| `--method-c` | off | shallow trunk on frozen `e` |
| `--method-c-layers`, `--method-c-steps` | 1, fixed in code | the shallow trunk's depth (0 or 1) and training steps |
| `--games-per-stratum` | 300 | probe games enumerated per validation stratum at freeze time |
| `--items-per-target` | 3,000 | line-level probe items per target at freeze time |
| `--ablation-records` | fixed in code | records the ablation scores |
| `--no-sweeps`, `--no-ablation` | off | skip those sections |
| `--profile-shards` | fixed in code | corpus shards read to profile the interaction join |
| `--reports-dir` | `output/effects/reports/` | parent of the probe-set file and the output directory |
| `--vocab-path`, `--keyword-definitions` | the checkpoint's recorded paths | overrides |

Refuses to probe when no frozen set exists for `--corpus`, or when the set's corpus digest does not
match. The probe set stores line items grouped per key (`{key, labels: {target: value}, weights}`);
each ablation entry carries the field's `base_loss` beside the increases. Output: `output/effects/reports/knowledge-probes-<checkpoint stem>-<date>/` with tables and
`scorecard.json`.

## `scripts/effect_knowledge_probes/compare.py` (new)

`compare.py SCORECARD [SCORECARD …] [--output PATH]`: side-by-side table per family and rung, and a
ranking on rung 1 − rung 1w per probe type. Warns when the scorecards record different probe-set
digests.

# `draft` — draft-agent ML pipeline

Entry point: `python -m draft <subcommand>` (see `src/draft/__main__.py` and `infrastructure/cli.py`). A draft agent in three generations: gen-1 imitation of Forge's draft AI, gen-2 RL self-play fine-tuning, gen-3 online self-play GRPO. The model is a two-headed Set Transformer, an imitation **policy** over the cards in the current pack plus a Monte-Carlo **critic** on a context token. `draft` **imports from** `sealed` and `price_predictor`, never the reverse, and adds only the genuinely new logic (booster→state geometry, typed-token state, the two-headed model, the Java draft worker).

Every subcommand that writes files keeps a `run.json` run manifest in its run directory (`.claude/skills/artifact-layout/SKILL.md`), declared in `RUN_MANIFESTS` in `infrastructure/cli.py`; every trainer records in its `--checkpoint-dir` (on `--resume`, by default the directory the resumed run recorded). Paths follow the `artifact-layout` skill and every one is a required flag, with no default: a fresh offline trainer without `--drafts-path`, `--cards-path` or `--checkpoint-dir` exits 2 (a resumed run inherits what its checkpoint recorded), and `--picker-checkpoint` is required exactly when `--build-method picker` needs it. Every reader of the `.npz` card embeddings takes `--embeddings-path` (`src/sealed/docs/embedding-cache.md`).

## Topic files — read on demand

This file holds what applies to any work in the package. The rest is in `src/draft/docs/`, and none of it loads by itself. Read a topic file in full when its condition is met.

| Read | Before |
|---|---|
| `docs/generating-drafts.md` | running or changing `generate-draft-data`, live-seat piloting, `validate-builder` or `analyze-generated-decks` |
| `docs/training-runs.md` | running or changing `train-draft-agent`, `train-draft-agent-rl` or `train-draft-agent-online`; reading an anchor margin; pairing a corpus with a checkpoint |
| `docs/draft-games.md` | running or changing `play-draft-games`; comparing agents by games won; anything about `forge-native` |

Where the rest lives:

- Flags and defaults: each subcommand's `--help`.
- What each generation specifies: `specs/2026-05-28-draft-agent.md`, `specs/2026-06-04-draft-agent-live-play.md`, `specs/2026-06-10-draft-agent-gen2-rl.md`, `specs/2026-08-04-draft-agent-gen3-online-grpo.md`, `specs/2026-08-09-draft-game-evaluation.md`.
- Why: `experiments/2026-05-30-draft-agent-design.md`, `experiments/2026-06-10-draft-agent-gen2-design.md`, `experiments/2026-06-15-draft-agent-gen3-online-grpo-design.md`.

## Subcommands

- `generate-draft-data` — Python supervisor plus Java `DraftWorkerMain`; drafts whole pods, builds and scores every seat's deck, appends to `drafts.jsonl`. A trained agent can pilot live seats.
- `train-draft-agent` — gen-1: imitation policy and critic, trained jointly from a drafts corpus.
- `train-draft-agent-rl` — gen-2: on-policy actor-critic fine-tuning over a corpus the reference checkpoint generated.
- `train-draft-agent-online` — gen-3: one in-process generate → update → discard loop.
- `analyze-generated-decks` — deck composition and `deck_score` summary for one agent's seats of a drafts corpus.
- `play-draft-games` — plays matches between seats of the same pod; the measure that is not a function of the frozen scorer.
- `validate-builder` — picker-versus-SA builder diagnostic, run once per picker/scorer pair to choose `--build-method`.

## Agent labels

- A mix label names how a seat drafts. `forge-full` is pure Forge AI; `forge-r30` and `forge-r100` replace 30% and 100% of that seat's picks with uniform-random legal picks. `--agent-mix` is a categorical sampled independently per seat. Any other label must be listed in `-Ddraft.external.agents`; the worker refuses a mix with an unknown label at start, naming it (`DraftAgent.requireKnown`).
- A model-piloted seat's `agent` is just the mix label its checkpoint was bound to.
- A mix label never names deck building. Every seat's deck in a drafts corpus is built by this project's picker or SA builder, because `generate-draft-data` never calls a Forge deck builder. Every seat is built and scored on the same scale.
- `forge-native` exists only in `play-draft-games`: full-strength Forge drafting with Forge's own `BoosterDeckBuilder` building the deck too, so it alone is Forge end to end.

## Corpus file formats

Append-only data contracts spanning the Java writers and the Python readers — no single source file teaches a whole one, and a mismatch silently corrupts a corpus that cannot easily be rebuilt.

**Drafts** (`drafts.jsonl`, `generate-draft-data --output-path`): one self-contained JSON record per line, append-only; readers tolerate a trailing partial final line. Each record has `draft_id`, `run_id`, `timestamp` (ISO 8601 UTC), `seats` (length `pod_size`; each `{agent, deck (40 names incl. basics, or [] on failed build), deck_score (scorer scalar, or null on failed build)}`), and `boosters` (length `pod_size × packs`; each `{set_code, picks}` where `picks` is the cards drafted from that physical pack in pick order, fully drained to `pack_size`). The booster *ordering* pins all geometry: `pack_number(k)=⌊k/pod_size⌋+1`, `opening_seat(k)=k mod pod_size`, and the pick at offset `j` of `boosters[k]` was made by seat `(opening_seat + j·dir_p) mod pod_size` with `dir_p=+1` for packs 1 & 3 and `−1` for pack 2 — so any seat's full observation history reconstructs from the record alone (`draft/domain/draft_geometry.py`).

- The schema is the same across the three generations. No field says which checkpoint generated a corpus.

**Draft games** (`draft-games.txt`, `play-draft-games --output-path`): the same ten-field match-outcome format as the sealed `match-outcomes.txt` (`src/sealed/CLAUDE.md` § Corpus file formats), one row per match, written by `play-draft-games`.

- `method_A`/`method_B` carry the two seats' labels; `run_id` is the `play-draft-games` invocation.
- Both decks of a row were drafted in the same pod, but the format has no field for pod identity.
- It is kept in a separate file from the sealed corpus on purpose: these are draft-pool matches, and `train-scorer` and `train-encoder` must not read them.
- Tally with `python scripts/analyze_winrates.py <run dir>/draft-games.txt --cards-path <converted cardsfolder>`.

## Model artifacts

Checkpoints live in the trainer's `--checkpoint-dir`, a run directory under `models/draft/<gen>/agent/`. The best checkpoint of an offline run is the one with the lowest validation loss; of a gen-3 online run, the one with the highest anchor margin. All three trainers write the gen-1 payload format; the RL trainers add `rl_metadata`.

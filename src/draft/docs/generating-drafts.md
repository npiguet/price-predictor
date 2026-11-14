# Generating drafts and piloting live seats

Rules to respect when running or changing `generate-draft-data`, live-seat piloting, `validate-builder` or `analyze-generated-decks`. Flags and defaults are in `--help`; the live-play protocol is specified in `specs/2026-06-04-draft-agent-live-play.md`.

## `generate-draft-data`

- The Java worker streams one `<<DRAFT-EVENT-JSON>>` transcript per completed draft. The supervisor reconstructs each seat's pool from the booster geometry, builds a deck per seat (`--build-method picker`, the default, or `greedy`), scores the non-basic subset with the frozen scorer and appends one record.
- There is one `run_id` per invocation. Crashed workers are restarted toward `--n-drafts`, and `--resume` counts pre-existing records.
- With no `--agent-checkpoint` the command must stay byte-for-byte the gen-1 behaviour: no pick side-channel, worker stderr discarded.

## Live seats

- `--agent-checkpoint LABEL=PATH` binds a mix label to a checkpoint. At a pick for such a seat the worker emits `<<DRAFT-PICK-REQUEST>>` and blocks on stdin for the supervisor's `<<DRAFT-PICK-RESPONSE>>`, which is the policy's choice or `abort`.
- Every mix label must be a Forge built-in or bound to a checkpoint, and each checkpoint's `packs`/`P` are validated against the live geometry. Both fail fast.
- The supervisor rebuilds the live state incrementally with `OnlineDraftStateTracker`. It must equal `build_state`; a gating equivalence test pins it.
- A pick fault (a policy or tracker error, a protocol desync, or every legal action un-embeddable) abandons the whole in-flight draft. No substitute pick is ever made, which keeps the corpus clean by construction. The draft is not recorded or counted.
- `--max-consecutive-faults` consecutive abandonments abort the run with a nonzero exit.
- Worker stderr goes to a per-run log, `output/draft/worker-<run_id>.log`.
- A seeded `--pick-mode sample` run is reproducible.
- A mixed pod yields a per-draft agent-minus-Forge `deck_score` delta, since every seat is built and scored the same way.

## `analyze-generated-decks`

- `--agent` is required and scopes the whole report to one label's seats; run it once per agent to compare.
- Seats with a failed build are skipped. An `--agent` matching no built deck fails fast and lists the available agents.
- The composition engine is `sealed`'s, shared.

## `validate-builder`

- `--pools-from` and `--fresh-pools` are a required, mutually exclusive pair.
- Run it once per picker/scorer pair to choose `--build-method`.

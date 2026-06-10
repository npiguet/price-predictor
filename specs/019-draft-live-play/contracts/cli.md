# CLI contract: `generate-draft-data` (extended for live model play)

`generate-draft-data` is **extended**, not replaced (design §5). With no model
agents it is byte-for-byte the gen-1 command (SC-004); model agents are opt-in.
Gen-1 flags (`--n-drafts`, `--set`, `--agent-mix`, `--scorer-checkpoint`,
`--build-method`, `--picker-checkpoint`, `--cards-path`, `--output-path`,
`--resume`) keep their meanings — see
[`specs/018-draft-agent/contracts/cli.md`](../../018-draft-agent/contracts/cli.md).

## Added flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--agent-checkpoint LABEL=PATH` | _(none; repeatable)_ | Bind a mix label to a trained agent checkpoint. Repeat to pit checkpoints (e.g. `a=…/best_A.pt b=…/best_B.pt`). A bare `PATH` is shorthand for label `draft-agent`. A mix label that is neither a Forge built-in nor bound here ⇒ **fail fast** (FR-011). |
| `--pick-mode {argmax,sample}` | `argmax` | `argmax` = the policy's strongest legal card (default; evaluation + high-quality self-play). `sample` = temperature-scaled softmax over PACK logits (rollout diversity). |
| `--temperature FLOAT` | `1.0` | Softmax temperature for `--pick-mode sample`; ignored for `argmax`. |
| `--seed INT` | _(none → nondeterministic)_ | Seeds the supervisor's pick-sampling RNG for reproducible sampled rollouts (SC-007). Forge-side randomness (boosters, Forge-AI seats) is the JVM's own RNG and is not seeded here. |
| `--max-consecutive-faults INT` | `5` | Abort the whole run with a nonzero exit after this many **consecutive** pick-fault-abandoned drafts (FR-016). Recovered worker crashes do not count; any completed draft resets the counter. |

## Validation & exit codes

- **Unknown label** (mix label neither Forge built-in nor bound) → stderr error,
  exit `2` (FR-011).
- **Geometry mismatch** (a bound checkpoint's `packs` ≠ live packs, or `P` <
  live pack size when `--set` fixes it) → stderr error, exit `2` (FR-012).
- **Malformed `LABEL=PATH`** / missing checkpoint file → stderr error, exit `2`.
- **`--temperature` ≤ 0** with `--pick-mode sample` → stderr error, exit `2`.
- **Consecutive-fault auto-abort reached** → prominent stderr error, **nonzero
  exit** (FR-016, SC-008).
- **SIGINT** → clean stop, exit `130`.

## Examples

One model seat (sampled ~⅛ of seats) against Forge:

```
python -m draft generate-draft-data --n-drafts 500 --set BLB \
  --agent-mix forge-full:7,draft-agent:1 \
  --agent-checkpoint draft-agent=models/draft/agent/latest.pt
```

Two checkpoints drafting against each other and Forge, sampled + seeded:

```
python -m draft generate-draft-data --n-drafts 500 \
  --agent-mix forge-full:4,a:2,b:2 \
  --agent-checkpoint a=models/draft/agent/best_A.pt \
  --agent-checkpoint b=models/draft/agent/best_B.pt \
  --pick-mode sample --temperature 1.2 --seed 7
```

## Console output (FR-013)

Logs as gen-1 (startup config, per-draft progress + ETA) with model-agent
additions visible at startup (which labels are model-backed, their checkpoint
paths, pick mode, seed) and which seats each draft model-piloted; a prominent
`ERROR` line whenever a pick fault abandons a draft. An abandoned draft is not
counted toward `--n-drafts`. Illustrative (exact wording non-normative):

```
generate-draft-data: target 500 drafts, set BLB, resume off
  agent mix: forge-full:7, draft-agent:1
  model agents: draft-agent -> models/draft/agent/latest.pt  (pick-mode=argmax)
  scorer: models/sealed/scorer/latest.pt   builder: picker
draft 1/500  set BLB  model seats [3]   done in 14s  ETA ~1h57m
ERROR pick fault on draft <uuid> seat 5 (pack 2 pick 3): <reason> — draft abandoned, not recorded
draft 2/500  set BLB  model seats [0,5] done in 13s  ETA ~1h49m
```

## Files

Only persistent output is the appended `drafts.jsonl` (unchanged schema, FR-009)
plus the worker's per-run stderr log. **No** model artifacts are written
(FR-015) — the policy loads existing `models/draft/agent/*.pt` and writes nothing
under `models/`.

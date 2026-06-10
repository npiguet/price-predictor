# Quickstart: Draft agent — live Forge integration

Pilot a trained draft agent as a live Forge seat and emit a self-play corpus.
This extends `generate-draft-data` (018); see that feature's quickstart for the
Forge build + scorer/picker prerequisites.

## Prerequisites

- A built forge-connector fat JAR (`cd forge-connector && mvn package -DskipTests`)
  with the live-play pick protocol — rebuild after this feature lands.
- A sibling `../forge` checkout built with `mvn install -DskipTests`.
- A trained draft-agent checkpoint at `models/draft/agent/latest.pt` (from
  `python -m draft train-draft-agent`).
- A frozen scorer + picker for deck labeling (`models/sealed/scorer/latest.pt`,
  `models/sealed/picker/latest.pt`) and the `.npz` cache under
  `output/cardsfolder/` (from `sealed encode-cards`).

## Workflow 1 — produce a self-play corpus (US1, P1)

Run a pod where ~⅛ of seats are agent-piloted and the rest Forge:

```
python -m draft generate-draft-data --n-drafts 500 --set BLB \
  --agent-mix forge-full:7,draft-agent:1 \
  --agent-checkpoint draft-agent=models/draft/agent/latest.pt
```

Each completed draft is appended to `output/draft/drafts.jsonl` with the
agent-piloted seats carrying the `draft-agent` label and every seat built +
scored. Concatenate this with prior `drafts.jsonl` and retrain the next
generation; the model's own picks become imitation targets only if you pass
`train-draft-agent --imitation-agents draft-agent` (the critic trains on every
seat regardless).

`--resume` counts drafts already in the file toward `--n-drafts`.

## Workflow 2 — measure agent strength vs Forge in one pod (US2, P2)

Use a mix with both agent and Forge labels; all seats draft the same boosters:

```
python -m draft generate-draft-data --n-drafts 1000 --set BLB \
  --agent-mix forge-full:6,draft-agent:2 \
  --agent-checkpoint draft-agent=models/draft/agent/latest.pt
```

For each draft, compare the `draft-agent` seats' `deck_score` against the
`forge-full` seats' in the **same** record — shared boosters cancel set/pool
luck, so a per-draft agent-minus-Forge score delta is directly comparable with no
cross-pod normalization.

## Workflow 3 — rival checkpoints + sampling (US3, P3)

Pit two checkpoints against each other and Forge, with seeded temperature
sampling for reproducible rollout diversity:

```
python -m draft generate-draft-data --n-drafts 500 \
  --agent-mix forge-full:4,a:2,b:2 \
  --agent-checkpoint a=models/draft/agent/best_A.pt \
  --agent-checkpoint b=models/draft/agent/best_B.pt \
  --pick-mode sample --temperature 1.2 --seed 7
```

Each checkpoint's seats are recorded under its own label. Re-running with the
same `--seed` and inputs reproduces identical agent-seat picks (SC-007).
`--pick-mode argmax` (default) instead always takes each agent's strongest legal
card.

## What happens on a fault

If a model seat ever can't make a genuine pick (policy error, protocol desync, or
every legal card un-embeddable), the **whole draft is abandoned**: it is never
written, an `ERROR` line is logged, and the run continues toward `--n-drafts`
without counting it (no substitute pick is ever recorded). A persistent
deterministic fault aborts the run with a nonzero exit after
`--max-consecutive-faults` consecutive abandonments (default 5) — recovered
worker crashes don't count.

## Verify

- `output/draft/drafts.jsonl` gains N records, each with ≥ 1 agent-piloted seat
  (`seats[i].agent` = your model label) and every seat carrying `deck` +
  `deck_score`.
- Running with **no** `--agent-checkpoint` reproduces gen-1 output exactly
  (SC-004).
- The worker's per-run stderr log under `output/draft/` captures diagnostics.

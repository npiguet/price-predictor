# Quickstart: Draft agent (generation 1)

Generation 1 ships **data generation + training + offline evaluation** only —
no live Forge seat, no RL (those are gen 2).

## Prerequisites

- Project installed (`pip install -e ".[dev]" --extra-index-url …`), sibling
  `../forge` built (`mvn install -DskipTests`), MTGJSON dumps in `resources/`.
- `forge-connector` fat JAR built (now also contains `DraftWorkerMain`):
  ```bash
  cd forge-connector && mvn package -DskipTests
  ```
- A populated `.npz` card cache from a prior `encode-cards` run
  (`output/cardsfolder/*.npz`).
- A frozen sealed **scorer** (`models/sealed/scorer/latest.pt`) and **picker**
  (`models/sealed/picker/latest.pt`) whose `.npz` width matches the cache.

## 1. (Optional) Validate the picker as a label-builder  (User Story 3)

Before a large corpus run, decide whether the fast picker can build the
labeling decks or whether you need the slower SA builder. Run the diagnostic
subcommand:

```bash
python -m draft validate-builder --pools-from output/draft/drafts.jsonl
# or --fresh-pools --set BLB --n-pools 300
```

It prints the picker-vs-SA Spearman correlation, the SA−picker score-gap
median/spread, and the SA-vs-SA reference ceiling. If the picker tracks SA
about as well as SA tracks itself, proceed with `--build-method picker`;
otherwise pass `--build-method greedy` in step 2.

## 2. Generate a labeled draft corpus  (User Story 1)

```bash
python -m draft generate-draft-data --n-drafts 1000
# restrict to one set:
python -m draft generate-draft-data --n-drafts 1000 --set BLB
# resume an interrupted run:
python -m draft generate-draft-data --n-drafts 1000 --resume
```

Produces `output/draft/drafts.jsonl` (one self-contained JSON record per line).
The supervisor restarts crashed worker JVMs automatically and continues toward
`--n-drafts`. Each record carries per-seat agents/decks/scores and the full
per-booster pick transcript (see `contracts/drafts-jsonl.md`).

## 3. Train the two-headed agent  (User Story 2)

```bash
python -m draft train-draft-agent
# critic-only / imitation-only ablations:
python -m draft train-draft-agent --imitation-weight 0
python -m draft train-draft-agent --critic-weight 0
# imitate only the strong agent's picks (default), or widen the whitelist:
python -m draft train-draft-agent --imitation-agents forge-full,forge-r30
# continue a stopped run (inherits its settings); override only what you pass,
# e.g. anneal the LR:
python -m draft train-draft-agent --resume models/draft/agent/latest.pt --lr 3e-5
```

Training validates and checkpoints `--evals-per-epoch` times per epoch
(default 100 — "mini-epochs"). Each mini-epoch logs the training and validation
loss split plus validation imitation top-1/top-3 accuracy and critic MSE sliced
by pack number; `latest.pt` and the best checkpoint (by validation `L`) land
under `models/draft/agent/`, so a crash costs at most one mini-epoch and
`--resume` continues. `--resume` inherits the run's training settings and any
CLI flag overrides them (resume precedence).

## 4. Inspect the artefact

The checkpoint reloads and produces policy picks (`argmax` over PACK) and a
critic scalar (de-standardized to raw scorer-score space) on any held-out
state. Architecture, derived `embedding_dim`, pack size `P`, and the
critic-target standardization mean/std are stored in the checkpoint config.

## Validating against the success criteria

| Check | Criterion |
|---|---|
| `drafts.jsonl` gained exactly N parseable records | SC-001 |
| Any seat's POOL/PACK/PASSED/TAKEN reconstructs from a record alone | SC-002 |
| Worker crash didn't abort the run | SC-003 |
| Training emits a reloadable best checkpoint producing picks + critic | SC-004 |
| Mini-epoch val log has train/val loss + top-1/top-3 + per-pack critic MSE | SC-005 |
| `d_model % n_heads != 0` (or arch flags with `--resume`) fails fast | SC-006 |
| Diagnostic prints a gating Spearman + SA-vs-SA ceiling | SC-007 |

## Tests

```bash
pytest tests/unit/draft/                 # fast unit tests (geometry, state, model, loss)
pytest tests/integration -k draft        # worker/pipeline smoke (Forge-dependent)
ruff check src/draft tests
cd forge-connector && mvn test           # DraftWorkerMain (integration-tagged)
```

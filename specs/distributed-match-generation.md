# Distributed match-outcomes generation

## Problem

Generating `match-outcomes.txt` is slow: each match is a full Forge AI
best-of-7, which is CPU-bound and cannot be parallelised further inside a
single JVM. On one desktop this caps the scorer's training-data throughput
well below what the rest of the pipeline can consume. Two more machines
(a Windows laptop and a TrueNAS SCALE 2025.10 server on a Ryzen 7 9700X
with 32 GB RAM) are idle most of the day.

## Goal

Fan the `sealed match-outcomes` supervisor out across all three machines
with zero new infrastructure: no coordinator, no queue, no shared state
beyond files. Each machine runs an independent supervisor; outputs are
append-only shards that `train-scorer` concatenates at read time.

## Topology

- The **NAS** is the file server. `/mnt/tank/price-predictor/` hosts
  `output/`, `resources/`, `models/` as ZFS datasets and re-exports them
  over SMB for the Windows clients.
- The **NAS** also runs a Docker container with the match-outcomes
  supervisor, bind-mounted directly to the ZFS datasets (no SMB loopback
  for local workers).
- The **Windows desktop** and **laptop** mount the SMB share, run the
  supervisor natively inside their local venv, and let it read/write the
  same three datasets.

```
+----------------+       SMB        +--------------------------+
|  Windows       | <--------------> |  TrueNAS SCALE 2025.10   |
|  desktop       |                  |                          |
|  (6 workers)   |                  |  ZFS: output/, resources/|
+----------------+                  |       models/            |
                                    |                          |
+----------------+       SMB        |  Docker: match-worker    |
|  Windows       | <--------------> |  container (8 workers,   |
|  laptop        |                  |  bind-mounts ZFS direct) |
|  (2-4 workers) |                  |                          |
+----------------+                  +--------------------------+
```

## Output-file sharding

Concurrent appends to a single `match-outcomes.txt` across SMB are **not**
atomic — Windows SMB does not guarantee the POSIX `O_APPEND` semantics
that local filesystems provide, so simultaneous writes from two clients
can interleave or clobber lines. To sidestep this entirely:

- Each supervisor writes to `output/sealed/match-outcomes-<hostname>.txt`.
  Default hostname comes from `socket.gethostname()`; override with a
  `--output-file` flag.
- The `run_id` already present in the file format disambiguates supervisor
  invocations inside a shard, so restarts are still lossless.
- `train-scorer` globs `match-outcomes*.txt` and concatenates shards in
  chronological order (by line timestamp, not by filename).

Required code changes (tracked separately):

1. `match_outcomes.py` — derive the default output path from the hostname,
   accept `--output-file` override.
2. `match_data_loader.py` — glob `match-outcomes*.txt` instead of reading
   one hard-coded path, sort merged lines by the ISO-8601 timestamp column.

## Docker image

Multi-stage build, one file: `docker/Dockerfile`.

**Stage 1 (builder, `debian:bookworm-slim`)**:

- Install Oracle JDK 25 from `download.oracle.com/java/25/latest/` (the
  no-auth NFTC URL) + `maven` + `git`.
- Clone Card-Forge/forge at `--build-arg FORGE_REF=<branch-or-sha>`
  (default is the pinned SHA
  `189a8b661d0a180eaf79773a81f421357ac0acdd`, matching the `..\forge`
  checkout on the dev machine) and `mvn -q install -DskipTests`.
- Copy in `forge-connector/` from the build context and `mvn -q package
  -DskipTests` to produce the fat JAR.

**Stage 2 (runtime, `python:3.14-slim`)**:

- Install the same Oracle JDK 25 tarball.
- Copy `/build/forge` and `/build/forge-connector` from stage 1 (the
  Python supervisor reaches into these exactly as it does on the host).
- `pip install --no-cache-dir --extra-index-url
  https://download.pytorch.org/whl/cpu -e .` — CPU-only torch wheels
  because match generation never touches the scorer at runtime.
- `ENTRYPOINT ["python", "-m", "sealed", "match-outcomes"]`, default
  `CMD ["--workers", "8"]`.

Rationale for a few choices:

- **Oracle JDK, not Temurin** — user preference.
- **Java 25 LTS** — matches the host JVM; NFTC license allows commercial
  use at no cost.
- **CPU-only torch** — saves ~1.5 GB of image weight; self-play mode
  reads `generated-decks.txt` produced offline, no inference at match
  time.
- **Clone Forge in the builder** instead of mounting a local checkout —
  keeps the image self-contained and reproducible. If iteration speed on
  Forge changes is ever needed, swap the `git clone` for a bind mount in
  a dev-compose override.

## Build procedure

From a developer machine (faster than building on the NAS):

```bash
# From the repo root. Build context is the repo root so the Dockerfile
# can COPY forge-connector and src/.
docker build \
  -f docker/Dockerfile \
  -t price-predictor-match:latest \
  .

# Override the pinned Forge commit (default is the SHA currently in ..\forge):
docker build \
  -f docker/Dockerfile \
  --build-arg FORGE_REF=<branch-or-sha> \
  -t price-predictor-match:latest \
  .
```

First build is ~8–15 min (Maven downloads + Forge compile + torch wheel).
Subsequent builds are cached unless `src/` or `forge-connector/` changes.

## Deploy to TrueNAS SCALE 2025.10

TrueNAS SCALE 24.10+ switched from K3s to Docker Compose under the
"Custom App" system, so the same `docker-compose.yml` works either via
the UI or via SSH.

**Option A — transfer the pre-built image:**

```bash
# On the dev machine:
docker save price-predictor-match:latest \
  | ssh truenas 'docker load'

# On the NAS:
cd /mnt/tank/price-predictor/docker
docker compose up -d
docker compose logs -f
```

**Option B — build on the NAS:**

```bash
# On the NAS (after cloning the repo to /mnt/tank/price-predictor):
cd /mnt/tank/price-predictor
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
```

The compose file binds three volumes (see `docker/docker-compose.yml`).
Adjust the left-hand host paths if the ZFS dataset is not under
`/mnt/tank/price-predictor`.

**Worker concurrency:** 8 on the NAS (1 per physical core; leaves RAM for
ZFS ARC, which TrueNAS sizes to ~50% of system memory by default). Watch
`htop` + `arc_summary.py` during the first run and adjust `--workers`
and `mem_limit` if ARC pressure shows up.

## Windows-client bootstrap

One-time per machine. Both the desktop and laptop share the same steps.

1. Install Python 3.14, JDK 17+ (any distribution), Maven, Git.
2. Clone this repo and `Card-Forge/forge` as siblings. Build Forge:
   `cd forge && mvn install -DskipTests`.
3. Download MTGJSON dumps (`AllPrintings.json`, `AllPricesToday.json`)
   into `resources/`.
4. `python -m venv .venv && .venv\Scripts\activate && pip install -e
   ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126`.
5. Mount the NAS SMB share at a drive letter (e.g. `Z:\`). Either:
   - Replace the local `output/`, `resources/`, `models/` with junctions
     pointing into `Z:\price-predictor\`, or
   - Run the repo directly from `Z:\` (simpler, slightly slower
     cold-start on Forge class loading).
6. Run the supervisor:
   `python -m sealed match-outcomes --workers 6` (desktop) or
   `--workers 2` (laptop, adjust for thermals).

The supervisor's hostname-derived output filename naturally segregates
each machine's writes.

## Merging outcomes for training

After the output-sharding change lands, `train-scorer` transparently
reads every `match-outcomes-*.txt` under `output/sealed/`. Before that
change ships, a manual merge works as a stopgap:

```bash
cat output/sealed/match-outcomes-*.txt \
  | sort \
  > output/sealed/match-outcomes.txt
```

(Sorting by the full line works because the ISO-8601 timestamp is the
first field.)

## Worker-count starting points

| Machine        | CPU                | RAM    | Workers |
|----------------|--------------------|--------|---------|
| NAS            | Ryzen 7 9700X (8c) | 32 GB  | 8       |
| Desktop        | (existing host)    | -      | 6       |
| Laptop         | Windows laptop     | -      | 2–4     |

Tune after the first full day: if the NAS hits ARC pressure or the
laptop thermal-throttles, drop by one or two.

## Non-goals

- A coordinator service, work queue, or distributed scheduler. Every
  supervisor is independent; the only shared resource is the filesystem.
- Cross-machine model inference. Self-play's pre-built
  `generated-decks.txt` is written once per generation and distributed
  via the SMB share.
- GPU acceleration. Match simulation is 100% CPU-bound.
- Live-merging shards into a single `match-outcomes.txt`. The glob-read
  change in `train-scorer` makes that unnecessary.

## Open questions

- **Forge pin cadence** — rebuild the image on every Forge upstream
  change, or pin to a known-good commit and rebuild quarterly? Pinning
  avoids AI-regression surprises mid-generation.
- **Network isolation** — `network_mode: none` would make the container
  strictly offline. Forge's first-run bootstrap may fetch assets; verify
  before locking it down.
- **Explicit `--output-file`** — worth adding even with the hostname
  default, for the case where two supervisors run on the same machine
  (e.g. phase-0 + self-play side by side).

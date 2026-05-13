# April 23, 2026 — Distributed match generation with Docker

**TL;DR:** Match generation was too slow, so I designed a
distributed setup across three machines. The NAS gets a Docker
container; the two Windows machines read/write over SMB.

The starting observation was simple: Forge AI matches are slow and
I can't speed up the game-playing itself, but I have two other
machines at home — a Windows laptop and a Linux-based NAS (TrueNAS
SCALE) — that could run workers in parallel. I asked Claude what the
simplest distributed architecture would look like, with a strong
preference for zero extra infrastructure.

Claude's first answer was to just run `match-outcomes` independently
on all three machines and concatenate the output files before
training. The `run_id` field already distinguishes supervisor
invocations, so a simple `cat` merge works. My thought was to put
the `output/` folder on an SMB share so all machines could see the
same `generated-decks.txt` for self-play. Claude flagged the one
real gotcha: concurrent appends to a single file over SMB aren't
atomic, so concurrent workers could interleave or lose lines. The
fix is per-machine output filenames that the training step globs
together.

For the NAS specifically, I couldn't install Python or Java directly
— TrueNAS doesn't recommend it. Docker was the natural answer. The
NAS turned out to be an x86 machine (AMD Ryzen 9700X, 32 GB RAM,
TrueNAS SCALE 2025.10 which already ships Docker), so no ARM
cross-build pain. Claude suggested capping at 8 Forge JVM workers
(one per physical core) because Forge matches are CPU-bound enough
that SMT siblings don't help much, and ZFS ARC wants roughly half
the RAM.

I decided I wanted Oracle JDK 25 specifically (not Eclipse
Temurin), and Python 3.14 to match my local setup. Claude drafted a
multi-stage Dockerfile: a builder stage that fetches Oracle JDK 25,
clones and builds Forge, and builds `forge-connector`; then a slim
runtime stage on `python:3.14-slim` that copies in the built JARs
and installs the Python package with CPU-only torch wheels. The
Compose file mounts the ZFS dataset directly so local workers get
full ZFS speed, with only the Windows machines going through SMB.

I also asked Claude to pin the Forge clone to the exact commit
currently checked out in `../forge` rather than tracking `master`
blindly. The Dockerfile had been using `git clone --depth 1 --branch
<sha>`, which doesn't accept bare SHAs — Claude switched it to `git
init` + `git fetch --depth 1` + `git checkout FETCH_HEAD` to handle
both branch names and commit SHAs.

The outputs were `docker/Dockerfile`, `docker/docker-compose.yml`,
and a new spec at `specs/2026-04-23-distributed-match-generation.md` (sibling
of `2026-03-28-sealed-deck-picker.md`). The spec documents the full topology,
the SMB concurrent-append problem and its hostname-sharding fix,
build and deploy procedures for TrueNAS, and the two small code
changes still needed: a hostname-derived default output filename and
a glob-read in the data loader.

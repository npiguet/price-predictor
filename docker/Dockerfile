# syntax=docker/dockerfile:1.7

###############################################################################
# How to build this image
#
# Nothing is built inside the image; the host must produce everything first.
#
# Host prerequisites (run from the project root):
#   1. Build forge:           (cd ../forge && mvn install -DskipTests)
#   2. Build forge-connector: (cd forge-connector && mvn package -DskipTests)
#
# The forge checkout lives at ../forge (sibling of this project), outside the
# default build context. It is brought in as a BuildKit named build context
# (--build-context forge=../forge) and pulled into the image with
# `COPY --from=forge`, so no symlinking, junction, or parent-dir context
# rewrite is needed. Requires the dockerfile-1.4+ frontend, which is enabled
# by the `# syntax=` directive at the top of this file.
#
# Build from the project root.
#
#   Linux / macOS / git-bash (line continuation with `\`):
#       docker build \
#           --build-context forge=../forge \
#           -f docker/Dockerfile \
#           -t price-predictor .
#
#   Windows PowerShell (line continuation with backtick `):
#       docker build `
#           --build-context forge=../forge `
#           -f docker/Dockerfile `
#           -t price-predictor .
#
#   Windows cmd.exe (line continuation with `^`):
#       docker build ^
#           --build-context forge=../forge ^
#           -f docker/Dockerfile ^
#           -t price-predictor .
###############################################################################

###############################################################################
# How to ship this image (no registry)
#
# Step 1 — save the built image to a tar archive on the build host. Use
# docker's native -o flag so the same command works on every shell:
#       docker save price-predictor -o price-predictor.tar
#
# The tar is large (multi-GB) — compress it for transfer if bandwidth or
# disk matters.
#   Linux / macOS / git-bash:
#       gzip price-predictor.tar                       # → price-predictor.tar.gz
#   Windows PowerShell:
#       Compress-Archive -Path price-predictor.tar -DestinationPath price-predictor.zip
#
# Step 2 — transfer the archive to the target host (scp, rsync, USB, etc).
#
# Step 3 — on the target, decompress (if compressed) and load into the
# local Docker daemon:
#   Linux / macOS / git-bash:
#       gunzip price-predictor.tar.gz                  # → price-predictor.tar
#       docker load -i price-predictor.tar
#   Windows PowerShell:
#       Expand-Archive -Path price-predictor.zip -DestinationPath .
#       docker load -i price-predictor.tar
#
# After load, `docker images` on the target shows `price-predictor:latest`
# and it can be run as if it had been built locally.
###############################################################################

###############################################################################
# How to run this image
#
# resources/ (MTGJSON dumps) is baked into the image — don't mount over it.
# output/ and models/ are mounted from the host so generated artifacts and
# trained checkpoints survive container restarts and are visible from the
# host. The container's WORKDIR is /app, so the in-image paths are
# /app/output and /app/models.
#
# Default invocation runs `python -m sealed match-outcomes --workers 8`
# (the ENTRYPOINT + CMD set at the bottom of this Dockerfile). Run from a
# host directory that contains output/ and models/ subdirs:
#
#   Linux / macOS / git-bash:
#       docker run --rm \
#           -v "$(pwd)/output:/app/output" \
#           -v "$(pwd)/models:/app/models" \
#           price-predictor
#
#   Windows PowerShell:
#       docker run --rm `
#           -v "${PWD}/output:/app/output" `
#           -v "${PWD}/models:/app/models" `
#           price-predictor
#
#   Windows cmd.exe:
#       docker run --rm ^
#           -v "%cd%\output:/app/output" ^
#           -v "%cd%\models:/app/models" ^
#           price-predictor
#
# Override the worker count (or any other match-outcomes flag) by passing
# args after the image name — they recommplace CMD, not ENTRYPOINT:
#       docker run --rm -v ... price-predictor --workers 4 --best-of 5
#
# Run a different sealed/price_predictor subcommand by overriding the
# entrypoint (same syntax on all shells, modulo the line-continuation
# character — `\`, ` `` `, or `^` as shown above):
#       docker run --rm -v ... --entrypoint python price-predictor \
#           -m sealed train-scorer --epochs 50
###############################################################################

###############################################################################
# Runtime: python 3.14 + Oracle JDK 25 + pre-built forge + pre-built JARs
# Nothing is built inside the image; everything is COPY'd in pre-built.
###############################################################################
FROM python:3.14-slim
ARG JDK_URL=https://download.oracle.com/java/25/latest/jdk-25_linux-x64_bin.tar.gz

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/java \
 && curl -fsSL "$JDK_URL" | tar -xz -C /opt/java --strip-components=1
ENV JAVA_HOME=/opt/java
ENV PATH=/opt/java/bin:$PATH

WORKDIR /app

# Layers below are ordered from least-volatile to most-volatile so that a
# common change (Python source edit) only invalidates the cheap layers.

# MTGJSON dumps (AllPrintings.json, AllPricesToday.json) — baked in rather than
# mounted. They're stable inputs that change only on MTGJSON refresh, so a
# rebuild is the right cadence for updating them. Placed first because they're
# the largest rarely-changing payload in the image.
COPY resources/ ./resources/

# Pre-built forge checkout (mvn install -DskipTests already run on host).
# Pulled from the `forge` named build context (--build-context forge=../forge),
# not from the default context — see the header comment.
# Mirrors the host layout: forge is a SIBLING of the project, not a child —
# the Python code resolves it as `project_root().parent / "forge"`, which
# from /app yields /forge. Same goes for the CLI default
# `../forge/forge-gui/res/cardsfolder/`.
COPY --from=forge . /forge

# Pre-built forge-connector fat JAR (lives inside the project, at
# project_root()/forge-connector/target/...).
COPY forge-connector /app/forge-connector

# Python package. Install CPU-only torch wheels: match-outcomes never does
# model inference at match time, so CUDA wheels would waste ~1.5 GB.
# `pip install -e .` runs setuptools package discovery (configured under
# `[tool.setuptools.packages.find] where = ["src"]`), so src/ must be COPY'd
# *before* the install, not after.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -e .

VOLUME ["/app/output", "/app/models"]

ENTRYPOINT ["python", "-m", "sealed", "match-outcomes"]
CMD ["--workers", "8"]

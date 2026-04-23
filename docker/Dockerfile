# syntax=docker/dockerfile:1.7

ARG JDK_URL=https://download.oracle.com/java/25/latest/jdk-25_linux-x64_bin.tar.gz
# Pinned to the commit currently checked out in ../forge on the dev machine.
# Bump deliberately when Forge updates are wanted; don't track master blindly.
ARG FORGE_REF=189a8b661d0a180eaf79773a81f421357ac0acdd

###############################################################################
# Stage 1: builder — Oracle JDK 25 + Maven, build forge and forge-connector
###############################################################################
FROM debian:bookworm-slim AS builder
ARG JDK_URL
ARG FORGE_REF

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git maven \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/java \
 && curl -fsSL "$JDK_URL" | tar -xz -C /opt/java --strip-components=1
ENV JAVA_HOME=/opt/java
ENV PATH=/opt/java/bin:$PATH

WORKDIR /build

# Build Forge (Card-Forge/forge) at the pinned commit. Shallow-fetch the exact
# SHA instead of a branch so FORGE_REF can be either a branch name or a commit.
RUN git init forge \
 && cd forge \
 && git remote add origin https://github.com/Card-Forge/forge.git \
 && git fetch --depth 1 origin "$FORGE_REF" \
 && git checkout FETCH_HEAD \
 && mvn -q install -DskipTests

# Build forge-connector fat JAR.
COPY forge-connector /build/forge-connector
RUN cd /build/forge-connector && mvn -q package -DskipTests

###############################################################################
# Stage 2: runtime — python 3.14 + Oracle JDK 25 + built JARs + python src
###############################################################################
FROM python:3.14-slim AS runtime
ARG JDK_URL

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/java \
 && curl -fsSL "$JDK_URL" | tar -xz -C /opt/java --strip-components=1
ENV JAVA_HOME=/opt/java
ENV PATH=/opt/java/bin:$PATH

WORKDIR /app

# Forge game resources + built JARs. MatchWorkerMain reaches into /app/forge at
# runtime exactly the way the Python supervisor does on the host.
COPY --from=builder /build/forge /app/forge
COPY --from=builder /build/forge-connector /app/forge-connector

# Python package. Install CPU-only torch wheels: match-outcomes never does
# model inference at match time, so CUDA wheels would waste ~1.5 GB.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -e .

VOLUME ["/app/output", "/app/resources", "/app/models"]

ENTRYPOINT ["python", "-m", "sealed", "match-outcomes"]
CMD ["--workers", "8"]

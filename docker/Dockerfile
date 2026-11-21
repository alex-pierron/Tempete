# syntax=docker/dockerfile:1.7
#
# wmkit -- CPU image (portable, reproducible).
#
# READ THIS FIRST (macOS):
#   Docker on macOS runs containers inside a Linux VM that has NO access to the
#   Apple GPU (MPS). This image is therefore CPU-only on a Mac and noticeably
#   SLOWER than the native `.venv` (which uses MPS). Use it for portability and
#   reproducibility (a colleague's machine, the competition's Linux server), not
#   for speed on a Mac. Docker is the fast path only on a Linux host with an
#   NVIDIA GPU: see docker/Dockerfile.cuda.
#
# What is inside:  Python 3.12, torch 2.15 (CPU wheels), the locked dependencies of
#   uv.lock, wmkit and the vendored Meta videoseal (installed --no-deps, as in
#   scripts/setup_env.sh), plus pytest so that `make docker-test` works.
# What is NOT inside: weights, datasets and results (large, gitignored). Mount them:
#   -v $PWD/weights:/app/weights -v $PWD/data:/app/data -v $PWD/results:/app/results
#   (docker/compose.yml does it for you).
#
# Build (from the repository root; third_party/videoseal must exist, see `make setup`):
#   docker build -f docker/Dockerfile -t wmkit:cpu .
# Run:
#   docker run --rm wmkit:cpu list
#
# Reproducibility: dependency versions come from uv.lock (exported, not re-resolved),
# so the lockfile is never regenerated here (local uv/npm versions cannot drift it).

ARG PYTHON_IMAGE=python:3.12.16-slim-bookworm

# ---------------------------------------------------------------------------
# Stage 1: builder. Holds uv and the build toolchain; only /opt/venv leaves it.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

# Pinned to the uv used to produce uv.lock.
ARG UV_VERSION=0.8.22
# CPU-only torch wheels: PyPI's default Linux x86_64 torch drags ~3 GB of CUDA libs.
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

# Layer 1 (rarely changes): dependencies, resolved from uv.lock only.
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    uv venv /opt/venv --python /usr/local/bin/python; \
    uv export --frozen --no-dev --no-emit-project --no-hashes --no-header \
        | grep -vE '^[[:space:]]|^#' > /tmp/requirements.all.txt; \
    # torch, torchvision and every CUDA/Triton wheel are replaced by the CPU build below.
    grep -vE '^(torch|torchvision|triton|cuda-[a-z-]+|nvidia-[a-z0-9-]+)==' \
        /tmp/requirements.all.txt > /tmp/requirements.cpu.txt; \
    uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/requirements.cpu.txt; \
    TORCH_VERSION="$(grep -E '^torch==' /tmp/requirements.all.txt | sed -E 's/^torch==([^ ;]+).*/\1/')"; \
    TORCHVISION_VERSION="$(grep -E '^torchvision==' /tmp/requirements.all.txt | sed -E 's/^torchvision==([^ ;]+).*/\1/')"; \
    uv pip install --python /opt/venv/bin/python --no-deps \
        --index-url "${TORCH_CPU_INDEX_URL}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
    # Test runner for `make docker-test` (kept out of the runtime lockfile export).
    PYTEST_VERSION="$(uv export --frozen --only-group dev --no-emit-project --no-hashes --no-header | grep -E '^pytest==' | sed -E 's/^pytest==([^ ;]+).*/\1/')"; \
    uv pip install --python /opt/venv/bin/python "pytest==${PYTEST_VERSION}"

# Layer 2 (changes with the code): wmkit and videoseal, installed editable.
# Editable on purpose: wmkit.seal reads videoseal's model cards and configs/ relative to the
# checkout root, and wmkit.config resolves ./data, ./weights, ./results from PROJECT_ROOT.
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY third_party/videoseal ./third_party/videoseal
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps \
        -e /app \
        -e /app/third_party/videoseal

# ---------------------------------------------------------------------------
# Stage 2: runtime. No compiler, no uv, no pip cache.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# libgl1 + libglib2.0-0: opencv-python (imported by videoseal and wmkit) needs them.
# make: `make docker-test` runs the project Makefile inside the container.
# ffmpeg is NOT installed: video I/O goes through PyAV, whose wheels bundle their codecs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 make \
    && rm -rf /var/lib/apt/lists/*

# Fixed uid/gid so bind-mounted ./results are writable on Linux hosts (override with
# `docker run --user "$(id -u):$(id -g)"`, or WMKIT_UID/WMKIT_GID in compose).
ARG WMKIT_UID=1000
ARG WMKIT_GID=1000
RUN groupadd --gid "${WMKIT_GID}" wmkit \
    && useradd --uid "${WMKIT_UID}" --gid "${WMKIT_GID}" --create-home --shell /usr/sbin/nologin wmkit

ENV PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    # Explicit locations: they are the mount points of the host volumes.
    WMKIT_DATA_DIR=/app/data \
    WMKIT_WEIGHTS_DIR=/app/weights \
    WMKIT_RESULTS_DIR=/app/results \
    # Third-party caches (lpips AlexNet, Hugging Face VAE) land in the weights volume,
    # so they are downloaded once and survive `--rm` containers.
    TORCH_HOME=/app/weights/torch \
    HF_HOME=/app/weights/hf \
    # No GPU is reachable from this image; say so instead of probing.
    WMKIT_DEVICE=cpu

COPY --from=builder /opt/venv /opt/venv
# /app belongs to the runtime user: pytest writes .pytest_cache there, and the three
# mount points exist and are writable even when no volume is mounted over them.
RUN mkdir -p /app/data /app/weights /app/results \
    && chown -R wmkit:wmkit /app
WORKDIR /app
COPY --from=builder --chown=wmkit:wmkit /app /app
COPY --chown=wmkit:wmkit tests ./tests
COPY --chown=wmkit:wmkit scripts ./scripts
COPY --chown=wmkit:wmkit Makefile ./Makefile

USER wmkit

# Cheap liveness probe: the whole stack imports. Meaningful for long `docker run -d` use.
HEALTHCHECK --interval=5m --timeout=60s --start-period=30s --retries=2 \
    CMD ["python", "-c", "import torch, videoseal, wmkit"]

ENTRYPOINT ["python", "-m", "wmkit"]
CMD ["list"]

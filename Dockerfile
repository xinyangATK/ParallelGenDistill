# =============================================================================
# Draft-OPD runtime image for the AMLT / Singularity OPD training job.
#
#   * CUDA 12.8 devel base — matches the torch 2.9.1 (cu128) wheels the
#     sglang-dflash fork pins (sglang-dflash/python/pyproject.toml). Keeping the
#     base CUDA aligned with torch's bundled cu128 runtime avoids the libcudart
#     symbol mismatch that bites when a 12.4 system CUDA shadows torch's 12.8 libs.
#     The devel base also keeps nvcc around for flashinfer / sgl-kernel runtime JIT.
#   * Ubuntu 24.04 -> Python 3.12 (the version the `draftopd` env uses).
#   * InfiniBand / RDMA userspace libs so NCCL can use the host IB fabric if you
#     ever scale past one node (the default job is single-node, 8 GPUs).
#
# Build is layered for cache friendliness (same idea as opd_verl06's Dockerfile):
# the heavy env (sglang-dflash fork -> torch 2.9.1 + the whole pinned stack, verl
# deps, flash_attn) lives in early layers keyed only on the dependency manifests,
# and verl's source is COPYed LAST + installed `--no-deps -e .`. So editing verl
# only re-runs the final cheap layer instead of reinstalling the whole stack.
# This mirrors what amlt does at job time (amlt_draftopd.yaml runs
# `pip install --no-deps -e ./sglang-dflash/python -e ./verl` from the uploaded
# checkout), so the running code always matches the pushed commit regardless of
# what verl source got baked here. NOTE: this intentionally does NOT use the
# single `bash install.sh` step — install.sh stays the README/local path.
#
# Build / push (from the REPO ROOT so the COPY paths resolve; --platform needed
# on Apple-Silicon Macs):
#   docker build --platform linux/amd64 -f Dockerfile \
#     -t 44359f3e6e864ddba0853a221be97722.azurecr.io/draft-opd/draft-opd:latest .
#   docker push 44359f3e6e864ddba0853a221be97722.azurecr.io/draft-opd/draft-opd:latest
# Then set `environment.image` in amlt/amlt_draftopd.yaml to that tag.
# =============================================================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# --- 1. System deps: toolchain + Python 3.12 + GL/glib (opencv/vision) + IB. ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ca-certificates build-essential ninja-build \
      python3.12 python3.12-dev python3.12-venv python3-pip \
      libgl1 libglib2.0-0 \
      rdma-core ibverbs-providers libibverbs1 librdmacm1 ibverbs-utils \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv on PATH so `python` / `pip` at runtime hit this interpreter.
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN ln -sf /opt/venv/bin/python /usr/local/bin/python \
    && pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# --- 2. HEAVY ENV LAYER: sglang-dflash fork (editable). This pulls the whole
#     pinned runtime stack — torch==2.9.1, sgl-kernel==0.3.21, flashinfer==0.6.4,
#     transformers==4.57.1, torch_memory_saver, ... — so torch arrives here as a
#     dependency rather than a separate pre-step. Rebuilds only when the fork
#     itself changes (which you rarely edit), NOT when you touch verl. ---
COPY sglang-dflash/ ./sglang-dflash/
RUN cd sglang-dflash && pip install -e ./python && pip install cachetools

# --- 3. verl DEPENDENCIES ONLY (no verl source yet). Uses verl's full dev
#     manifest requirements.txt, a SUPERSET of setup.py's install_requires:
#     beyond the core deps it also pulls liger-kernel, pre-commit, and the
#     OPTIONAL math reward libs (math_verify + latex2sympy2_extended). NOTE the
#     default validation scorers are built-in (reward_score/__init__.py routes
#     gsm8k/math500/aime to gsm8k / math_reward / math_dapo) and do NOT import
#     math_verify — it's only an opt-in enhancement, so these extras are inert at
#     runtime, just harmless pre-staged wheels. Installing the full manifest here
#     warms the dep cache; keyed only on requirements.txt so it stays cached
#     unless deps change, letting the verl-source layer below run --no-deps. ---
COPY verl/requirements.txt ./verl/requirements.txt
RUN pip install -r verl/requirements.txt

# --- 4. flash_attn: rmpad's `flash_attn.bert_padding` is hard-imported by the
#     trainer and is NOT a verl/sglang dep. Depends only on torch (already
#     present), so it sits in this stable pre-source layer. Prefer the prebuilt
#     cp312 / torch2.9 / cu12 wheel; fall back to a (slow) source build if the
#     wheel name ever drifts. ---
RUN pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    || MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation

# --- 5. verl SOURCE (changes often). COPYed last + registered `--no-deps -e .`
#     so editing verl only re-runs this cheap layer. `--no-deps` keeps the pinned
#     versions from layers 2-3 instead of letting setup.py bump them, and matches
#     the runtime install in amlt_draftopd.yaml. ---
COPY verl/ ./verl/
RUN cd verl && pip install --no-deps -e .

# Fail the build early if the core stack can't import (CPU-only check).
RUN python -c "import torch, verl; import flash_attn.bert_padding; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

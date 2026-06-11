# =============================================================================
# Draft-OPD runtime image for the AMLT / Singularity OPD training job.
#
#   * CUDA 12.8 devel base — matches the torch 2.9.1 (cu128) wheels the
#     sglang-dflash fork pins (sglang-dflash/python/pyproject.toml). Keeping the
#     base CUDA aligned with torch's bundled cu128 runtime avoids the libcudart
#     symbol mismatch that bites when a 12.4 system CUDA shadows torch's 12.8 libs.
#   * Ubuntu 24.04 -> Python 3.12 (the version the `draftopd` env uses).
#   * InfiniBand / RDMA userspace libs so NCCL can use the host IB fabric if you
#     ever scale past one node (the default job is single-node, 8 GPUs).
#   * Deps are baked from the in-repo sglang-dflash + verl (mirrors install.sh).
#     At job time amlt re-runs the editable installs from the uploaded checkout,
#     so the running code always matches the pushed commit.
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

# System deps: toolchain + Python 3.12 + GL/glib (opencv/vision) + IB userspace.
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

# --- torch FIRST (big, cacheable layer). PyPI's default torch 2.9.1 wheel for
#     linux x86_64 is the cu128 build, which is what the fork pins. ---
RUN pip install torch==2.9.1 torchvision torchaudio==2.9.1 torchao==0.9.0

# --- sglang-dflash fork: pulls the pinned runtime stack (sgl-kernel==0.3.21,
#     flashinfer_python==0.6.4, transformers==4.57.1, torch_memory_saver, ...). ---
COPY sglang-dflash/ ./sglang-dflash/
RUN cd sglang-dflash && pip install -e ./python && pip install cachetools

# --- verl (training framework). Plain editable install == install.sh; the
#     vllm / sglang extras are intentionally NOT requested (the editable
#     sglang-dflash fork above is the SGLang runtime). ---
COPY verl/ ./verl/
RUN cd verl && pip install -e .

# --- flash_attn: rmpad's `flash_attn.bert_padding` is hard-imported by the
#     trainer. Prefer the prebuilt cp312 / torch2.9 / cu12 wheel; fall back to a
#     (slow) source build if the wheel name ever drifts. ---
RUN pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    || MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation

# --- Re-pin the ABI-critical versions in case verl bumped them (sgl-kernel /
#     flashinfer are ABI-bound to torch 2.9 + transformers 4.57.1). ---
RUN pip install --no-deps torch==2.9.1 torchaudio==2.9.1 transformers==4.57.1

# --- math reward verifier used by verl validation. ---
RUN pip install math-verify latex2sympy2_extended

# Fail the build early if the core stack can't import (CPU-only check).
RUN python -c "import torch, verl; import flash_attn.bert_padding; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

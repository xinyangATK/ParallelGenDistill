#!/usr/bin/env bash
# =============================================================================
# One-time downloader for the Draft-OPD eval datasets — run BEFORE the eval.
#
# Builds the datasets straight into the HF datasets cache on the blob
# (${HF_HOME}/datasets), exactly like training writes its data to the blob. HF
# datasets' disk-space precheck misreads blobfuse's statvfs as 0 free (a false
# positive), so — same as verl's training path (verl/.../rl_dataset.py) — we disable
# that one precheck while building. It's confined to THIS one-time prep process; the
# eval/inference path stays patch-free and just reuses the cached build afterwards.
#
# Usage (anywhere the blob is mounted; amlt/amlt_eval.yaml runs this as step 1):
#   HF_HOME=/blob/.../cache/hf DATASETS_CSV="gsm8k:128,math500:128" \
#     bash prepare_eval_data.sh
#
# Inputs (env):
#   HF_HOME            HF cache root on the blob                 (REQUIRED)
#   DATASETS_CSV       comma list of dataset:N (N ignored here)  [gsm8k:128,math500:128]
#   HF_DATASETS_CACHE  blob datasets cache to fill               [${HF_HOME}/datasets]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # diffusion/dflash (has model/)
: "${HF_HOME:?Set HF_HOME to the blob HF cache root (e.g. /blob/<you>/cache/hf)}"
export DATASETS_CSV="${DATASETS_CSV:-gsm8k:128,math500:128}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"   # = blob

mkdir -p "${HF_DATASETS_CACHE}"
echo "[prep] building datasets straight into blob ${HF_DATASETS_CACHE} ..."
( cd "${SCRIPT_DIR}" && python - <<'PY'
import os
import datasets.builder as _b
# Same disable verl training uses: blobfuse statvfs falsely reports 0 free, and this
# precheck only runs while building (skipped on every later reuse).
_b.has_sufficient_disk_space = lambda *a, **k: True
from model import load_and_process_dataset   # same loader benchmark_sglang.py uses
for spec in os.environ["DATASETS_CSV"].split(","):
    name = spec.split(":")[0].strip()
    if not name:
        continue
    load_and_process_dataset(name)
    print(f"[prep] built: {name}")
PY
)
echo "[prep] done. eval runs now reuse ${HF_DATASETS_CACHE} (no patch needed there)."

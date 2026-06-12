#!/usr/bin/env python
"""One-shot pipeline: get the 4 source datasets and build the 16K OPD prompt pool.

Self-contained version of the original build_opd_pool.py. For each source it reads
a pre-downloaded local copy if present (so you can reuse an existing data dir and
skip all network), otherwise it downloads through `datasets`. Same recipe + schema
either way.

Recipe (fixed seed, random sampling):
    2K  GSM8K       openai/gsm8k                (main / train)
    5K  MATH        EleutherAI/hendrycks_math   (7 subjects, train+test) MINUS MATH-500
    4K  AoPS        AI-MO/aops
    5K  CodeAlpaca  sahil2801/CodeAlpaca-20k
  = 16K combined, shuffled.

Output schema (verl canonical; reference answers preserved):
    {
      "data_source": <hf repo id>,
      "prompt": [{"role": "user", "content": <raw problem/instruction>}],
      "ability": "math" | "code",
      "reward_model": {"style": "rule"|"code", "ground_truth": <final answer / reference output>},
      "extra_info": {"split": "train", "index": <int>, "source": <tag>,
                     "answer": <full raw reference solution/output>, ...source extras...}
    }
Prompts are RAW (no answer-format suffix); the trainer applies the chat template.

Local layout reused when present (under --raw-dir, default = output_dir):
    gsm8k/main/train-*.parquet
    math/<subject>/{train,test}-*.parquet
    aops/data/*.parquet
    codealpaca/code_alpaca_20k.json
    math500/test.jsonl  (key "problem")   OR   val/math500.jsonl  (verl schema)

Usage (one path argument):
    python data_prep/prepare_opd_pool.py /path/to/output_dir
    # reuses /path/to/output_dir/{gsm8k,math,aops,codealpaca,...} if already there,
    # else downloads (cached under /path/to/output_dir/hf_cache);
    # -> /path/to/output_dir/opd_pool_16k.jsonl  (+ manifest.json)

    # reuse an existing data dir but write the pool elsewhere:
    python data_prep/prepare_opd_pool.py /out --raw-dir /data/.../draftopd

Then train on it:
    TRAIN_JSONL=/path/to/output_dir/opd_pool_16k.jsonl \
      bash verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-ins.sh ...

Requires: `datasets` (already in verl/requirements.txt). Set HF_ENDPOINT to use a
HF mirror if downloading from a restricted network.
"""
import argparse
import glob
import json
import os
import random
import re
from collections import Counter

import numpy as np
from datasets import load_dataset

SEED = 42
N = {"gsm8k": 2000, "math": 5000, "aops": 4000, "codealpaca": 5000}
# alphabetical -> mirrors the original sorted(glob("math/*/*.parquet")) ordering
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry",
                 "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
MATH_CANDIDATES_EXPECTED = 12000  # 7500 train + 5000 test - 500 (MATH-500)


# ---- math answer extraction (inlined from verl.utils.reward_score.math_reward) ----
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx:right_brace_idx + 1]


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    assert s[:len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]


def boxed_or_empty(solution):
    try:
        b = last_boxed_only_string(str(solution))
        return remove_boxed(b) if b else ""
    except Exception:
        return ""


def gsm8k_final(answer):
    m = re.search(r"#### (\-?[0-9\.\,]+)", str(answer))
    return m.group(0).split("#### ")[1].replace(",", "") if m else ""


def jsonable(v):
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def main():
    ap = argparse.ArgumentParser(description="Get 4 sources (local-first) + build the 16K OPD pool.")
    ap.add_argument("output_dir", help="Where to write opd_pool_16k.jsonl + manifest.json.")
    ap.add_argument("--raw-dir", default=None,
                    help="Dir holding pre-downloaded sources to reuse (default: output_dir).")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cache-dir", default=None,
                    help="HF datasets cache for anything NOT found locally (default: <output_dir>/hf_cache).")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    raw_dir = os.path.abspath(args.raw_dir) if args.raw_dir else out_dir
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(out_dir, "hf_cache")

    rng = random.Random(args.seed)

    def sample_idx(n_total, k):
        assert k <= n_total, f"want {k} but only {n_total} available"
        return rng.sample(range(n_total), k)

    def local_files(*patterns):
        files = []
        for pat in patterns:
            files += glob.glob(os.path.join(raw_dir, pat))
        return sorted(set(files))

    def get_source(name, patterns, fmt, hf_repo, hf_config=None, split="train"):
        """Read local files if present, else download from HF. Returns a datasets.Dataset."""
        files = local_files(*patterns)
        if files:
            shown = os.path.relpath(files[0], raw_dir) + (" ..." if len(files) > 1 else "")
            print(f"      [{name}] local: {len(files)} file(s) -> {shown}", flush=True)
            return load_dataset(fmt, data_files=files, split="train", cache_dir=cache_dir)
        print(f"      [{name}] not found under {raw_dir} -> downloading from HF", flush=True)
        return load_dataset(hf_repo, hf_config, split=split, cache_dir=cache_dir)

    def get_math500_problems():
        p = os.path.join(raw_dir, "math500", "test.jsonl")
        if os.path.isfile(p):
            print("      [math500] local math500/test.jsonl", flush=True)
            return {json.loads(l)["problem"].strip() for l in open(p)}
        p = os.path.join(raw_dir, "val", "math500.jsonl")
        if os.path.isfile(p):
            print("      [math500] local val/math500.jsonl (verl schema)", flush=True)
            return {json.loads(l)["prompt"][0]["content"].strip() for l in open(p)}
        print("      [math500] not found locally -> downloading HuggingFaceH4/MATH-500", flush=True)
        return {str(x).strip() for x in load_dataset("HuggingFaceH4/MATH-500", split="test",
                                                     cache_dir=cache_dir)["problem"]}

    records = []

    # ---------------- GSM8K (2K from main/train) ----------------
    print("[1/4] GSM8K  openai/gsm8k (main/train)", flush=True)
    g = get_source("gsm8k", ["gsm8k/main/train-*.parquet"], "parquet", "openai/gsm8k", "main")
    for i in sample_idx(len(g), N["gsm8k"]):
        r = g[i]
        records.append({
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": str(r["question"])}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gsm8k_final(r["answer"])},
            "extra_info": {"source": "gsm8k", "answer": str(r["answer"])},
        })

    # ---------------- MATH (5K from train+test minus MATH-500) ----------------
    print("[2/4] MATH   EleutherAI/hendrycks_math (+ MATH-500 exclusion)", flush=True)
    m500 = get_math500_problems()
    math_rows = []
    for sub in MATH_SUBJECTS:
        for split in ("test", "train"):  # test < train -> matches original sorted-glob order
            ds = get_source(f"math/{sub}/{split}", [f"math/{sub}/{split}-*.parquet"],
                            "parquet", "EleutherAI/hendrycks_math", sub, split=split)
            for r in ds:
                if str(r["problem"]).strip() in m500:
                    continue  # drop held-out MATH-500
                math_rows.append((r, split, sub))
    print(f"      MATH candidates after MATH-500 exclusion: {len(math_rows)} "
          f"(original recipe expects {MATH_CANDIDATES_EXPECTED})", flush=True)
    if len(math_rows) != MATH_CANDIDATES_EXPECTED:
        print(f"      WARNING: candidate count != {MATH_CANDIDATES_EXPECTED}; upstream data may "
              f"have changed. Continuing with {len(math_rows)} candidates.", flush=True)
    for i in sample_idx(len(math_rows), N["math"]):
        r, split, sub = math_rows[i]
        records.append({
            "data_source": "EleutherAI/hendrycks_math",
            "prompt": [{"role": "user", "content": str(r["problem"])}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": boxed_or_empty(r["solution"])},
            "extra_info": {"source": "math", "answer": str(r["solution"]),
                           "level": jsonable(r.get("level")), "type": jsonable(r.get("type")),
                           "orig_split": split, "subject": sub},
        })

    # ---------------- AoPS (4K) ----------------
    print("[3/4] AoPS   AI-MO/aops", flush=True)
    a = get_source("aops", ["aops/data/*.parquet"], "parquet", "AI-MO/aops")
    for i in sample_idx(len(a), N["aops"]):
        r = a[i]
        records.append({
            "data_source": "AI-MO/aops",
            "prompt": [{"role": "user", "content": str(r["problem"])}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": boxed_or_empty(r["solution"])},
            "extra_info": {"source": "aops", "answer": str(r["solution"]),
                           "candidates": jsonable(r["candidates"]), "tags": jsonable(r["tags"]),
                           "metadata": jsonable(r["metadata"])},
        })

    # ---------------- CodeAlpaca (5K) ----------------
    print("[4/4] CodeAlpaca  sahil2801/CodeAlpaca-20k", flush=True)
    ca = get_source("codealpaca", ["codealpaca/code_alpaca_20k.json"], "json", "sahil2801/CodeAlpaca-20k")
    for i in sample_idx(len(ca), N["codealpaca"]):
        x = ca[i]
        instr, inp, out = str(x["instruction"]), str(x.get("input", "")), str(x["output"])
        content = f"{instr}\n\n{inp}".strip() if inp.strip() else instr
        records.append({
            "data_source": "sahil2801/CodeAlpaca-20k",
            "prompt": [{"role": "user", "content": content}],
            "ability": "code",
            "reward_model": {"style": "code", "ground_truth": out},
            "extra_info": {"source": "codealpaca", "answer": out, "instruction": instr, "input": inp},
        })

    # ---------------- shuffle + finalize index ----------------
    rng.shuffle(records)
    for idx, rec in enumerate(records):
        rec["extra_info"]["split"] = "train"
        rec["extra_info"]["index"] = idx

    out_path = os.path.join(out_dir, "opd_pool_16k.jsonl")
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------------- manifest ----------------
    src_counts = Counter(r["extra_info"]["source"] for r in records)
    gt_nonempty = Counter(r["extra_info"]["source"] for r in records if r["reward_model"]["ground_truth"])
    manifest = {
        "seed": args.seed, "total": len(records), "per_source_target": N,
        "per_source_actual": dict(src_counts),
        "ground_truth_nonempty_per_source": dict(gt_nonempty),
        "math_candidate_pool": len(math_rows), "math500_excluded": len(m500),
        "sources": {"gsm8k": "openai/gsm8k (main/train)",
                    "math": "EleutherAI/hendrycks_math (train+test minus MATH-500)",
                    "aops": "AI-MO/aops", "codealpaca": "sahil2801/CodeAlpaca-20k"},
        "output": out_path,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

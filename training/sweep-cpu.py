"""
sweep_cpu.py
────────────
Hyperparameter sweep for CPU-only models: fasttext and tfidf_logreg.
Drop this next to train.py and run it on any machine — no GPU needed.

Sweep grid:
    fasttext:     lr in [0.5, 2.0]         (2 runs)
    tfidf_logreg: C  in [1.0, 10.0]        (2 runs)

Usage:
    python3 sweep-cpu.py                    # sweep both models
    python3 sweep-cpu.py --model fasttext   # one model only
    python3 sweep-cpu.py --model tfidf_logreg
    python3 sweep-cpu.py --dry-run          # print configs, don't train
"""

import argparse
import copy
import os
import subprocess
import sys
import tempfile

import yaml

# ── Sweep grids ───────────────────────────────────────────────────────────────

FASTTEXT_LRS = [0.5, 1.0]
TFIDF_CS     = [1.0, 10.0]

BASE_CONFIG_PATH = "config.yaml"

# Resolve train.py relative to this script so it works from any working
# directory — repo root, training/, or inside Docker (WORKDIR=/app).
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train.py")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["fasttext", "tfidf_logreg", "all"],
        default="all",
        help="Which model(s) to sweep (default: all)",
    )
    p.add_argument("--config",  default=BASE_CONFIG_PATH)
    p.add_argument("--dry-run", action="store_true",
                   help="Print configs without running training")
    return p.parse_args()


def build_fasttext_config(base: dict, lr: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["model"] = "fasttext"
    cfg["fasttext"]["lr"] = lr
    return cfg


def build_tfidf_config(base: dict, C: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["model"] = "tfidf_logreg"
    cfg["tfidf_logreg"]["C"] = C
    return cfg


def run_training(config_path: str) -> int:
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT, "--config", config_path],
        check=False,
        # Run from the training/ directory so that
        # importlib.import_module("models.layer1.fasttext_model") resolves correctly.
        cwd=SCRIPT_DIR,
    )
    return result.returncode


def main():
    args = parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    # Build the full list of (label, config) pairs
    combos = []

    if args.model in ("fasttext", "all"):
        for lr in FASTTEXT_LRS:
            combos.append((f"fasttext  lr={lr}", build_fasttext_config(base_cfg, lr)))

    if args.model in ("tfidf_logreg", "all"):
        for C in TFIDF_CS:
            combos.append((f"tfidf_logreg  C={C}", build_tfidf_config(base_cfg, C)))

    print(f"[sweep-cpu] {len(combos)} run(s) planned")
    for i, (label, _) in enumerate(combos, 1):
        print(f"  [{i}/{len(combos)}] {label}")

    if args.dry_run:
        print("\n[sweep-cpu] --dry-run: exiting without training.")
        return

    failed = []
    for i, (label, cfg) in enumerate(combos, 1):
        print(f"\n{'='*60}")
        print(f"[sweep-cpu] Run {i}/{len(combos)}: {label}")
        print(f"{'='*60}")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            rc = run_training(tmp_path)
            if rc != 0:
                print(f"[sweep-cpu] ⚠  Run {i} exited with code {rc}")
                failed.append((label, rc))
        finally:
            os.unlink(tmp_path)

    print(f"\n[sweep-cpu] Done. {len(combos) - len(failed)}/{len(combos)} runs succeeded.")
    if failed:
        print("[sweep-cpu] Failed runs:")
        for label, rc in failed:
            print(f"  {label}  exit_code={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
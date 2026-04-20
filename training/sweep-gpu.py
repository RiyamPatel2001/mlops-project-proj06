"""
sweep.py
────────
Runs a hyperparameter sweep over transformer models (minilm, distilbert, mpnet)
by calling train.py once per config combination.

Each run gets its own config.yaml written to a temp file — no manual
editing required. All runs land in the same MLflow experiment so you can
compare them directly in the UI.

Usage:
    python3 sweep.py                          # sweeps all models
    python3 sweep.py --model minilm           # one model only
    python3 sweep.py --dry-run                # print configs, don't train

Sweep grid (9 runs total across all three models):
    learning_rate : 2e-5, 3e-5, 5e-5
    num_epochs    : 3  (fixed — change EPOCHS below to add 2-epoch runs)

To also sweep num_epochs, add values to the EPOCHS list.
"""

import argparse
import copy
import subprocess
import sys
import tempfile
import os

import yaml

# ── Paths — resolved relative to this script so it works regardless of cwd ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY   = os.path.join(SCRIPT_DIR, "train.py")

# ── Sweep grid ────────────────────────────────────────────────────────────────
LEARNING_RATES = [2e-5, 3e-5, 5e-5]
EPOCHS         = [3]           # add 2 here to also sweep num_epochs

MODELS = ["minilm"] #, "distilbert", "mpnet"]

BASE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   choices=MODELS + ["all"], default="all",
                   help="Which model(s) to sweep (default: all)")
    p.add_argument("--config",  default=BASE_CONFIG_PATH,
                   help="Base config.yaml to build sweep configs from")
    p.add_argument("--dry-run", action="store_true",
                   help="Print configs without running training")
    return p.parse_args()


def build_config(base: dict, model: str, lr: float, epochs: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["model"] = model
    cfg[model]["learning_rate"] = lr
    cfg[model]["num_epochs"]    = epochs
    return cfg


def run_training(config_path: str) -> int:
    result = subprocess.run(
        [sys.executable, TRAIN_PY, "--config", config_path],
        check=False,
    )
    return result.returncode


def main():
    args = parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    models_to_sweep = MODELS if args.model == "all" else [args.model]

    combos = [
        (model, lr, epochs)
        for model  in models_to_sweep
        for lr     in LEARNING_RATES
        for epochs in EPOCHS
    ]

    print(f"[sweep] {len(combos)} run(s) planned")
    for i, (model, lr, epochs) in enumerate(combos, 1):
        print(f"  [{i}/{len(combos)}] model={model}  lr={lr}  epochs={epochs}")

    if args.dry_run:
        print("\n[sweep] --dry-run: exiting without training.")
        return

    failed = []
    for i, (model, lr, epochs) in enumerate(combos, 1):
        print(f"\n{'='*60}")
        print(f"[sweep] Run {i}/{len(combos)}: model={model}  lr={lr}  epochs={epochs}")
        print(f"{'='*60}")

        cfg = build_config(base_cfg, model, lr, epochs)

        # Write a temp config for this run so train.py can read it normally
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            rc = run_training(tmp_path)
            if rc != 0:
                print(f"[sweep] ⚠  Run {i} exited with code {rc}")
                failed.append((model, lr, epochs, rc))
        finally:
            os.unlink(tmp_path)

    print(f"\n[sweep] Done. {len(combos) - len(failed)}/{len(combos)} runs succeeded.")
    if failed:
        print("[sweep] Failed runs:")
        for model, lr, epochs, rc in failed:
            print(f"  model={model}  lr={lr}  epochs={epochs}  exit_code={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
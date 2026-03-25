"""
preprocess.py
─────────────
Payee normalization, user-stratified train/val split, and class weight
computation for the MoneyData transaction categorizer.

Usage (standalone):
    python3 preprocess.py --data_path data/raw/transactions.csv \
                          --output_dir data/processed/ \
                          --val_frac 0.2 \
                          --random_state 42

Outputs:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/label_classes.json   ← ordered list of category strings
    data/processed/class_weights.json   ← {label_index: weight, ...}
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# ── 1. Payee normalization ────────────────────────────────────────────────────

def normalize_payee(s: str) -> str:
    """
    Collapse merchant variants to a canonical form.
    Steps (order matters):
      1. Uppercase + strip
      2. Remove store/branch numbers  (#7548, # 03, etc.)
      3. Remove platform order suffixes  (*ABCDEF (Uber, Amazon, etc.))
      4. Remove trailing digits left after step 2
      5. Collapse whitespace
    """
    s = s.upper().strip()
    s = re.sub(r'#\s*\d+', '', s)        # #7548 → ''
    s = re.sub(r'\*\S+', '', s)           # *9XJUY0 → ''
    s = re.sub(r'\s\d+$', '', s)          # trailing lone digits
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ── 2. User-stratified split (Strategy D) ────────────────────────────────────

def user_stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by user_id so no user appears in both train and val.

    Rationale: simulates the new-user generalisation scenario — the model
    must categorise transactions for people it has never seen.  This is the
    honest eval for a shared Layer-1 model (Strategy D from EDA).

    Returns:
        df_train, df_val
    """
    rng = np.random.default_rng(random_state)
    all_users = df["user_id"].unique()
    n_val = int(len(all_users) * val_frac)
    val_users = set(rng.choice(all_users, size=n_val, replace=False))

    df_train = df[~df["user_id"].isin(val_users)].reset_index(drop=True)
    df_val   = df[ df["user_id"].isin(val_users)].reset_index(drop=True)
    return df_train, df_val


# ── 3. Class weight computation ───────────────────────────────────────────────

def compute_weights(
    labels: np.ndarray,
    classes: np.ndarray,
) -> dict[int, float]:
    """
    Compute balanced class weights using sklearn's formula:
        weight_i = total / (n_classes * count_i)

    Args:
        labels:  integer-encoded label array (from LabelEncoder)
        classes: sorted unique class indices (le.transform(le.classes_))

    Returns:
        {label_index: weight}  — ready to pass to LogisticRegression or
        to build a torch.Tensor for CrossEntropyLoss.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


# ── 4. Main pipeline ──────────────────────────────────────────────────────────

def run(
    data_path: str,
    output_dir: str,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Load
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")

    # Normalize payee
    df["payee_norm"] = df["payee"].apply(normalize_payee)

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])

    # Split
    df_train, df_val = user_stratified_split(df, val_frac, random_state)
    print(f"Split → train: {len(df_train):,} rows ({df_train['user_id'].nunique()} users) | "
          f"val: {len(df_val):,} rows ({df_val['user_id'].nunique()} users)")

    # Class weights (computed on training set only)
    classes = np.arange(len(le.classes_))
    weights = compute_weights(df_train["label"].values, classes)

    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    print(f"Saved → {train_path}, {val_path}")

    # Save label classes (ordered list — index == label integer)
    classes_path = os.path.join(output_dir, "label_classes.json")
    with open(classes_path, "w") as f:
        json.dump(le.classes_.tolist(), f, indent=2)
    print(f"Saved → {classes_path}  ({len(le.classes_)} categories)")

    # Save class weights
    weights_path = os.path.join(output_dir, "class_weights.json")
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Saved → {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MoneyData for training.")
    parser.add_argument("--data_path",    default="data/raw/transactions.csv")
    parser.add_argument("--output_dir",   default="data/processed/")
    parser.add_argument("--val_frac",     type=float, default=0.2)
    parser.add_argument("--random_state", type=int,   default=42)
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        output_dir=args.output_dir,
        val_frac=args.val_frac,
        random_state=args.random_state,
    )